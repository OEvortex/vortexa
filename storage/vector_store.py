"""NumPy-based vector store with LMDB persistence for the codebase indexer.

Stores embedding vectors as a NumPy matrix and persists mappings to LMDB.
Supports incremental add/remove and cosine similarity search.
"""

from __future__ import annotations

from pathlib import Path

import lmdb
import numpy as np
import numpy.typing as npt

_LMDB_MAP_SIZE = 10 * 1024 * 1024
_LMDB_MAX_DBS = 10


class VectorStore:
    """In-memory vector store backed by NumPy with LMDB persistence.

    Vectors are stored as a float32 matrix. Cosine similarity search
    uses normalized dot product for cosine similarity.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self._vectors: npt.NDArray[np.float32] = np.empty((0, dim), dtype=np.float32)
        self._id_to_idx: dict[str, int] = {}
        self._idx_to_id: dict[int, str] = {}

    @property
    def size(self) -> int:
        return len(self._vectors)

    def add(self, vectors: npt.NDArray[np.float32], ids: list[str]) -> None:
        if len(vectors) == 0:
            return
        n_existing = len(self._vectors)
        self._vectors = np.vstack([self._vectors, vectors]) if n_existing > 0 else vectors.copy()
        for i, cid in enumerate(ids):
            idx = n_existing + i
            self._id_to_idx[cid] = idx
            self._idx_to_id[idx] = cid

    def remove(self, ids: list[str]) -> None:
        if not ids:
            return
        remove_indices = {self._id_to_idx[cid] for cid in ids if cid in self._id_to_idx}
        if not remove_indices:
            return

        keep = [(idx, cid) for cid, idx in self._id_to_idx.items() if idx not in remove_indices]

        if keep:
            keep_indices = [idx for idx, _ in keep]
            self._vectors = self._vectors[keep_indices]
            self._id_to_idx = {cid: i for i, (_, cid) in enumerate(keep)}
            self._idx_to_id = {i: cid for i, (_, cid) in enumerate(keep)}
        else:
            self._vectors = np.empty((0, self.dim), dtype=np.float32)
            self._id_to_idx.clear()
            self._idx_to_id.clear()

    def rebuild(self, vectors: npt.NDArray[np.float32], ids: list[str]) -> None:
        self._vectors = vectors.copy() if len(vectors) > 0 else np.empty((0, self.dim), dtype=np.float32)
        self._id_to_idx = dict(zip(ids, range(len(ids)), strict=False))
        self._idx_to_id = dict(zip(range(len(ids)), ids, strict=False))

    def get_vector(self, chunk_id: str) -> npt.NDArray[np.float32] | None:
        idx = self._id_to_idx.get(chunk_id)
        if idx is None:
            return None
        return self._vectors[idx]

    def query(
        self,
        query_vector: npt.NDArray[np.float32],
        k: int = 10,
        selector: npt.NDArray[np.int_] | None = None,
    ) -> list[tuple[int, float]]:
        if len(self._vectors) == 0:
            return []

        effective_k = min(k, len(self._vectors))
        if selector is not None:
            effective_k = min(effective_k, len(selector))

        if selector is not None:
            selected_vectors = self._vectors[selector]
            q_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
            v_norms = selected_vectors / (np.linalg.norm(selected_vectors, axis=1, keepdims=True) + 1e-10)
            similarities = v_norms @ q_norm
            distances = 1.0 - similarities

            if effective_k >= len(distances):
                sorted_idx = np.argsort(distances)
            else:
                partitioned = np.argpartition(distances, kth=effective_k - 1)[:effective_k]
                sorted_idx = partitioned[np.argsort(distances[partitioned])]

            return [(int(selector[i]), float(distances[i])) for i in sorted_idx]
        else:
            q_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
            v_norms = self._vectors / (np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-10)
            similarities = v_norms @ q_norm
            distances = 1.0 - similarities

            if effective_k >= len(distances):
                sorted_idx = np.argsort(distances)
            else:
                partitioned = np.argpartition(distances, kth=effective_k - 1)[:effective_k]
                sorted_idx = partitioned[np.argsort(distances[partitioned])]

            return [(int(i), float(distances[i])) for i in sorted_idx]

    def save(self, directory: Path, env: lmdb.Environment | None = None) -> None:
        """Persist vectors and ID mappings to disk.

        :param directory: Directory to save in.
        :param env: Optional shared LMDB environment. If None, creates its own.
        """
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self._vectors)

        if env is None:
            env = lmdb.open(str(directory / "state.lmdb"), map_size=_LMDB_MAP_SIZE, max_dbs=_LMDB_MAX_DBS)
            try:
                self._write_lmdb(env)
            finally:
                env.close()
        else:
            self._write_lmdb(env)

    def _write_lmdb(self, env: lmdb.Environment) -> None:
        vm_db = env.open_db(b"vector_map")
        meta_db = env.open_db(b"meta")
        with env.begin(write=True) as txn:
            for key, _ in txn.cursor(db=vm_db):
                txn.delete(key, db=vm_db)
            for key, _ in txn.cursor(db=meta_db):
                txn.delete(key, db=meta_db)
            for cid, idx in self._id_to_idx.items():
                txn.put(cid.encode(), str(idx).encode(), db=vm_db)
            txn.put(b"dim", str(self.dim).encode(), db=meta_db)

    @classmethod
    def load(cls, directory: Path, env: lmdb.Environment | None = None) -> VectorStore | None:
        """Load a persisted vector store.

        :param directory: Directory to load from.
        :param env: Optional shared LMDB environment. If None, creates its own.
        """
        vectors_path = directory / "vectors.npy"
        if not vectors_path.exists():
            return None

        if env is None:
            lmdb_path = directory / "state.lmdb"
            if not lmdb_path.exists():
                return None
            env = lmdb.open(str(lmdb_path), map_size=_LMDB_MAP_SIZE, max_dbs=_LMDB_MAX_DBS)
            try:
                return cls._read_lmdb(env, vectors_path)
            finally:
                env.close()
        else:
            return cls._read_lmdb(env, vectors_path)

    @classmethod
    def _read_lmdb(cls, env: lmdb.Environment, vectors_path: Path) -> VectorStore:
        meta_db = env.open_db(b"meta")
        vm_db = env.open_db(b"vector_map")

        with env.begin() as txn:
            dim_bytes = txn.get(b"dim", db=meta_db)
            dim = int(bytes(dim_bytes).decode()) if dim_bytes else 256

        store = cls(dim=dim)
        store._vectors = np.load(vectors_path)

        with env.begin() as txn:
            with txn.cursor(db=vm_db) as cursor:
                for key, value in cursor:
                    cid = bytes(key).decode()
                    idx = int(bytes(value).decode())
                    store._id_to_idx[cid] = idx
                    store._idx_to_id[idx] = cid

        return store
