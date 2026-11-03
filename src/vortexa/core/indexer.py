"""Main CodebaseIndexer — orchestrates chunking, embedding, and search.

Inspired by cocoindex's incremental processing model:
- Memoization: skip embedding if chunk content unchanged (content + code hash)
- Lineage: every chunk traces back to its exact source location
- Declarative state: compute desired state, reconcile with existing
- Incremental BM25: add/remove from index instead of full rebuild
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import lmdb
import numpy as np

from vortexa.core.chunking import chunk_source
from vortexa.core.embedding import Embedder
from vortexa.core.graph import RepoGraph
from vortexa.core.language import detect_language, get_extensions
from vortexa.core.progress import ProgressCallback, emit_progress
from vortexa.core.types import (
    Chunk,
    ChunkConfig,
    Encoder,
    IndexStats,
    SearchMode,
    SearchResult,
)
from vortexa.search.search import search as _search
from vortexa.storage.bm25 import BM25Index
from vortexa.storage.vector_store import VectorStore
from vortexa.storage.walker import walk_files

logger = logging.getLogger(__name__)

_MAX_FILE_BYTES = 1_000_000  # 1 MB
_DEFAULT_MODEL_ID = "VTXAI/vtx-embed-7M-lf2"
_DEFAULT_EMBED_BATCH_SIZE = 64

# Hash of the chunking logic for memoization — bump when chunking changes
_CHUNKING_LOGIC_VERSION = "2"

_GLOBAL_INDEX_ROOT = Path.home() / ".vortexa"


def global_index_root() -> Path:
    """Return the directory where all indexes are stored."""
    return _GLOBAL_INDEX_ROOT


def _resolve_model_id(model_id: str | None) -> str:
    """Resolve model aliases to a concrete ID, defaulting to the LF2 mini model."""
    from vortexa.core.inference import resolve_model_id

    return resolve_model_id(model_id or _DEFAULT_MODEL_ID)


def _index_dir_for_root(root: Path, model_id: str | None = None) -> Path:
    """Derive the global index directory for a project root and embedding model.

    Stores under ``~/.vortexa/<hex_hash>/`` where *hex_hash* is the first
    16 chars of the SHA-256 of the resolved absolute path plus the resolved
    model id. The model id is part of the key because vectors from different
    embedding models are not interchangeable.
    """
    resolved_root = Path(root).resolve()
    resolved_model = _resolve_model_id(model_id)
    key = f"{resolved_root}\x00{resolved_model}"
    path_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
    return _GLOBAL_INDEX_ROOT / path_hash


def _file_hash(path: Path) -> str:
    """Compute SHA256 hash of file contents."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _compute_chunking_code_hash() -> str:
    """Hash of the chunking logic for memoization (cocoindex-style code fingerprinting)."""
    return hashlib.sha256(_CHUNKING_LOGIC_VERSION.encode()).hexdigest()[:8]


class CodebaseIndexer:
    """Persistent, incremental codebase indexer with memoization and lineage.

    Inspired by cocoindex's declarative model:
    - Target = F(Source): declare what should exist, engine reconciles
    - Memoization: skip expensive operations when inputs unchanged
    - Lineage: every output traces to its exact source byte range
    """

    def __init__(
        self,
        root: str | Path,
        model: Encoder | Embedder | None = None,
        model_id: str | None = None,
        index_dir: str | Path | None = None,
        chunk_config: ChunkConfig | None = None,
        embed_batch_size: int = _DEFAULT_EMBED_BATCH_SIZE,
    ) -> None:
        """Initialize the indexer.

        :param root: Root directory of the codebase to index.
        :param model: Embedding model (Encoder or Embedder). Created from model_id if None.
        :param model_id: Model ID for auto-creating the embedder (ignored if model provided).
        :param index_dir: Directory for persistent storage. Defaults to ~/.vortexa/<hash>/.
        :param chunk_config: Chunking configuration.
        :param embed_batch_size: Number of chunks embedded per batch. Smaller
            batches give smoother progress updates at a small throughput cost.
        """
        self.root = Path(root).resolve()
        if model is not None:
            model_ref = (
                getattr(model, "model_id", None)
                or getattr(model, "_model_id", None)
                or model_id
            )
            self.model_id = str(model_ref or _DEFAULT_MODEL_ID)
        else:
            self.model_id = _resolve_model_id(model_id)
        self.index_dir = (
            Path(index_dir) if index_dir else _index_dir_for_root(self.root, self.model_id)
        )
        self.chunk_config = chunk_config or ChunkConfig()
        self.embed_batch_size = max(1, int(embed_batch_size))

        # Resolve model: prefer explicit model, else create VortexEmbedderV4
        if model is not None:
            if isinstance(model, Embedder):
                self._embedder: Embedder | None = model
                self._model = model
            else:
                self._embedder = None
                self._model = model
        else:
            from vortexa.core.model import VortexEmbedderV4

            self._embedder = VortexEmbedderV4(self.model_id)
            self._model = self._embedder

        # In-memory state
        self.chunks: list[Chunk] = []
        self.chunk_ids: list[str] = []
        self.file_hashes: dict[str, str] = {}  # relative path -> SHA256
        self.chunk_memo: dict[str, str] = {}  # chunk_id -> chunk_hash (for memoization)
        self._vector_store: VectorStore | None = None
        self._bm25_index: BM25Index | None = None
        self._repo_graph: RepoGraph | None = None
        self._state_loaded = False

        # Incremental state
        self._include_text_files = False
        self._indexing = False
        self._last_index_started: float | None = None
        self._last_index_finished: float | None = None
        self._last_index_error: str | None = None
        self._last_stats: IndexStats | None = None
        self._lock = threading.RLock()

        # Stats for the current run
        self._memo_hits = 0
        self._memo_misses = 0

    def _get_dim(self) -> int:
        """Get embedding dimensionality."""
        if self._embedder is not None:
            return self._embedder.dim
        return getattr(self._model, 'dim', 256)

    @property
    def include_text_files(self) -> bool:
        """Whether the current index includes Markdown/JSON/YAML-style files."""
        return self._include_text_files

    @property
    def index_exists(self) -> bool:
        """Whether a persisted index exists for this root and model."""
        return (self.index_dir / "state.lmdb").exists() and (
            self.index_dir / "vectors.npy"
        ).exists()

    @property
    def is_indexing(self) -> bool:
        """Whether this indexer is currently building an index."""
        return self._indexing

    @property
    def last_index_error(self) -> str | None:
        """Error message from the most recent failed index run, if any."""
        return self._last_index_error

    @property
    def last_index_stats(self) -> IndexStats | None:
        """Statistics from the most recent successful index run."""
        return self._last_stats

    @property
    def vector_store(self) -> VectorStore:
        """Get or create the vector store."""
        if self._vector_store is None:
            dim = self._get_dim()
            self._vector_store = VectorStore(dim=dim)
        return self._vector_store

    @property
    def bm25_index(self) -> BM25Index:
        """Get or create the BM25 index."""
        if self._bm25_index is None:
            self._bm25_index = BM25Index(index_dir=self.index_dir / "bm25")
        return self._bm25_index

    def index(
        self,
        force: bool = False,
        include_text_files: bool | None = None,
        chunk_config: ChunkConfig | None = None,
        progress: ProgressCallback | None = None,
    ) -> IndexStats:
        """Build or incrementally update the index.

        Memoization reuses vectors for chunks whose content is unchanged, even
        when the surrounding file or line numbers move.

        :param force: If True, rebuild everything from scratch.
        :param include_text_files: Index .md, .yaml, .json, etc. When omitted,
            reuse the setting from the previous run (or ``False``).
        :param chunk_config: Override chunking config for this index run.
        :param progress: Optional callback receiving realtime progress events.
        :return: Index statistics.
        """
        with self._lock:
            if self._indexing:
                raise RuntimeError("an index run is already in progress")
            if include_text_files is None:
                include_text_files = self._include_text_files
            self._include_text_files = bool(include_text_files)
            self._indexing = True
            self._last_index_error = None
            self._last_index_started = time.time()
            try:
                stats = self._index_impl(
                    force=force,
                    include_text_files=bool(include_text_files),
                    chunk_config=chunk_config,
                    progress=progress,
                )
                self._last_stats = stats
                return stats
            except Exception as exc:
                self._last_index_error = f"{type(exc).__name__}: {exc}"
                # Force the next run to reload the last good on-disk state
                # instead of trusting partially-updated in-memory state.
                self._state_loaded = False
                raise
            finally:
                self._indexing = False
                self._last_index_finished = time.time()

    def _index_impl(
        self,
        *,
        force: bool,
        include_text_files: bool,
        chunk_config: ChunkConfig | None,
        progress: ProgressCallback | None,
    ) -> IndexStats:
        _t0 = time.perf_counter()

        def elapsed() -> float:
            return time.perf_counter() - _t0

        config = chunk_config or self.chunk_config
        extensions = get_extensions(include_text_files)
        self._memo_hits = 0
        self._memo_misses = 0
        self._repo_graph = None
        errors: list[str] = []

        if force:
            self._reset_in_memory()
        else:
            self._ensure_loaded()

        emit_progress(
            progress,
            "scan",
            message="Discovering files",
            detail=str(self.root),
            elapsed=elapsed(),
        )
        candidates: list[Path] = []
        for file_path in walk_files(self.root, extensions):
            try:
                if file_path.stat().st_size > _MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            candidates.append(file_path)

        total_files = len(candidates)
        current_files: dict[str, str] = {}
        for index, file_path in enumerate(candidates, 1):
            try:
                rel = str(file_path.relative_to(self.root))
                current_files[rel] = _file_hash(file_path)
            except OSError as exc:
                errors.append(f"{file_path}: {exc}")
                continue
            if index == total_files or index % 25 == 0:
                emit_progress(
                    progress,
                    "scan",
                    current=index,
                    total=total_files,
                    message="Scanning files",
                    detail=rel,
                    elapsed=elapsed(),
                )
        emit_progress(
            progress,
            "scan",
            current=total_files,
            total=total_files,
            message="Scanning files",
            elapsed=elapsed(),
        )

        # Determine changed files.
        if force or not self.file_hashes:
            changed_files = set(current_files)
            removed_files: set[str] = set()
        else:
            changed_files = {
                rel
                for rel, digest in current_files.items()
                if self.file_hashes.get(rel) != digest
            }
            removed_files = set(self.file_hashes) - set(current_files)

        if not changed_files and not removed_files:
            self._memo_hits = len(self.chunks)
            self._memo_misses = 0
            stats = self._make_stats(
                elapsed=elapsed(),
                up_to_date=True,
                changed_files=0,
                removed_files=0,
                embedded_chunks=0,
                errors=errors,
            )
            logger.info("No files changed, index is up to date")
            emit_progress(
                progress,
                "done",
                current=1,
                total=1,
                message=(
                    f"Index up to date: {stats.total_chunks} chunks "
                    f"in {stats.indexed_files} files"
                ),
                elapsed=elapsed(),
            )
            return stats

        logger.info(
            "Indexing: %d changed, %d removed, %d total files",
            len(changed_files),
            len(removed_files),
            len(current_files),
        )

        # Snapshot reusable vectors by content hash before removing old chunks.
        reuse_vectors: dict[str, np.ndarray] = {}
        if not force and self._vector_store is not None:
            for cid, chunk in zip(self.chunk_ids, self.chunks, strict=False):
                digest = chunk.chunk_hash
                if not digest or digest in reuse_vectors:
                    continue
                vector = self._vector_store.get_vector(cid)
                if vector is not None:
                    reuse_vectors[digest] = vector

        self._remove_file_chunks(changed_files | removed_files)

        # Chunk changed files and decide which chunks can reuse old vectors.
        new_chunks: list[Chunk] = []
        new_chunk_ids: list[str] = []
        reused_rows: list[tuple[int, np.ndarray]] = []
        embed_rows: list[int] = []
        embed_texts: list[str] = []

        changed_sorted = sorted(changed_files)
        total_changed = len(changed_sorted)
        for file_index, rel in enumerate(changed_sorted, 1):
            file_path = self.root / rel
            try:
                source = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                # Drop the file hash so the next run retries it instead of
                # treating a failed read as an empty, unchanged file.
                current_files.pop(rel, None)
                errors.append(f"{rel}: {exc}")
                continue
            language = detect_language(file_path)
            try:
                chunks = chunk_source(source, rel, language, config)
            except Exception as exc:  # pragma: no cover - defensive
                current_files.pop(rel, None)
                errors.append(f"{rel}: chunking failed: {exc}")
                continue

            for chunk in chunks:
                cid = hashlib.sha256(
                    f"{chunk.file_path}:{chunk.start_line}:{chunk.end_line}:{chunk.content}".encode()
                ).hexdigest()[:16]
                row = len(new_chunks)
                new_chunks.append(chunk)
                new_chunk_ids.append(cid)

                digest = chunk.chunk_hash
                reusable = reuse_vectors.get(digest) if (digest and not force) else None
                if reusable is not None:
                    reused_rows.append((row, reusable))
                    self._memo_hits += 1
                else:
                    embed_rows.append(row)
                    embed_texts.append(chunk.content)
                    self._memo_misses += 1

            if file_index == total_changed or file_index % 10 == 0:
                emit_progress(
                    progress,
                    "chunk",
                    current=file_index,
                    total=total_changed,
                    message="Chunking code",
                    detail=rel,
                    elapsed=elapsed(),
                )
        emit_progress(
            progress,
            "chunk",
            current=total_changed,
            total=total_changed,
            message="Chunking code",
            elapsed=elapsed(),
        )

        # Add chunks to in-memory state before embedding so a failed save or
        # vector write keeps the in-memory index usable for this process.
        self.chunks.extend(new_chunks)
        self.chunk_ids.extend(new_chunk_ids)
        self.chunk_memo = {
            cid: chunk.chunk_hash
            for cid, chunk in zip(self.chunk_ids, self.chunks, strict=False)
            if chunk.chunk_hash
        }

        # Embed only chunks that could not reuse an existing vector.
        if new_chunks:
            total_embed = len(embed_texts)
            if total_embed:
                emit_progress(
                    progress,
                    "model",
                    current=0,
                    total=1,
                    message="Loading model",
                    elapsed=elapsed(),
                )
                dim = self._get_dim()
                emit_progress(
                    progress,
                    "model",
                    current=1,
                    total=1,
                    message="Loading model",
                    elapsed=elapsed(),
                )
                embeddings = np.empty((len(new_chunks), dim), dtype=np.float32)
                for row, vector in reused_rows:
                    embeddings[row] = vector

                emit_progress(
                    progress,
                    "embed",
                    current=0,
                    total=total_embed,
                    message="Embedding chunks",
                    elapsed=elapsed(),
                )
                batch_size = self.embed_batch_size
                for start in range(0, total_embed, batch_size):
                    end = min(start + batch_size, total_embed)
                    batch = embed_texts[start:end]
                    batch_vectors = self._encode_texts(batch)
                    rows = embed_rows[start:end]
                    embeddings[rows] = np.asarray(batch_vectors, dtype=np.float32)
                    emit_progress(
                        progress,
                        "embed",
                        current=end,
                        total=total_embed,
                        message="Embedding chunks",
                        detail="",
                        elapsed=elapsed(),
                    )
            else:
                emit_progress(
                    progress,
                    "embed",
                    current=0,
                    total=0,
                    message="Embedding chunks (all reused)",
                    elapsed=elapsed(),
                )
                dim = self._get_dim()
                embeddings = np.empty((len(new_chunks), dim), dtype=np.float32)
                for row, vector in reused_rows:
                    embeddings[row] = vector

            logger.info(
                "Adding %d chunks to vector store (%d reused, %d embedded)",
                len(new_chunks),
                len(reused_rows),
                len(embed_texts),
            )
            self.vector_store.add(embeddings, new_chunk_ids)
        else:
            emit_progress(
                progress,
                "embed",
                current=0,
                total=0,
                message="No new chunks to embed",
                elapsed=elapsed(),
            )

        # BM25 is rebuilt from the reconciled chunk list. It is fast compared
        # with embedding and guarantees the persisted keyword index stays in
        # sync with chunk IDs.
        emit_progress(
            progress,
            "bm25",
            current=0,
            total=len(self.chunks),
            message="Building keyword index",
            elapsed=elapsed(),
        )
        bm25 = BM25Index(index_dir=self.index_dir / "bm25")
        bm25.build(
            self.chunks,
            self.chunk_ids,
            persist_dir=self.index_dir / "bm25",
            progress=lambda current, total: emit_progress(
                progress,
                "bm25",
                current=current,
                total=total,
                message="Building keyword index",
                elapsed=elapsed(),
            ),
        )
        self._bm25_index = bm25
        self.file_hashes = current_files

        emit_progress(
            progress,
            "save",
            current=0,
            total=1,
            message="Saving index",
            elapsed=elapsed(),
        )
        self._save_state()
        emit_progress(
            progress,
            "save",
            current=1,
            total=1,
            message="Saving index",
            elapsed=elapsed(),
        )

        stats = self._make_stats(
            elapsed=elapsed(),
            up_to_date=False,
            changed_files=len(changed_files),
            removed_files=len(removed_files),
            embedded_chunks=len(embed_texts),
            errors=errors,
        )
        logger.info(
            "Index complete: %d files, %d chunks (%d reused, %d embedded) in %.1fms",
            stats.indexed_files,
            stats.total_chunks,
            stats.memo_hits,
            stats.embedded_chunks,
            stats.index_time_ms,
        )
        emit_progress(
            progress,
            "done",
            current=1,
            total=1,
            message=(
                f"{stats.indexed_files} files · {stats.total_chunks} chunks · "
                f"{stats.memo_hits} reused · {stats.embedded_chunks} embedded"
            ),
            elapsed=elapsed(),
        )
        return stats

    def load(self) -> bool:
        """Load the persisted index from disk.

        :return: True when an index was loaded successfully.
        """
        with self._lock:
            self._state_loaded = False
            loaded = self._load_state()
            return loaded

    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Search with hybrid semantic + BM25 retrieval.

        :param query: Natural-language query or symbol name.
        :param top_k: Maximum number of results.
        :param alpha: Semantic weight (0.0–1.0); adaptive when omitted.
        :param min_score: Drop results below this score. The default of ``0``
            still removes zero-score noise.
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        with self._lock:
            self._ensure_loaded()
            if not self.chunks:
                return []
            results = _search(
                query=query,
                model=self._model,
                store=self.vector_store,
                bm25_index=self.bm25_index,
                chunks=self.chunks,
                chunk_ids=self.chunk_ids,
                top_k=top_k,
                alpha=alpha,
            )
            return [
                result
                for result in results
                if np.isfinite(result.score) and result.score > 0 and result.score >= min_score
            ]

    def status(self) -> dict:
        """Return a machine-readable snapshot of index state for agents."""
        with self._lock:
            self._ensure_loaded()
            stats = self._compute_stats()
            selected = self._last_stats or stats
            last_indexed = None
            if self._last_index_finished is not None:
                last_indexed = datetime.fromtimestamp(
                    self._last_index_finished, tz=timezone.utc
                ).isoformat()
            elif self.index_exists:
                last_indexed = datetime.fromtimestamp(
                    (self.index_dir / "state.lmdb").stat().st_mtime, tz=timezone.utc
                ).isoformat()

            if self._indexing:
                state = "indexing"
            elif self._last_index_error:
                state = "error"
            elif selected.total_chunks > 0:
                state = "ready"
            else:
                state = "empty"

            payload = {
                "status": state,
                "root": str(self.root),
                "index_dir": str(self.index_dir),
                "model": self.model_id,
                "index_exists": self.index_exists,
                "indexed_files": selected.indexed_files,
                "total_chunks": selected.total_chunks,
                "languages": selected.languages,
                "include_text_files": self._include_text_files,
                "is_indexing": self._indexing,
                "last_indexed_at": last_indexed,
                "last_index_error": self._last_index_error,
                "index_time_ms": selected.index_time_ms,
                "memo_hits": selected.memo_hits,
                "memo_misses": selected.memo_misses,
                "up_to_date": (
                    self._last_stats.up_to_date if self._last_stats is not None else None
                ),
                "errors": list(selected.errors),
            }
            return payload

    # ── Context resolution ──────────────────────────────────────────────

    def _build_repo_graph(self) -> RepoGraph:
        """Build (and cache) a repo graph from indexed Python files."""
        if self._repo_graph is not None:
            return self._repo_graph
        from vortexa.core.graph import RepoGraphBuilder

        builder = RepoGraphBuilder()
        files: dict[str, str] = {}
        for rel in self.file_hashes:
            file_path = self.root / rel
            if file_path.suffix == ".py":
                try:
                    files[rel] = file_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    pass
        self._repo_graph = builder.build(files)
        return self._repo_graph

    def _find_test_files(self, primary_files: set[str]) -> list[str]:
        test_files: list[str] = []
        for file_path in primary_files:
            path = Path(file_path)
            stem = path.stem
            suffix = path.suffix
            parent = path.parent
            candidates = [
                parent / f"test_{stem}{suffix}",
                parent / f"{stem}_test{suffix}",
            ]
            for candidate in candidates:
                candidate_str = str(candidate)
                if candidate_str != file_path and candidate_str not in test_files:
                    if candidate_str in self.file_hashes:
                        test_files.append(candidate_str)
        return test_files

    def _find_imports_importers(
        self, primary_files: set[str], graph: RepoGraph
    ) -> tuple[list[str], list[str]]:
        imports: list[str] = []
        imported_by: list[str] = []
        for file_path in primary_files:
            file_node = graph.find_file_node(file_path)
            if file_node is None:
                continue
            for edge in graph.edges_from(file_node.id, kind="IMPORTS"):
                target = graph.nodes.get(edge.dst)
                if target and target.path and target.path not in primary_files:
                    if target.path not in imports:
                        imports.append(target.path)
            for edge in graph.edges_from(file_node.id, kind="IMPORTS_FROM"):
                target = graph.nodes.get(edge.dst)
                if target and target.path and target.path not in primary_files:
                    if target.path not in imports:
                        imports.append(target.path)
            for edge in graph.edges_to(file_node.id, kind="IMPORTS"):
                source = graph.nodes.get(edge.src)
                if source and source.path and source.path not in primary_files:
                    if source.path not in imported_by:
                        imported_by.append(source.path)
            for edge in graph.edges_to(file_node.id, kind="IMPORTS_FROM"):
                source = graph.nodes.get(edge.src)
                if source and source.path and source.path not in primary_files:
                    if source.path not in imported_by:
                        imported_by.append(source.path)
        return imports, imported_by

    def _find_symbols(self, primary_files: set[str], graph: RepoGraph) -> list[dict]:
        symbols: list[dict] = []
        seen: set[str] = set()
        skip_kinds = {"file"}
        for file_path in primary_files:
            for node_id in graph._file_symbols.get(file_path, set()):
                if node_id in seen:
                    continue
                seen.add(node_id)
                node = graph.nodes.get(node_id)
                if not node or node.kind in skip_kinds:
                    continue
                symbols.append({
                    "name": node.name,
                    "kind": node.kind,
                    "file": node.path,
                    "line": node.line,
                })
        return symbols[:15]

    def _find_callers_callees(
        self, primary_files: set[str], graph: RepoGraph
    ) -> tuple[list[dict], list[dict]]:
        callers: list[dict] = []
        callees: list[dict] = []
        for file_path in primary_files:
            for node_id in graph._file_symbols.get(file_path, set()):
                node = graph.nodes.get(node_id)
                if not node:
                    continue
                for edge in graph.edges_from(node_id, kind="CALLS"):
                    target = graph.nodes.get(edge.dst)
                    if target and target.path != file_path:
                        callees.append({
                            "name": target.name,
                            "file": target.path,
                            "line": target.line,
                        })
                        break
                for edge in graph.edges_to(node_id, kind="CALLS"):
                    source = graph.nodes.get(edge.src)
                    if source and source.path != file_path:
                        callers.append({
                            "name": source.name,
                            "file": source.path,
                            "line": source.line,
                        })
                        break
        return callers[:10], callees[:10]

    def _find_dependency_chain(
        self, primary_files: set[str], graph: RepoGraph, depth: int = 1
    ) -> list[str]:
        chain: list[str] = []
        visited: set[str] = set(primary_files)
        for file_path in primary_files:
            file_node = graph.find_file_node(file_path)
            if file_node is None:
                continue
            expanded = graph.expand([file_node.id], max_hops=depth, max_size=50)
            for nid, hop in expanded:
                if hop == 0:
                    continue
                node = graph.nodes.get(nid)
                if node and node.path and node.path not in visited:
                    visited.add(node.path)
                    chain.append(node.path)
        return chain[:10]

    def _expand_context(
        self, query: str, primary: list[SearchResult], graph: RepoGraph
    ) -> dict:
        primary_files = {r.chunk.file_path for r in primary}
        test_files = self._find_test_files(primary_files)
        imports, imported_by = self._find_imports_importers(primary_files, graph)
        symbols = self._find_symbols(primary_files, graph)
        callers, callees = self._find_callers_callees(primary_files, graph)
        dependency_chain = self._find_dependency_chain(primary_files, graph, depth=1)

        scores = [r.score for r in primary if r.score > 0]
        confidence = sum(scores) / len(scores) if scores else 0.0

        related_files = list(
            primary_files | set(imports) | set(imported_by) | set(test_files)
        )

        return {
            "query": query,
            "confidence": round(confidence, 3),
            "primary_chunks": primary,
            "related_files": related_files,
            "test_files": test_files,
            "imports": imports,
            "imported_by": imported_by,
            "symbols": symbols,
            "callers": callers,
            "callees": callees,
            "dependency_chain": dependency_chain,
            "total_tokens": sum(len(r.chunk.content) for r in primary) // 4,
        }

    def _compress_pack(self, pack: dict) -> dict:
        if not pack.get("primary_chunks"):
            return pack
        pack = dict(pack)
        pack["primary_chunks"] = pack["primary_chunks"][:10]
        pack["related_files"] = pack["related_files"][:5]
        pack["test_files"] = pack["test_files"][:3]
        pack["imports"] = pack["imports"][:2]
        pack["imported_by"] = pack["imported_by"][:1]
        pack["symbols"] = pack["symbols"][:5]
        pack["callers"] = pack["callers"][:2]
        pack["callees"] = pack["callees"][:2]
        pack["dependency_chain"] = pack["dependency_chain"][:3]
        pack["total_tokens"] = sum(len(r.chunk.content) for r in pack["primary_chunks"]) // 4
        return pack

    def _empty_pack(self, query: str) -> dict:
        return {
            "query": query,
            "confidence": 0.0,
            "primary_chunks": [],
            "related_files": [],
            "test_files": [],
            "imports": [],
            "imported_by": [],
            "symbols": [],
            "callers": [],
            "callees": [],
            "dependency_chain": [],
            "total_tokens": 0,
        }

    def resolve(self, query: str, top_k: int = 5) -> dict:
        """Full context resolution: search + expand + compress."""
        with self._lock:
            self._ensure_loaded()
            if not self.chunks:
                return self._empty_pack(query)
            primary = self.search(query, top_k=top_k)
            if not primary:
                return self._empty_pack(query)
            graph = self._build_repo_graph()
            pack = self._expand_context(query, primary, graph)
            return self._compress_pack(pack)

    def _resolve_indexed_file(self, location: str) -> str | None:
        """Map a user-supplied path to an indexed repo-relative file path."""
        normalized = location.strip().replace("\\", "/")
        if normalized.startswith("./"):
            normalized = normalized[2:]
        if normalized in self.file_hashes:
            return normalized
        try:
            absolute = Path(location).expanduser()
            if absolute.is_absolute():
                relative = absolute.resolve().relative_to(self.root).as_posix()
                if relative in self.file_hashes:
                    return relative
        except (OSError, ValueError):
            pass
        suffix_matches = [
            rel
            for rel in self.file_hashes
            if rel == normalized or rel.endswith("/" + normalized)
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        return None

    def explain(self, code_location: str) -> dict:
        """Explain a code location: find symbol, expand context, return pack."""
        if not code_location or not code_location.strip():
            raise ValueError("code_location must not be empty")
        query = code_location.strip()
        with self._lock:
            self._ensure_loaded()

            file_path = self._resolve_indexed_file(query)
            if file_path is not None:
                chunks_here = [c for c in self.chunks if c.file_path == file_path]
                if chunks_here:
                    primary = [
                        SearchResult(chunk=chunk, score=1.0, source=SearchMode.SEMANTIC)
                        for chunk in chunks_here
                    ]
                    graph = self._build_repo_graph()
                    pack = self._expand_context(query, primary, graph)
                    return self._compress_pack(pack)

            if ":" in query:
                parts = query.rsplit(":", 1)
                if parts[0].endswith(
                    (".py", ".js", ".ts", ".rs", ".go", ".java", ".jsx", ".tsx")
                ):
                    file_path = self._resolve_indexed_file(parts[0])
                    try:
                        line = int(parts[1])
                    except ValueError:
                        line = 0
                    if file_path is not None:
                        chunks_here = [
                            c for c in self.chunks if c.file_path == file_path
                        ]
                        if line > 0:
                            chunks_here = [
                                c
                                for c in chunks_here
                                if c.start_line <= line <= c.end_line
                            ]
                        if chunks_here:
                            primary = [
                                SearchResult(
                                    chunk=chunk,
                                    score=1.0,
                                    source=SearchMode.SEMANTIC,
                                )
                                for chunk in chunks_here
                            ]
                            graph = self._build_repo_graph()
                            pack = self._expand_context(query, primary, graph)
                            return self._compress_pack(pack)
            return self.resolve(query, top_k=3)

    def format_context(self, pack: dict) -> str:
        """Format a context pack as a human-readable string for agents."""
        lines = [f"[{pack['confidence']:.2f}] {pack['query']}"]
        if pack.get("primary_chunks"):
            lines.append(f"  ({len(pack['primary_chunks'])} files)")
        lines.append("")
        for i, result in enumerate(pack.get("primary_chunks", []), 1):
            chunk = result.chunk
            span = (
                f"{chunk.start_line}-{chunk.end_line}"
                if chunk.end_line != chunk.start_line
                else str(chunk.start_line)
            )
            lines.append(f"  {i}. {chunk.file_path}:{span}  [{result.score:.2f}]")
            snippet = chunk.content.strip()[:300]
            for line in snippet.split("\n"):
                lines.append(f"     {line}")
        if pack.get("symbols"):
            names = [s["name"] for s in pack["symbols"][:6]]
            if len(pack["symbols"]) > 6:
                names.append("...")
            lines.append(f"  sym: {' '.join(names)}")
        hints = []
        if pack.get("test_files"):
            hints.append(
                f"tests: {', '.join(f.split('/')[-1] for f in pack['test_files'][:2])}"
            )
        if pack.get("imports"):
            hints.append(
                f"deps: {', '.join(f.split('/')[-1] for f in pack['imports'][:2])}"
            )
        if pack.get("callers"):
            hints.append(
                f"callers: {', '.join(c['name'] for c in pack['callers'][:2])}"
            )
        if hints:
            lines.append(f"  {' | '.join(hints)}")
        return "\n".join(lines)

    def find_related(self, chunk_idx: int, top_k: int = 5) -> list[SearchResult]:
        """Find chunks semantically similar to a given chunk (cocoindex find_related)."""
        with self._lock:
            self._ensure_loaded()
            if chunk_idx < 0 or chunk_idx >= len(self.chunks):
                return []
            chunk = self.chunks[chunk_idx]
            # Use the chunk's content as the query and exclude the source chunk.
            return self.search(chunk.content, top_k=top_k + 1)[1:top_k + 1]

    def find_symbol(self, name: str, top_k: int = 10) -> list[SearchResult]:
        """Find code that defines a symbol by name."""
        return self.search(name, top_k=top_k, alpha=0.0)

    def stats(self) -> IndexStats:
        """Get index statistics, loading the persisted index if needed."""
        with self._lock:
            self._ensure_loaded()
            return self._compute_stats()

    def clear(self) -> None:
        """Delete the persistent index from disk and clear in-memory state."""
        import shutil

        with self._lock:
            if self.index_dir.exists():
                shutil.rmtree(self.index_dir)
            self._reset_in_memory()
            self._state_loaded = False
            self._last_index_started = None
            self._last_index_finished = None
            self._last_index_error = None
            self._last_stats = None

    # ── Internal ──────────────────────────────────────────────────────

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode texts using the model (handles both Encoder and Embedder)."""
        if self._embedder is not None:
            result = self._embedder.embed_batch(texts)
            return np.array(result, dtype=np.float32)
        else:
            result = cast(Encoder, self._model).encode(texts)
            return np.array(result, dtype=np.float32)

    def _remove_file_chunks(self, file_rels: set[str]) -> None:
        """Remove all chunks belonging to the given files."""
        if not file_rels:
            return
        remove_ids = [
            cid
            for cid, chunk in zip(self.chunk_ids, self.chunks, strict=False)
            if chunk.file_path in file_rels
        ]
        self.chunks = [c for c in self.chunks if c.file_path not in file_rels]
        self.chunk_ids = [cid for cid in self.chunk_ids if cid not in remove_ids]
        self.chunk_memo = {
            cid: digest
            for cid, digest in self.chunk_memo.items()
            if cid not in remove_ids
        }

        if remove_ids and self._vector_store is not None:
            self._vector_store.remove(remove_ids)
        self._repo_graph = None

    def _compute_stats(self) -> IndexStats:
        """Compute index statistics (language counts are files per language)."""
        files: set[str] = set()
        file_languages: dict[str, str | None] = {}
        for chunk in self.chunks:
            files.add(chunk.file_path)
            file_languages.setdefault(chunk.file_path, chunk.language)
        languages: Counter[str] = Counter(
            language for language in file_languages.values() if language
        )
        return IndexStats(
            indexed_files=len(files),
            total_chunks=len(self.chunks),
            languages=dict(languages),
            memo_hits=self._memo_hits,
            memo_misses=self._memo_misses,
        )

    def _make_stats(
        self,
        *,
        elapsed: float,
        up_to_date: bool,
        changed_files: int,
        removed_files: int,
        embedded_chunks: int,
        errors: list[str],
    ) -> IndexStats:
        """Build an IndexStats object for the current run."""
        stats = self._compute_stats()
        return IndexStats(
            indexed_files=stats.indexed_files,
            total_chunks=stats.total_chunks,
            languages=stats.languages,
            memo_hits=self._memo_hits,
            memo_misses=self._memo_misses,
            index_time_ms=round(elapsed * 1000, 1),
            changed_files=changed_files,
            removed_files=removed_files,
            embedded_chunks=embedded_chunks,
            up_to_date=up_to_date,
            errors=tuple(errors),
        )

    def _reset_in_memory(self) -> None:
        """Drop all in-memory index state (does not touch disk)."""
        self.chunks = []
        self.chunk_ids = []
        self.file_hashes = {}
        self.chunk_memo = {}
        self._vector_store = None
        self._bm25_index = None
        self._repo_graph = None
        self._memo_hits = 0
        self._memo_misses = 0

    def _ensure_loaded(self) -> None:
        """Load persisted state once, if this indexer has not done so already."""
        if not self._state_loaded:
            self._load_state()

    def _save_state(self) -> None:
        """Persist index state to LMDB."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        lmdb_path = self.index_dir / "state.lmdb"
        env = lmdb.open(str(lmdb_path), map_size=256 * 1024 * 1024, max_dbs=10)
        try:
            self.vector_store.save(self.index_dir, env)

            fh_db = env.open_db(b"file_hashes")
            c_db = env.open_db(b"chunks")
            cm_db = env.open_db(b"chunk_memo")
            meta_db = env.open_db(b"index_meta")
            with env.begin(write=True) as txn:
                for db in (fh_db, c_db, cm_db, meta_db):
                    for key, _ in txn.cursor(db=db):
                        txn.delete(key, db=db)

                for path, digest in self.file_hashes.items():
                    txn.put(path.encode(), digest.encode(), db=fh_db)
                for cid, chunk in zip(self.chunk_ids, self.chunks, strict=False):
                    val = json.dumps(
                        [
                            chunk.file_path,
                            chunk.content,
                            chunk.start_line,
                            chunk.end_line,
                            chunk.language,
                            chunk.chunk_hash,
                        ]
                    )
                    txn.put(cid.encode(), val.encode(), db=c_db)
                for cid, digest in self.chunk_memo.items():
                    txn.put(cid.encode(), digest.encode(), db=cm_db)
                txn.put(b"model_id", self.model_id.encode(), db=meta_db)
                txn.put(
                    b"include_text_files",
                    b"1" if self._include_text_files else b"0",
                    db=meta_db,
                )
                txn.put(
                    b"chunking_logic_version",
                    _CHUNKING_LOGIC_VERSION.encode(),
                    db=meta_db,
                )
        finally:
            env.close()
        self._state_loaded = True

    def _load_state(self) -> bool:
        """Load persisted index state. Returns True if successful."""
        self._state_loaded = True
        lmdb_path = self.index_dir / "state.lmdb"
        if not lmdb_path.exists():
            return False

        env = lmdb.open(str(lmdb_path), map_size=256 * 1024 * 1024, max_dbs=10)
        try:
            fh_db = env.open_db(b"file_hashes")
            c_db = env.open_db(b"chunks")
            cm_db = env.open_db(b"chunk_memo")
            meta_db = env.open_db(b"index_meta")

            with env.begin() as txn:
                self.file_hashes = {}
                with txn.cursor(db=fh_db) as cursor:
                    for key, value in cursor:
                        self.file_hashes[bytes(key).decode()] = bytes(value).decode()

                self.chunk_ids = []
                self.chunks = []
                with txn.cursor(db=c_db) as cursor:
                    for key, value in cursor:
                        cid = bytes(key).decode()
                        data = json.loads(bytes(value).decode())
                        self.chunk_ids.append(cid)
                        self.chunks.append(
                            Chunk(
                                file_path=data[0],
                                content=data[1],
                                start_line=data[2],
                                end_line=data[3],
                                language=data[4],
                                chunk_hash=data[5],
                            )
                        )

                self.chunk_memo = {}
                with txn.cursor(db=cm_db) as cursor:
                    for key, value in cursor:
                        self.chunk_memo[bytes(key).decode()] = bytes(value).decode()

                include_text = txn.get(b"include_text_files", db=meta_db)
                if include_text is not None:
                    self._include_text_files = bytes(include_text) == b"1"
                stored_model = txn.get(b"model_id", db=meta_db)
                if stored_model is not None:
                    stored_model_id = bytes(stored_model).decode()
                    if stored_model_id != self.model_id:
                        logger.warning(
                            "Index at %s was built with model %s, not %s",
                            self.index_dir,
                            stored_model_id,
                            self.model_id,
                        )

            loaded_store = VectorStore.load(self.index_dir, env)
            if loaded_store is None or loaded_store.size != len(self.chunks):
                logger.warning(
                    "Ignoring incompatible vector store at %s (vectors=%s, chunks=%d)",
                    self.index_dir,
                    None if loaded_store is None else loaded_store.size,
                    len(self.chunks),
                )
                self._reset_in_memory()
                return False
            if any(not loaded_store.has_id(cid) for cid in self.chunk_ids):
                logger.warning("Ignoring vector store with missing chunk vectors at %s", self.index_dir)
                self._reset_in_memory()
                return False

            self._vector_store = loaded_store
            bm25 = BM25Index(index_dir=self.index_dir / "bm25")
            if bm25.load(self.index_dir / "bm25"):
                self._bm25_index = bm25
            else:
                self._bm25_index = None
            self._repo_graph = None
        finally:
            env.close()

        return bool(self.chunks)
