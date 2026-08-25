"""Main CodebaseIndexer — orchestrates chunking, embedding, and search.

Inspired by cocoindex's incremental processing model:
- Memoization: skip embedding if chunk content unchanged (content + code hash)
- Lineage: every chunk traces back to its exact source location
- Declarative state: compute desired state, reconcile with existing
- Incremental BM25: add/remove from index instead of full rebuild
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import cast

import lmdb
import numpy as np

from vortexa.core.chunking import chunk_source
from vortexa.core.embedding import Embedder
from vortexa.core.graph import RepoGraph
from vortexa.core.language import detect_language, get_extensions
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

# Hash of the chunking logic for memoization — bump when chunking changes
_CHUNKING_LOGIC_VERSION = "2"

_GLOBAL_INDEX_ROOT = Path.home() / ".vortexa"


def _index_dir_for_root(root: Path) -> Path:
    """Derive the global index directory for a given project root.

    Stores under ``~/.vortexa/<hex_hash>/`` where *hex_hash* is the first
    16 chars of the SHA-256 of the resolved absolute path.  This keeps every
    indexed project isolated while avoiding path-length issues.
    """
    path_hash = hashlib.sha256(str(root).encode()).hexdigest()[:16]
    return _GLOBAL_INDEX_ROOT / path_hash


def _file_hash(path: Path) -> str:
    """Compute SHA256 hash of file contents."""
    h = hashlib.sha256()
    try:
        h.update(path.read_bytes())
    except OSError:
        h.update(b"")
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
        model_id: str = "VTXAI/vtx-embed-7M",
        index_dir: str | Path | None = None,
        chunk_config: ChunkConfig | None = None,
    ) -> None:
        """Initialize the indexer.

        :param root: Root directory of the codebase to index.
        :param model: Embedding model (Encoder or Embedder). Created from model_id if None.
        :param model_id: Model ID for auto-creating embedder (ignored if model provided).
        :param index_dir: Directory for persistent storage. Defaults to ~/.vortexa/<hash>/.
        :param chunk_config: Chunking configuration.
        """
        self.root = Path(root).resolve()
        self.index_dir = Path(index_dir) if index_dir else _index_dir_for_root(self.root)
        self.chunk_config = chunk_config or ChunkConfig()

        # Resolve model: prefer explicit model, else create VortexEmbedderV4
        if model is not None:
            if isinstance(model, Embedder):
                self._embedder: Embedder | None = model
                self._model = model
            else:
                self._embedder = None
                self._model = model
        else:
            from vortexa.core.v4_embedder import VortexEmbedderV4
            self._embedder = VortexEmbedderV4(model_id or "VTXAI/vtx-embed-7M")
            self._model = self._embedder

        # In-memory state
        self.chunks: list[Chunk] = []
        self.chunk_ids: list[str] = []
        self.file_hashes: dict[str, str] = {}  # relative path -> SHA256
        self.chunk_memo: dict[str, str] = {}  # chunk_id -> chunk_hash (for memoization)
        self._vector_store: VectorStore | None = None
        self._bm25_index: BM25Index | None = None
        self._repo_graph: RepoGraph | None = None

        # Stats
        self._memo_hits = 0
        self._memo_misses = 0

    def _get_dim(self) -> int:
        """Get embedding dimensionality."""
        if self._embedder is not None:
            return self._embedder.dim
        return getattr(self._model, 'dim', 256)

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
        include_text_files: bool = False,
        chunk_config: ChunkConfig | None = None,
    ) -> IndexStats:
        """Build or incrementally update the index.

        Memoization: only re-embeds chunks whose content changed.
        Declarative state: computes desired state and reconciles with existing.

        :param force: If True, re-index everything from scratch.
        :param include_text_files: If True, also index .md, .yaml, .json, etc.
        :param chunk_config: Override chunking config for this index run.
        :return: Index statistics.
        """
        _t0 = time.perf_counter()
        config = chunk_config or self.chunk_config
        extensions = get_extensions(include_text_files)

        # Load existing state if not forcing
        if not force:
            self._load_state()

        # Walk files and compute hashes
        current_files: dict[str, str] = {}
        for file_path in walk_files(self.root, extensions):
            if file_path.stat().st_size > _MAX_FILE_BYTES:
                continue
            rel = str(file_path.relative_to(self.root))
            current_files[rel] = _file_hash(file_path)

        # Determine changed files
        if force or not self.file_hashes:
            changed_files = set(current_files.keys())
            removed_files = set()
        else:
            changed_files = {
                rel
                for rel, h in current_files.items()
                if self.file_hashes.get(rel) != h
            }
            removed_files = set(self.file_hashes.keys()) - set(current_files.keys())

        if not changed_files and not removed_files:
            logger.info("No files changed, index is up to date")
            # All existing chunks are memo hits (nothing changed)
            self._memo_hits = len(self.chunks)
            self._memo_misses = 0
            return self._compute_stats()

        logger.info(
            "Indexing: %d changed, %d removed, %d total files",
            len(changed_files),
            len(removed_files),
            len(current_files),
        )

        # Remove chunks from deleted/changed files
        self._remove_file_chunks(changed_files | removed_files)

        # Chunk changed files and apply memoization
        new_chunks: list[Chunk] = []
        memo_skipped: list[Chunk] = []

        for rel in changed_files:
            file_path = self.root / rel
            language = detect_language(file_path)
            with contextlib.suppress(OSError):
                source = file_path.read_text(encoding="utf-8", errors="replace")
                chunks = chunk_source(source, rel, language, config)

                for chunk in chunks:
                    # Memoization check: skip if chunk_hash matches existing
                    if chunk.chunk_hash and not force:
                        existing_memo = self._get_memo_for_hash(chunk.chunk_hash)
                        if existing_memo:
                            memo_skipped.append(chunk)
                            self._memo_hits += 1
                            continue

                    self._memo_misses += 1
                    new_chunks.append(chunk)

        # Assign chunk IDs (include line range to avoid collisions on duplicate content)
        new_chunk_ids = []
        for chunk in new_chunks:
            cid = hashlib.sha256(
                f"{chunk.file_path}:{chunk.start_line}:{chunk.end_line}:{chunk.content}".encode()
            ).hexdigest()[:16]
            new_chunk_ids.append(cid)

        # Add to in-memory state
        self.chunks.extend(new_chunks)
        self.chunk_ids.extend(new_chunk_ids)
        self.file_hashes = current_files

        # Update memo cache
        for cid, chunk in zip(new_chunk_ids, new_chunks, strict=False):
            if chunk.chunk_hash:
                self.chunk_memo[cid] = chunk.chunk_hash

        # Embed new chunks (memoization means we skip unchanged ones)
        if new_chunks:
            logger.info("Embedding %d new chunks (%d memo hits)...", len(new_chunks), self._memo_hits)
            texts = [c.content for c in new_chunks]
            embeddings = self._encode_texts(texts)
            self.vector_store.add(embeddings, new_chunk_ids)

        # BM25: incremental update
        logger.info("Updating BM25 index...")
        bm25 = BM25Index(index_dir=self.index_dir / "bm25")
        bm25.build(self.chunks, self.chunk_ids, persist_dir=self.index_dir / "bm25")
        self._bm25_index = bm25

        # Save state
        self._save_state()

        elapsed = time.perf_counter() - _t0
        stats = self._compute_stats()
        stats = IndexStats(
            indexed_files=stats.indexed_files,
            total_chunks=stats.total_chunks,
            languages=stats.languages,
            memo_hits=stats.memo_hits,
            memo_misses=stats.memo_misses,
            index_time_ms=round(elapsed * 1000, 1),
        )
        logger.info(
            "Index complete: %d files, %d chunks (%d memo hits) in %.1fms",
            stats.indexed_files,
            stats.total_chunks,
            stats.memo_hits,
            stats.index_time_ms,
        )
        return stats

    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float | None = None,
    ) -> list[SearchResult]:
        """Search with hybrid semantic + BM25 retrieval."""
        if not self.chunks:
            return []
        return _search(
            query=query,
            model=self._model,
            store=self.vector_store,
            bm25_index=self.bm25_index,
            chunks=self.chunks,
            chunk_ids=self.chunk_ids,
            top_k=top_k,
            alpha=alpha,
        )

    # ── Context resolution ──────────────────────────────────────────────

    def _build_repo_graph(self) -> RepoGraph:
        """Build a repo graph from indexed Python files."""
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
        return builder.build(files)

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
        if not self.chunks:
            return self._empty_pack(query)
        primary = self.search(query, top_k=top_k)
        if not primary:
            return self._empty_pack(query)
        graph = self._build_repo_graph()
        pack = self._expand_context(query, primary, graph)
        return self._compress_pack(pack)

    def explain(self, code_location: str) -> dict:
        """Explain a code location: find symbol, expand context, return pack."""
        query = code_location
        if ":" in code_location:
            parts = code_location.rsplit(":", 1)
            if parts[0].endswith(
                (".py", ".js", ".ts", ".rs", ".go", ".java", ".jsx", ".tsx")
            ):
                file_path = parts[0]
                try:
                    line = int(parts[1])
                except ValueError:
                    line = 0
                chunks_here = [c for c in self.chunks if c.file_path == file_path]
                if line > 0:
                    chunks_here = [
                        c for c in chunks_here if c.start_line <= line <= c.end_line
                    ]
                if chunks_here:
                    primary = [
                        SearchResult(
                            chunk=chunks_here[0], score=1.0, source=SearchMode.SEMANTIC
                        )
                    ]
                    graph = self._build_repo_graph()
                    pack = self._expand_context(query, primary, graph)
                    return self._compress_pack(pack)
        return self.resolve(code_location, top_k=3)

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
        if chunk_idx < 0 or chunk_idx >= len(self.chunks):
            return []

        chunk = self.chunks[chunk_idx]
        # Use the chunk's content as the query
        return self.search(chunk.content, top_k=top_k + 1)[1:top_k + 1]  # Exclude self

    def find_symbol(self, name: str, top_k: int = 10) -> list[SearchResult]:
        """Find code that defines a symbol by name."""
        return self.search(name, top_k=top_k, alpha=0.0)

    def stats(self) -> IndexStats:
        """Get index statistics."""
        return self._compute_stats()

    def clear(self) -> None:
        """Delete the persistent index from disk and clear in-memory state."""
        import shutil

        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
        self.chunks.clear()
        self.chunk_ids.clear()
        self.file_hashes.clear()
        self.chunk_memo.clear()
        self._vector_store = None
        self._bm25_index = None
        self._memo_hits = 0
        self._memo_misses = 0

    # ── Internal ──────────────────────────────────────────────────────

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode texts using the model (handles both Encoder and Embedder)."""
        if self._embedder is not None:
            result = self._embedder.embed_batch(texts)
            return np.array(result, dtype=np.float32)
        else:
            result = cast(Encoder, self._model).encode(texts)
            return np.array(result, dtype=np.float32)

    def _get_memo_for_hash(self, chunk_hash: str) -> str | None:
        """Check if a chunk_hash exists in the memo cache."""
        for cid, memo_hash in self.chunk_memo.items():
            if memo_hash == chunk_hash:
                return cid
        return None

    def _remove_file_chunks(self, file_rels: set[str]) -> None:
        """Remove all chunks belonging to the given files."""
        if not file_rels:
            return
        remove_ids = [
            cid for cid, c in zip(self.chunk_ids, self.chunks, strict=False)
            if c.file_path in file_rels
        ]
        self.chunks = [c for c in self.chunks if c.file_path not in file_rels]
        self.chunk_ids = [cid for cid in self.chunk_ids if cid not in remove_ids]

        if remove_ids and self._vector_store is not None:
            self._vector_store.remove(remove_ids)
        elif not self.chunks:
            self._vector_store = VectorStore(dim=self._get_dim())

    def _compute_stats(self) -> IndexStats:
        """Compute index statistics."""
        files = set()
        languages: Counter[str] = Counter()
        for chunk in self.chunks:
            files.add(chunk.file_path)
            if chunk.language:
                languages[chunk.language] += 1
        return IndexStats(
            indexed_files=len(files),
            total_chunks=len(self.chunks),
            languages=dict(languages),
            memo_hits=self._memo_hits,
            memo_misses=self._memo_misses,
        )

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
            with env.begin(write=True) as txn:
                for key, _ in txn.cursor(db=fh_db):
                    txn.delete(key, db=fh_db)
                for key, _ in txn.cursor(db=c_db):
                    txn.delete(key, db=c_db)
                for key, _ in txn.cursor(db=cm_db):
                    txn.delete(key, db=cm_db)

                for path, h in self.file_hashes.items():
                    txn.put(path.encode(), h.encode(), db=fh_db)
                for cid, chunk in zip(self.chunk_ids, self.chunks, strict=False):
                    val = json.dumps(
                        [chunk.file_path, chunk.content, chunk.start_line, chunk.end_line, chunk.language, chunk.chunk_hash]
                    )
                    txn.put(cid.encode(), val.encode(), db=c_db)
                for cid, h in self.chunk_memo.items():
                    txn.put(cid.encode(), h.encode(), db=cm_db)
        finally:
            env.close()

    def _load_state(self) -> bool:
        """Load persisted index state. Returns True if successful."""
        lmdb_path = self.index_dir / "state.lmdb"
        if not lmdb_path.exists():
            return False

        env = lmdb.open(str(lmdb_path), map_size=256 * 1024 * 1024, max_dbs=10)
        try:
            fh_db = env.open_db(b"file_hashes")
            c_db = env.open_db(b"chunks")
            cm_db = env.open_db(b"chunk_memo")

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

            self._vector_store = VectorStore.load(self.index_dir, env)

            bm25 = BM25Index(index_dir=self.index_dir / "bm25")
            if bm25.load(self.index_dir / "bm25"):
                self._bm25_index = bm25
            else:
                self._bm25_index = None
        finally:
            env.close()

        return bool(self.chunks)
