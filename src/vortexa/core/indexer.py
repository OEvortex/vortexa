"""Main CodebaseIndexer — orchestrates chunking, embedding, parsing, graph building, and search.

V2: Integrates knowledge graph, multi-level indexing, path intelligence,
structural retrieval, context expansion, and Vortex Score.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import lmdb
import numpy as np
from vortexa.core.chunking import chunk_source
from vortexa.core.embedding import Embedder, LF4Embedder
from vortexa.core.graph import KnowledgeGraph, build_graph_from_symbols
from vortexa.core.language import detect_language, get_extensions
from vortexa.core.parser import parse_symbols
from vortexa.core.types import (
    Chunk,
    ChunkConfig,
    ContextPack,
    Encoder,
    GraphContext,
    GraphPath,
    GraphTraversalMode,
    IndexStats,
    SearchResult,
    VortexScoreConfig,
)
from vortexa.search.context_compressor import compress as _compress_pack, format_for_agent
from vortexa.search.context_expansion import expand_context
from vortexa.search.path_scorer import path_retrieve as _path_retrieve
from vortexa.search.search import search as _search
from vortexa.search.vortex_score import compute_vortex_score, load_weights
from vortexa.storage.bm25 import BM25Index
from vortexa.storage.session_memory import SessionMemory
from vortexa.storage.vector_store import VectorStore
from vortexa.storage.walker import walk_files

logger = logging.getLogger(__name__)

_MAX_FILE_BYTES = 1_000_000
_CHUNKING_LOGIC_VERSION = "2"


@dataclass(frozen=True, slots=True)
class SearchResultWithContext:
    """A SearchResult paired with compact hybrid graph context.

    Created by `CodebaseIndexer._attach_hybrid_context` when `hybrid=True`
    is passed to `search()`. The `result` carries the original chunk +
    score; `context` carries a `GraphContext` (key_symbol + one
    structural incoming edge + one structural outgoing edge) that the
    MCP formatter renders alongside the search hit.
    """

    result: SearchResult
    context: GraphContext

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - delegating
        # Delegate unknown attribute access to the underlying SearchResult
        # so callers (and downstream formatters) can treat either object
        # uniformly — `r.chunk`, `r.score`, `r.source`, `r.context` all work.
        return getattr(self.result, name)


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    try:
        h.update(path.read_bytes())
    except OSError:
        h.update(b"")
    return h.hexdigest()


def _compute_chunking_code_hash() -> str:
    return hashlib.sha256(_CHUNKING_LOGIC_VERSION.encode()).hexdigest()[:8]


class CodebaseIndexer:
    """Persistent, incremental codebase indexer with V2 context engine integration.

    V2 additions:
    - Knowledge graph for structural relationships
    - Multi-level indexing (file, function, symbol)
    - Path intelligence
    - Context expansion and compression (ContextPack)
    - Vortex Score (learned weighted fusion)
    - Session memory
    """

    def __init__(
        self,
        root: str | Path,
        model: Encoder | Embedder | None = None,
        model_id: str = "VTXAI/Vortex-Embed-4.7M",
        index_dir: str | Path | None = None,
        chunk_config: ChunkConfig | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.index_dir = Path(index_dir) if index_dir else self.root / ".jarvis" / "index"
        self.chunk_config = chunk_config or ChunkConfig()

        if model is not None:
            if isinstance(model, Embedder):
                self._embedder: Embedder | None = model
                self._model = model
            else:
                self._embedder = None
                self._model = model
        else:
            self._embedder = LF4Embedder(model_id)
            self._model = self._embedder

        # ── V1 State (unchanged) ──
        self.chunks: list[Chunk] = []
        self.chunk_ids: list[str] = []
        self.file_hashes: dict[str, str] = {}
        self.chunk_memo: dict[str, str] = {}
        self._vector_store: VectorStore | None = None
        self._bm25_index: BM25Index | None = None

        # ── V2 State ──
        self.graph: KnowledgeGraph = KnowledgeGraph()
        self._file_index: VectorStore | None = None  # file-level embeddings
        self._function_index: VectorStore | None = None  # function-level embeddings
        self._symbol_index: VectorStore | None = None  # symbol-level embeddings
        self._file_index_data: list[dict[str, Any]] = []  # metadata per file embedding
        self._function_index_data: list[dict[str, Any]] = []  # metadata per function embedding
        self._symbol_index_data: list[dict[str, Any]] = []  # metadata per symbol embedding
        self._vortex_config: VortexScoreConfig = VortexScoreConfig.default()
        self.session: SessionMemory = SessionMemory()
        self._graph_dir: Path = self.index_dir / "graph"

        self._memo_hits = 0
        self._memo_misses = 0

    def _get_dim(self) -> int:
        if self._embedder is not None:
            return self._embedder.dim
        return getattr(self._model, "dim", 256)

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = VectorStore(dim=self._get_dim())
        return self._vector_store

    @property
    def bm25_index(self) -> BM25Index:
        if self._bm25_index is None:
            self._bm25_index = BM25Index(index_dir=self.index_dir / "bm25")
        return self._bm25_index

    @property
    def file_index(self) -> VectorStore:
        if self._file_index is None:
            self._file_index = VectorStore(dim=self._get_dim())
        return self._file_index

    @property
    def function_index(self) -> VectorStore:
        if self._function_index is None:
            self._function_index = VectorStore(dim=self._get_dim())
        return self._function_index

    @property
    def symbol_index(self) -> VectorStore:
        if self._symbol_index is None:
            self._symbol_index = VectorStore(dim=self._get_dim())
        return self._symbol_index

    # ── Main index ───────────────────────────────────────────────────────

    def index(
        self,
        force: bool = False,
        include_text_files: bool = False,
        chunk_config: ChunkConfig | None = None,
    ) -> IndexStats:
        _t0 = time.perf_counter()
        config = chunk_config or self.chunk_config
        extensions = get_extensions(include_text_files)

        if not force:
            self._load_state()

        current_files: dict[str, str] = {}
        for file_path in walk_files(self.root, extensions):
            if file_path.stat().st_size > _MAX_FILE_BYTES:
                continue
            rel = str(file_path.relative_to(self.root))
            current_files[rel] = _file_hash(file_path)

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
            self._memo_hits = len(self.chunks)
            self._memo_misses = 0
            return self._compute_stats()

        logger.info(
            "Indexing: %d changed, %d removed, %d total files",
            len(changed_files), len(removed_files), len(current_files),
        )

        # Remove stale data
        stale = changed_files | removed_files
        self._remove_file_chunks(stale)
        self._remove_graph_entries(stale)

        # Chunk, parse, and embed changed files
        new_chunks: list[Chunk] = []
        all_symbols: list[tuple[str, list]] = []  # file_path -> list[SymbolInfo]
        all_imports: list[tuple[str, list]] = []  # file_path -> list[ImportInfo]

        for rel in changed_files:
            file_path = self.root / rel
            language = detect_language(file_path)
            with contextlib.suppress(OSError):
                source = file_path.read_text(encoding="utf-8", errors="replace")

                # Parse symbols and imports (V2)
                symbols, imports = parse_symbols(source, rel, language)
                all_symbols.append((rel, symbols))
                all_imports.append((rel, imports))

                # Chunk for existing index pipeline
                chunks = chunk_source(source, rel, language, config)
                for chunk in chunks:
                    if chunk.chunk_hash and not force:
                        existing_memo = self._get_memo_for_hash(chunk.chunk_hash)
                        if existing_memo:
                            self._memo_hits += 1
                            continue
                    self._memo_misses += 1
                    new_chunks.append(chunk)

        # Assign chunk IDs
        new_chunk_ids = []
        for chunk in new_chunks:
            cid = hashlib.sha256(
                f"{chunk.file_path}:{chunk.start_line}:{chunk.end_line}:{chunk.content}".encode()
            ).hexdigest()[:16]
            new_chunk_ids.append(cid)

        self.chunks.extend(new_chunks)
        self.chunk_ids.extend(new_chunk_ids)
        self.file_hashes = current_files

        for cid, chunk in zip(new_chunk_ids, new_chunks, strict=False):
            if chunk.chunk_hash:
                self.chunk_memo[cid] = chunk.chunk_hash

        # Embed chunk-level vectors (V1 pipeline)
        if new_chunks:
            logger.info("Embedding %d new chunks...", len(new_chunks))
            texts = [c.content for c in new_chunks]
            embeddings = self._encode_texts(texts)
            self.vector_store.add(embeddings, new_chunk_ids)

            # Build multi-level indexes (V2)
            self._build_file_index(changed_files)
            self._build_function_index(all_symbols)
            self._build_symbol_index(all_symbols)

        # Build knowledge graph (V2)
        if all_symbols or all_imports:
            logger.info("Building knowledge graph: %d symbols, %d import files",
                        sum(len(s) for _, s in all_symbols), len(all_imports))
            file_graph = build_graph_from_symbols(
                all_symbols, all_imports, list(current_files.keys())
            )
            # Merge into existing graph
            self._merge_graph(file_graph)

        # BM25 incremental update
        logger.info("Updating BM25 index...")
        bm25 = BM25Index(index_dir=self.index_dir / "bm25")
        bm25.build(self.chunks, self.chunk_ids, persist_dir=self.index_dir / "bm25")
        self._bm25_index = bm25

        # Load vortex weights
        weights_path = self.index_dir / "vortex_weights.json"
        loaded = load_weights(weights_path)
        if loaded:
            self._vortex_config = loaded

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
            "Index complete: %d files, %d chunks, %d graph nodes (%d memo hits) in %.1fms",
            stats.indexed_files, stats.total_chunks,
            self.graph.node_count, stats.memo_hits, stats.index_time_ms,
        )
        return stats

    # ── Search ───────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float | None = None,
        use_vortex_score: bool = False,
        hybrid: bool = False,
    ) -> list[SearchResult]:
        """Search with hybrid semantic + BM25 retrieval.

        :param use_vortex_score: If True, re-rank using the full Vortex Score.
        :param hybrid: If True, enrich each result with compact, query-aware
            structural context from the knowledge graph (one key symbol +
            one incoming + one outgoing structural edge per result file).
            Adds noise suppression via the structural-relations filter.
        """
        if not self.chunks:
            return []

        results = _search(
            query=query,
            model=self._model,
            store=self.vector_store,
            bm25_index=self.bm25_index,
            chunks=self.chunks,
            chunk_ids=self.chunk_ids,
            top_k=top_k if not use_vortex_score else top_k * 3,
            alpha=alpha,
        )

        if use_vortex_score and self.graph.node_count > 0:
            results = self._vortex_rerank(results, query, top_k)

        # Record in session
        self.session.record_query(query)
        for r in results:
            self.session.record_file_view(r.chunk.file_path)

        if hybrid and self.graph.node_count > 0:
            results = self._attach_hybrid_context(results, query)

        return results[:top_k]

    def _attach_hybrid_context(
        self,
        results: list[SearchResult],
        query: str,
    ) -> list[Any]:
        """Enrich each SearchResult with a compact GraphContext for its file.

        Picks the single most query-relevant code node in each file (via
        `score_nodes_against_query` + degree tiebreaker) and attaches:
          - key_symbol: that node's label
          - incoming: one structural edge pointing into this node (who imports it)
          - outgoing: one structural edge pointing out (a key contained method/symbol)

        Pure-prose / type-annotation edges (str/bool/Path, references) are
        filtered out — only structural relations surface, since those are
        what help the LLM understand what the code is connected to.

        Returns a list of either the original SearchResult (no enrichment
        possible) or a SearchResultWithContext wrapper. The wrapper
        delegates attribute access to the underlying result, so formatters
        can treat both uniformly.
        """
        from vortexa.core.types import GRAPH_STRUCTURAL_RELATIONS

        score_map: dict[str, float] = {
            nid: score for score, nid in self.graph.score_nodes_against_query(query)
        }

        # Group code nodes by source file for efficient lookup
        nodes_by_file: dict[str, list[str]] = defaultdict(list)
        for nid, node in self.graph._nodes.items():  # type: ignore[attr-defined]
            if node.file_path:
                nodes_by_file[node.file_path].append(nid)

        enriched: list[Any] = []
        for r in results:
            file_path = r.chunk.file_path
            candidates = nodes_by_file.get(file_path, [])
            if not candidates:
                enriched.append(r)
                continue
            # Rank: query score desc, then degree desc
            candidates.sort(
                key=lambda nid: (
                    score_map.get(nid, 0.0),
                    self.graph.degree_of(nid),
                ),
                reverse=True,
            )
            top_nid = candidates[0]
            top_node = self.graph.get_node(top_nid)

            incoming: list[str] = []
            for edge in self.graph.neighbors(top_nid, direction="in"):
                if edge.type.value in GRAPH_STRUCTURAL_RELATIONS:
                    src = self.graph.get_node(edge.source)
                    if src:
                        incoming.append(f"{src.label} ({edge.type.value})")
                    break

            outgoing: list[str] = []
            for edge in self.graph.neighbors(top_nid, direction="out"):
                if edge.type.value in GRAPH_STRUCTURAL_RELATIONS:
                    tgt = self.graph.get_node(edge.target)
                    if tgt:
                        outgoing.append(f"{tgt.label} ({edge.type.value})")
                    break

            ctx = GraphContext(
                key_symbol=top_node.label if top_node else top_nid,
                incoming=tuple(incoming),
                outgoing=tuple(outgoing),
            )
            enriched.append(SearchResultWithContext(result=r, context=ctx))
        return enriched

    # ── V2 Graph public API ──────────────────────────────────────────────

    def query_graph(
        self,
        question: str,
        mode: str = "bfs",
        depth: int = 3,
        context_filter: list[str] | None = None,
        token_budget: int = 2000,
    ) -> str:
        """Query the knowledge graph with BFS or DFS and render as text.

        See `KnowledgeGraph.query_graph` for full details. This is the
        indexer-level convenience wrapper used by the MCP `query_graph`
        tool.
        """
        try:
            traversal_mode = GraphTraversalMode(mode.lower())
        except ValueError:
            traversal_mode = GraphTraversalMode.BFS
        depth = max(1, min(int(depth), 6))
        return self.graph.query_graph(
            question=question,
            mode=traversal_mode,
            depth=depth,
            relation_filter=context_filter,
            token_budget=token_budget,
        )

    def get_god_nodes(self, top_n: int = 10) -> list[dict[str, Any]]:
        """Return the top-N most-connected real entities in the graph.

        File-level hub nodes are excluded — only meaningful abstractions
        (classes, functions, modules, interfaces) surface.
        """
        nodes = self.graph.god_nodes(top_n=top_n)
        return [
            {
                "id": n.id,
                "label": n.label,
                "kind": n.kind,
                "file_path": n.file_path,
                "degree": n.degree,
            }
            for n in nodes
        ]

    def get_graph_node(self, label: str) -> dict[str, Any] | None:
        """Return details about a single node by label or ID."""
        info = self.graph.get_node_info(label)
        if info is None:
            return None
        return {
            "id": info.id,
            "label": info.label,
            "kind": info.kind,
            "file_path": info.file_path,
            "degree": info.degree,
        }

    def get_graph_neighbors(self, label: str) -> list[dict[str, Any]]:
        """Return incoming and outgoing edges for a node."""
        edges = self.graph.get_neighbors(label)
        return [
            {
                "source": e.source,
                "target": e.target,
                "relation": e.relation,
                "direction": e.direction,
            }
            for e in edges
        ]

    def get_shortest_path(
        self, source: str, target: str, max_hops: int = 8
    ) -> GraphPath | None:
        """Find the shortest path between two nodes (matched by label)."""
        return self.graph.shortest_path_between(
            source_label=source,
            target_label=target,
            max_hops=max_hops,
        )

    def search_by_path(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Pure path-based retrieval — no embeddings, no BM25."""
        all_files = list(self.file_hashes.keys())
        scored_paths = _path_retrieve(query, all_files, top_k)

        results: list[SearchResult] = []
        for file_path, score in scored_paths:
            chunks_in_file = [
                c for c, cid in zip(self.chunks, self.chunk_ids, strict=False)
                if c.file_path == file_path
            ]
            if chunks_in_file:
                results.append(SearchResult(
                    chunk=chunks_in_file[0],
                    score=score,
                    source=cast(Any, "path"),
                ))

        return results

    def find_symbol(self, name: str, top_k: int = 10) -> list[SearchResult]:
        """Find code that defines a symbol by name. Uses symbol index + graph."""
        # Try graph first
        graph_nodes = self.graph.find_nodes_by_name(name)
        if graph_nodes:
            results: list[SearchResult] = []
            for node in graph_nodes[:top_k]:
                if node.file_path:
                    chunks_in_file = [
                        c for c in self.chunks if c.file_path == node.file_path
                    ]
                    if chunks_in_file:
                        results.append(SearchResult(
                            chunk=chunks_in_file[0],
                            score=1.0,
                            source=cast(Any, "symbol"),
                        ))
            if results:
                return results

        # Fall back to semantic search with BM25 bias
        return self.search(name, top_k=top_k, alpha=0.0)

    def resolve(self, query: str, top_k: int = 5) -> ContextPack:
        """Full context resolution: search + expand + compress.

        This is the primary V2 API — returns everything an agent needs.
        """
        primary = self.search(query, top_k=top_k, use_vortex_score=True)
        if not primary:
            return ContextPack(query=query, confidence=0.0)

        pack = expand_context(query, primary, self.graph, self.chunks, depth=1)
        compressed = _compress_pack(pack, max_tokens=8000)
        return compressed

    def explain(self, code_location: str) -> ContextPack:
        """Explain a code location: find symbol, expand context, return pack.

        :param code_location: e.g., "src/module.py:42" or "ClassName.method_name"
        """
        query = code_location

        # Try to parse as file:line
        if ":" in code_location:
            parts = code_location.rsplit(":", 1)
            if parts[0].endswith((".py", ".js", ".ts", ".rs", ".go", ".java")):
                file_path = parts[0]
                line = int(parts[1]) if parts[1].isdigit() else 0
                chunks_here = [c for c in self.chunks if c.file_path == file_path]
                if line > 0:
                    chunks_here = [c for c in chunks_here if c.start_line <= line <= c.end_line]
                if chunks_here:
                    primary = [SearchResult(chunk=chunks_here[0], score=1.0, source=cast(Any, "exact"))]
                    pack = expand_context(query, primary, self.graph, self.chunks)
                    return _compress_pack(pack)

        # Try as symbol name
        return self.resolve(code_location, top_k=3)

    def find_related(self, chunk_idx: int, top_k: int = 5) -> list[SearchResult]:
        if chunk_idx < 0 or chunk_idx >= len(self.chunks):
            return []
        chunk = self.chunks[chunk_idx]
        return self.search(chunk.content, top_k=top_k + 1)[1:top_k + 1]

    def format_context(self, pack: ContextPack) -> str:
        """Format a ContextPack as a human-readable string for agents."""
        return format_for_agent(pack)

    # ── Stats / Management ───────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        base = self._compute_stats()
        graph_stats = self.graph.stats()
        return {
            "indexed_files": base.indexed_files,
            "total_chunks": base.total_chunks,
            "languages": base.languages,
            "graph_nodes": graph_stats["nodes"],
            "graph_edges": graph_stats["edges"],
            "memo_hits": self._memo_hits,
            "memo_misses": self._memo_misses,
            "session_queries": self.session._query_count,
        }

    def clear(self) -> None:
        import shutil
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
        self.chunks.clear()
        self.chunk_ids.clear()
        self.file_hashes.clear()
        self.chunk_memo.clear()
        self._vector_store = None
        self._bm25_index = None
        self._file_index = None
        self._function_index = None
        self._symbol_index = None
        self._file_index_data.clear()
        self._function_index_data.clear()
        self._symbol_index_data.clear()
        self.graph = KnowledgeGraph()
        self.session.clear()
        self._memo_hits = 0
        self._memo_misses = 0

    # ── Internal: Multi-level indexes ────────────────────────────────────

    def _build_file_index(self, changed_files: set[str]) -> None:
        """Build file-level embeddings (one per file, from first chunk)."""
        file_id = hashlib.sha256("".join(sorted(changed_files)).encode()).hexdigest()[:8]
        file_texts: list[str] = []
        file_meta: list[dict[str, Any]] = []

        for rel in sorted(changed_files):
            chunks_in_file = [c for c in self.chunks if c.file_path == rel]
            if not chunks_in_file:
                continue
            first_chunk = chunks_in_file[0]
            last_chunk = chunks_in_file[-1]
            file_texts.append(first_chunk.content)
            file_meta.append({
                "file_path": rel,
                "start_line": first_chunk.start_line,
                "end_line": last_chunk.end_line,
                "language": first_chunk.language,
            })

        if file_texts:
            embeddings = self._encode_texts(file_texts)
            fids = [f"file:{rel}" for rel in sorted(changed_files) if any(c.file_path == rel for c in self.chunks)]
            self.file_index.add(embeddings, fids)
            self._file_index_data.extend(file_meta)

    def _build_function_index(self, all_symbols: list[tuple[str, list]]) -> None:
        """Build function-level embeddings (one per function/class signature + docstring)."""
        func_texts: list[str] = []
        func_meta: list[dict[str, Any]] = []

        for file_path, symbols in all_symbols:
            for sym in symbols:
                text = f"{sym.signature or ''}\n{sym.docstring or ''}"
                if text.strip():
                    func_texts.append(text)
                    func_meta.append({
                        "name": sym.name,
                        "kind": sym.kind.value,
                        "file_path": sym.file_path,
                        "start_line": sym.start_line,
                        "end_line": sym.end_line,
                    })

        if func_texts:
            embeddings = self._encode_texts(func_texts)
            func_ids = [f"func:{m['name']}@{m['file_path']}:{m['start_line']}" for m in func_meta]
            self.function_index.add(embeddings, func_ids)
            self._function_index_data.extend(func_meta)

    def _build_symbol_index(self, all_symbols: list[tuple[str, list]]) -> None:
        """Build symbol-level embeddings (one per symbol name + kind)."""
        sym_texts: list[str] = []
        sym_meta: list[dict[str, Any]] = []

        for file_path, symbols in all_symbols:
            for sym in symbols:
                text = f"{sym.name} ({sym.kind.value}): {sym.signature or sym.docstring or ''}"
                if sym.name.strip():
                    sym_texts.append(text)
                    sym_meta.append({
                        "name": sym.name,
                        "kind": sym.kind.value,
                        "file_path": sym.file_path,
                        "start_line": sym.start_line,
                    })

        if sym_texts:
            embeddings = self._encode_texts(sym_texts)
            sym_ids = [f"sym:{m['name']}" for m in sym_meta]
            self.symbol_index.add(embeddings, sym_ids)
            self._symbol_index_data.extend(sym_meta)

    def _merge_graph(self, other: KnowledgeGraph) -> None:
        """Merge another KnowledgeGraph into the current one."""
        for nid, node in other._nodes.items():
            if not self.graph.has_node(nid):
                self.graph.add_node(
                    node_id=nid,
                    kind=node.kind,
                    label=node.label,
                    file_path=node.file_path,
                    metadata=node.metadata,
                )
        for source_id, edges in other._out_edges.items():
            for edge in edges:
                if self.graph.has_node(edge.source) and self.graph.has_node(edge.target):
                    self.graph.add_edge(
                        source=edge.source,
                        target=edge.target,
                        edge_type=edge.type,
                        weight=edge.weight,
                        metadata=edge.metadata,
                    )

    def _remove_graph_entries(self, file_rels: set[str]) -> None:
        """Remove graph nodes and edges for removed files."""
        for rel in file_rels:
            file_node = self.graph.find_file_node(rel)
            if file_node:
                self.graph.remove_node(file_node.id)
            # Remove symbol nodes in this file
            for node in self.graph.find_nodes_in_file(rel):
                self.graph.remove_node(node.id)

    # ── Internal: Vortex Score reranking ─────────────────────────────────

    def _vortex_rerank(
        self,
        results: list[SearchResult],
        query: str,
        top_k: int,
    ) -> list[SearchResult]:
        """Re-rank results using the full Vortex Score."""
        if not results:
            return results

        # Collect BM25 scores for same chunks
        bm25_results = self._search_bm25_only(query, top_k=top_k * 5)
        bm25_map: dict[str, float] = {}
        for r in bm25_results:
            bm25_map[r.chunk.location] = r.score

        max_emb_score = max(r.score for r in results) if results else 1.0

        vortex_scored: list[tuple[float, SearchResult]] = []
        for r in results:
            bm25_score = bm25_map.get(r.chunk.location, 0.0)
            emb_score = r.score / max_emb_score if max_emb_score > 0 else 0.0

            vs = compute_vortex_score(
                chunk=r.chunk,
                query=query,
                embedding_score=emb_score,
                bm25_score=bm25_score,
                config=self._vortex_config,
                graph=self.graph if self.graph.node_count > 0 else None,
            )

            vortex_scored.append((vs, SearchResult(
                chunk=r.chunk,
                score=vs,
                source=r.source,
            )))

        vortex_scored.sort(key=lambda x: -x[0])
        return [r for _, r in vortex_scored[:top_k]]

    def _search_bm25_only(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Search using BM25 only."""
        if not self.chunks or self.bm25_index._index is None:
            return []
        from vortexa.search.search import search_bm25
        return search_bm25(query, self.bm25_index, self.chunks, self.chunk_ids, top_k)

    # ── Internal: Encoding ───────────────────────────────────────────────

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        if self._embedder is not None:
            result = self._embedder.embed_batch(texts)
            return np.array(result, dtype=np.float32)
        else:
            result = cast(Encoder, self._model).encode(texts)
            return np.array(result, dtype=np.float32)

    def _get_memo_for_hash(self, chunk_hash: str) -> str | None:
        for cid, memo_hash in self.chunk_memo.items():
            if memo_hash == chunk_hash:
                return cid
        return None

    def _remove_file_chunks(self, file_rels: set[str]) -> None:
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

    # ── Persistence ──────────────────────────────────────────────────────

    def _save_state(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Save V1 state
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

        # Save V2 state: graph
        self.graph.save(self._graph_dir)

        # Save multi-level indexes
        self._save_multi_index("file", self.file_index, self._file_index_data)
        self._save_multi_index("function", self.function_index, self._function_index_data)
        self._save_multi_index("symbol", self.symbol_index, self._symbol_index_data)

        # Save session
        self.session.save(self.index_dir)

        # Save vortex weights
        from vortexa.search.vortex_score import save_weights
        save_weights(self._vortex_config, self.index_dir / "vortex_weights.json")

    def _save_multi_index(self, name: str, store: VectorStore, meta: list[dict[str, Any]]) -> None:
        """Save a multi-level index and its metadata."""
        idx_dir = self.index_dir / name
        idx_dir.mkdir(parents=True, exist_ok=True)
        store.save(idx_dir)
        if meta:
            (idx_dir / "meta.json").write_text(json.dumps(meta))

    def _load_state(self) -> bool:
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
                        self.chunks.append(Chunk(
                            file_path=data[0],
                            content=data[1],
                            start_line=data[2],
                            end_line=data[3],
                            language=data[4],
                            chunk_hash=data[5],
                        ))

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

        # Load V2 state: graph
        loaded_graph = KnowledgeGraph.load(self._graph_dir)
        if loaded_graph:
            self.graph = loaded_graph
            logger.info("Loaded graph: %d nodes, %d edges", self.graph.node_count, self.graph.edge_count)

        # Load multi-level indexes
        self._load_multi_index("file")
        self._load_multi_index("function")
        self._load_multi_index("symbol")

        # Load session
        self.session = SessionMemory.load(self.index_dir)

        # Load vortex weights
        weights_path = self.index_dir / "vortex_weights.json"
        loaded = load_weights(weights_path)
        if loaded:
            self._vortex_config = loaded

        return bool(self.chunks)

    def _load_multi_index(self, name: str) -> None:
        """Load a multi-level index and its metadata."""
        idx_dir = self.index_dir / name
        if not idx_dir.exists():
            return
        store = VectorStore.load(idx_dir)
        if store is not None:
            if name == "file":
                self._file_index = store
            elif name == "function":
                self._function_index = store
            elif name == "symbol":
                self._symbol_index = store

        meta_path = idx_dir / "meta.json"
        if meta_path.exists():
            try:
                data = json.loads(meta_path.read_text())
                if name == "file":
                    self._file_index_data = data
                elif name == "function":
                    self._function_index_data = data
                elif name == "symbol":
                    self._symbol_index_data = data
            except Exception:
                pass
