"""VortexA v2 — integrated context engine.

Combines:
  - VortexEmbedderV4 (4-bit dense retrieval, SIF+PC, on-the-fly LF4 dequant)
  - RepoGraph (knowledge graph)
  - VortexScore (multi-signal ranking)
  - Context expansion (related files/tests)

The main entry point is VortexContextEngine, which takes a query and
returns a "context pack": the top files plus related tests, imports,
callers, and callees.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from vortexa.core.chunking import chunk_source
from vortexa.core.embedding import Embedder
from vortexa.core.graph import RepoGraph, RepoGraphBuilder
from vortexa.core.language import detect_language, get_extensions
from vortexa.core.v4_embedder import VortexEmbedderV4
from vortexa.core.vortex_score import VortexScoreWeights, vortex_score, tokenize

logger = logging.getLogger(__name__)


@dataclass
class ContextFile:
    """A file in the context pack with its score and reason."""
    path: str
    score: float
    components: Dict[str, float] = field(default_factory=dict)
    snippet: str = ""
    chunk_ids: List[str] = field(default_factory=list)
    relations: Dict[str, List[str]] = field(default_factory=dict)  # {relation: [related_paths]}


@dataclass
class ContextPack:
    """A full context pack returned by VortexA v2."""
    query: str
    primary: List[ContextFile]  # top files
    related: List[ContextFile]  # tests, imports, callers, callees
    graph_hits: List[str] = field(default_factory=list)  # files hit via graph
    elapsed_ms: float = 0.0
    n_chunks_scanned: int = 0


class VortexContextEngine:
    """The V2 context engine.

    Workflow:
        1. Encode query with V3 embedder
        2. Dense search against chunk index
        3. Re-rank with Vortex Score (multi-signal)
        4. Expand to related files via graph
        5. Return ContextPack

    Args:
        embedder: VortexEmbedderV4 (or compatible Embedder)
        chunks: List[dict] with 'path', 'content', 'start', 'end', 'chunk_id'
        chunk_embeddings: np.ndarray (N, dim)
        graph: Optional RepoGraph
        file_imports: Optional dict {path: set of imported paths}
        file_adjacency: Optional dict {path: set of directly related paths}
        weights: VortexScoreWeights
    """

    def __init__(
        self,
        embedder: Embedder,
        chunks: List[dict],
        chunk_embeddings: np.ndarray,
        *,
        graph: Optional[RepoGraph] = None,
        file_imports: Optional[Dict[str, Set[str]]] = None,
        file_adjacency: Optional[Dict[str, Set[str]]] = None,
        file_idf: Optional[Dict[str, Dict[str, float]]] = None,
        weights: Optional[VortexScoreWeights] = None,
    ):
        self.embedder = embedder
        self.chunks = chunks
        self.chunk_embeddings = chunk_embeddings
        self.graph = graph
        self.file_imports = file_imports or {}
        self.file_adjacency = file_adjacency or {}
        # file_idf: {file_path: {term: idf_weight}}
        self.file_idf = file_idf or {}
        self.weights = weights or VortexScoreWeights()
        # Build path -> chunks index
        self._path_to_chunks: Dict[str, List[int]] = {}
        for i, c in enumerate(chunks):
            self._path_to_chunks.setdefault(c["path"], []).append(i)
        # Build path -> symbols index
        self._path_to_symbols: Dict[str, Set[str]] = {}
        if self.graph is not None:
            for nid, node in self.graph.nodes.items():
                if node.kind != "file":
                    self._path_to_symbols.setdefault(node.path, set()).add(nid)

    def search(
        self,
        query: str,
        top_k_dense: int = 50,
        top_k_final: int = 10,
        expand_related: bool = True,
        max_related: int = 8,
    ) -> ContextPack:
        """Search and return a ContextPack."""
        t0 = time.perf_counter()
        # 1. Dense retrieval
        q_emb = self.embedder.embed(query)
        scores = self.chunk_embeddings @ q_emb
        top_idx = np.argsort(scores)[::-1][:top_k_dense]

        # 2. Aggregate to file-level (best chunk per file)
        file_best: Dict[str, Tuple[float, int]] = {}
        for idx in top_idx:
            path = self.chunks[idx]["path"]
            s = float(scores[idx])
            if path not in file_best or s > file_best[path][0]:
                file_best[path] = (s, idx)

        # 3. Build per-file file_idf (term frequency in this file)
        # 4. Get seed files for graph expansion
        seed_files = set(file_best.keys())

        # 5. Re-rank with Vortex Score
        scored: List[Tuple[str, float, dict, int]] = []  # (path, score, components, chunk_idx)
        for path, (emb_score, chunk_idx) in file_best.items():
            file_syms = self._path_to_symbols.get(path, set())
            chunk_text = self.chunks[chunk_idx].get("content", "")
            file_idf = self.file_idf.get(path, {})
            v_score, comps = vortex_score(
                emb_score, query, path,
                file_symbols=file_syms, chunk_text=chunk_text,
                file_idf=file_idf, seed_files=seed_files,
                adjacency=self.file_adjacency, imports=self.file_imports,
                weights=self.weights,
            )
            scored.append((path, v_score, comps, chunk_idx))
        # Sort by Vortex score
        scored.sort(key=lambda x: -x[1])

        # 6. Build top-K primary files
        primary = []
        for path, vscore, comps, chunk_idx in scored[:top_k_final]:
            cf = ContextFile(
                path=path, score=vscore, components=comps,
                snippet=self.chunks[chunk_idx].get("content", "")[:500],
                chunk_ids=[self.chunks[chunk_idx].get("chunk_id", "")],
            )
            primary.append(cf)

        # 7. Expand to related files
        related = []
        if expand_related:
            related_paths: Set[str] = set()
            # Related via graph (1-hop neighbors)
            if self.graph is not None:
                for cf in primary:
                    nid = f"file:{cf.path}"
                    if nid in self.graph.nodes:
                        for nb in self.graph.neighbors(nid)[:5]:
                            nb_path = nb.split("file:")[-1] if nb.startswith("file:") else None
                            if nb_path and nb_path not in seed_files and nb_path in self._path_to_chunks:
                                related_paths.add(nb_path)
            # Related via adjacency (imports)
            for cf in primary:
                for nb in list(self.file_adjacency.get(cf.path, set()))[:3]:
                    if nb not in seed_files and nb in self._path_to_chunks:
                        related_paths.add(nb)
            # Related via test files (heuristic: file path contains "test" and shares parent dir)
            for cf in primary:
                base = cf.path.replace(".py", "")
                test_candidates = [
                    f"tests/test_{base.split('/')[-1]}.py",
                    f"test_{base.split('/')[-1]}.py",
                    f"{base}_test.py",
                ]
                for tc in test_candidates:
                    if tc in self._path_to_chunks and tc not in seed_files:
                        related_paths.add(tc)
            # Score the related files
            related_scored = []
            for path in related_paths:
                # Use a lower-priority scoring (no emb contribution from seed)
                # Find best chunk for this file
                chunk_indices = self._path_to_chunks.get(path, [])
                if not chunk_indices:
                    continue
                # Use first chunk for scoring context
                chunk_idx = chunk_indices[0]
                v_score, comps = vortex_score(
                    0.0, query, path,
                    file_symbols=self._path_to_symbols.get(path, set()),
                    chunk_text=self.chunks[chunk_idx].get("content", ""),
                    file_idf=self.file_idf.get(path, {}),
                    seed_files=set(cf.path for cf in primary),
                    adjacency=self.file_adjacency, imports=self.file_imports,
                    weights=self.weights,
                )
                related_scored.append((path, v_score, comps, chunk_idx))
            related_scored.sort(key=lambda x: -x[1])
            for path, vscore, comps, chunk_idx in related_scored[:max_related]:
                related.append(ContextFile(
                    path=path, score=vscore, components=comps,
                    snippet=self.chunks[chunk_idx].get("content", "")[:500],
                    chunk_ids=[self.chunks[chunk_idx].get("chunk_id", "")],
                ))

        elapsed = (time.perf_counter() - t0) * 1000
        return ContextPack(
            query=query,
            primary=primary,
            related=related,
            graph_hits=[cf.path for cf in related if cf.path not in seed_files],
            elapsed_ms=elapsed,
            n_chunks_scanned=len(top_idx),
        )

    def to_text(self, pack: ContextPack) -> str:
        """Format a context pack as text for an LLM."""
        lines = [f"# Context for: {pack.query}", ""]
        lines.append("## Primary files")
        for cf in pack.primary:
            lines.append(f"\n### {cf.path} (score: {cf.score:.3f})")
            if cf.components:
                comp_str = ", ".join(f"{k}={v:.2f}" for k, v in cf.components.items() if v > 0)
                if comp_str:
                    lines.append(f"  Signals: {comp_str}")
            lines.append("```")
            lines.append(cf.snippet)
            lines.append("```")
        if pack.related:
            lines.append("\n## Related files (context expansion)")
            for cf in pack.related:
                lines.append(f"\n### {cf.path} (score: {cf.score:.3f})")
                lines.append("```")
                lines.append(cf.snippet[:300])
                lines.append("```")
        return "\n".join(lines)
