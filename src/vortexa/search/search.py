"""Hybrid search engine for the codebase indexer.

Combines semantic (vector) search with BM25 lexical search using
Reciprocal Rank Fusion (RRF) scoring with configurable alpha weighting.

Includes lineage-aware search and find_related (cocoindex-style).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from vortexa.core.embedding import Embedder
from vortexa.core.types import Chunk, Encoder, SearchMode, SearchResult
from vortexa.search.ranking import (
    apply_query_boost,
    boost_multi_chunk_files,
    rerank_topk,
    resolve_alpha,
)
from vortexa.storage.bm25 import BM25Index
from vortexa.storage.vector_store import VectorStore

_RRF_K = 60


def _rrf_scores(scores: dict[Chunk, float]) -> dict[Chunk, float]:
    """Convert raw scores to RRF scores 1/(k + rank); higher raw score -> rank 1."""
    if not scores:
        return scores
    ranked = sorted(scores, key=lambda c: -scores[c])
    return {chunk: 1.0 / (_RRF_K + rank) for rank, chunk in enumerate(ranked, 1)}


def _encode_query(model, query: str) -> np.ndarray:
    """Encode a query using either Encoder or Embedder."""
    from vortexa.core.embedding import Embedder
    if isinstance(model, Embedder):
        return np.array([model.embed(query)], dtype=np.float32)
    return model.encode([query])


def search_semantic(
    query: str,
    model: Encoder | Embedder,
    store: VectorStore,
    chunks: list[Chunk],
    chunk_ids: list[str],
    top_k: int,
    selector: npt.NDArray[np.int_] | None = None,
) -> list[SearchResult]:
    """Run semantic search for a query."""
    query_embedding = _encode_query(model, query)
    if len(query_embedding) == 0:
        return []

    query_vec = query_embedding[0]
    results = store.query(query_vec, k=top_k, selector=selector)

    # Map store vector indices → chunk IDs → chunks (positions may diverge)
    id_to_chunk = dict(zip(chunk_ids, chunks, strict=False))
    out: list[SearchResult] = []
    for idx, dist in results:
        cid = store.get_id(idx)
        if cid is None or cid not in id_to_chunk:
            continue
        out.append(SearchResult(chunk=id_to_chunk[cid], score=1.0 - dist, source=SearchMode.SEMANTIC))
    return out


def search_bm25(
    query: str,
    bm25_index: BM25Index,
    chunks: list[Chunk],
    chunk_ids: list[str],
    top_k: int,
    selector_set: set[int] | None = None,
) -> list[SearchResult]:
    """Return chunks ranked by BM25 score, excluding zero-score results."""
    results = bm25_index.search(query, chunks, chunk_ids, top_k, selector=selector_set)

    return [
        SearchResult(chunk=chunks[idx], score=score, source=SearchMode.BM25)
        for idx, score in results
        if score > 0
    ]


def search_hybrid(
    query: str,
    model: Encoder | Embedder,
    store: VectorStore,
    bm25_index: BM25Index,
    chunks: list[Chunk],
    chunk_ids: list[str],
    top_k: int,
    alpha: float | None = None,
    selector: npt.NDArray[np.int_] | None = None,
) -> list[SearchResult]:
    """Hybrid search: alpha-weighted combination of semantic and BM25 scores.

    Both score sets are converted to RRF scores before combining, so alpha has
    a consistent meaning regardless of raw score magnitude.
    """
    alpha_weight = resolve_alpha(query, alpha)

    # Over-fetch candidates so the merged pool is large enough
    candidate_count = top_k * 5

    semantic = search_semantic(query, model, store, chunks, chunk_ids, candidate_count, selector)
    semantic_scores: dict[Chunk, float] = {result.chunk: result.score for result in semantic}

    # Convert selector to set for BM25
    selector_set = set(selector.tolist()) if selector is not None else None
    bm25_results = search_bm25(query, bm25_index, chunks, chunk_ids, candidate_count, selector_set)
    bm25_scores = {result.chunk: result.score for result in bm25_results}

    normalized_semantic = _rrf_scores(semantic_scores)
    normalized_bm25 = _rrf_scores(bm25_scores)

    # Sort by file path and start line to counteract randomness from hashing
    all_candidates = sorted(
        {*normalized_semantic, *normalized_bm25},
        key=lambda c: c.start_line,
    )
    combined_scores: dict[Chunk, float] = {
        chunk: alpha_weight * normalized_semantic.get(chunk, 0.0)
        + (1.0 - alpha_weight) * normalized_bm25.get(chunk, 0.0)
        for chunk in all_candidates
    }

    # Boost files with multiple relevant chunks
    boost_multi_chunk_files(combined_scores)
    # Boost queries with specific identifiers in them
    combined_scores = apply_query_boost(combined_scores, query, chunks)
    # Rerank the top-k results by applying path-based penalties
    ranked = rerank_topk(combined_scores, top_k, penalise_paths=alpha_weight < 1.0)
    return [SearchResult(chunk=chunk, score=score, source=SearchMode.HYBRID) for chunk, score in ranked]


def search(
    query: str,
    model: Encoder | Embedder,
    store: VectorStore,
    bm25_index: BM25Index,
    chunks: list[Chunk],
    chunk_ids: list[str],
    top_k: int,
    alpha: float | None = None,
    selector: npt.NDArray[np.int_] | None = None,
) -> list[SearchResult]:
    """Search with hybrid semantic + BM25 retrieval and full lineage."""
    results = search_hybrid(
        query=query,
        model=model,
        store=store,
        bm25_index=bm25_index,
        chunks=chunks,
        chunk_ids=chunk_ids,
        top_k=top_k,
        alpha=alpha,
        selector=selector,
    )
    # Lineage is already attached to chunks via chunk.lineage
    return results
