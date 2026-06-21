"""Vortex Score — weighted fusion of all retrieval signals.

Combines embedding, filename, path, symbol, graph, import, BM25/IDF,
and structural scores into a single learned score.

Weights are stored persistently and can be tuned via benchmark feedback.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vortexa.core.graph import KnowledgeGraph
    from vortexa.core.types import Chunk

from vortexa.core.types import VortexScoreConfig

logger = logging.getLogger(__name__)


def compute_vortex_score(
    chunk: Chunk,
    query: str,
    embedding_score: float,
    bm25_score: float,
    config: VortexScoreConfig,
    graph: KnowledgeGraph | None = None,
) -> float:
    """Compute the composite Vortex Score for a single chunk.

    Signals (each normalized [0, 1]):
    - embedding_score: semantic similarity
    - filename_score: filename match
    - path_score: directory path match
    - symbol_score: symbol definition presence
    - graph_score: graph proximity to other results
    - import_score: import relationship
    - bm25_idf_score: BM25 lexical score (normalized)
    - structural_score: structural centrality

    :param chunk: The chunk to score.
    :param query: Original query.
    :param embedding_score: Semantic similarity score.
    :param bm25_score: BM25 score.
    :param config: Weight configuration.
    :param graph: Optional knowledge graph for graph/structural signals.
    :return: Composite score.
    """
    # Normalize BM25 score to [0, 1] range
    bm25_norm = min(bm25_score / 10.0, 1.0) if bm25_score > 0 else 0.0

    # Compute signal scores
    filename_sig = _filename_signal(query, chunk.file_path)
    path_sig = _path_signal(query, chunk.file_path)
    symbol_sig = _symbol_signal(query, chunk)
    graph_sig = _graph_signal(query, chunk, graph) if graph else 0.0
    import_sig = _import_signal(query, chunk, graph) if graph else 0.0
    structural_sig = _structural_signal(chunk, graph) if graph else 0.0

    # Weighted sum
    score = (
        config.embedding_weight * embedding_score
        + config.filename_weight * filename_sig
        + config.path_weight * path_sig
        + config.symbol_weight * symbol_sig
        + config.graph_weight * graph_sig
        + config.import_weight * import_sig
        + config.bm25_idf_weight * bm25_norm
        + config.structural_weight * structural_sig
    )

    # Normalize by sum of weights
    total_weight = sum(config.as_dict().values())
    return score / total_weight if total_weight > 0 else 0.0


# ── Signal functions ─────────────────────────────────────────────────────


def _filename_signal(query: str, file_path: str) -> float:
    """Score filename match."""
    from vortexa.search.path_scorer import path_score
    return path_score(query, file_path)


def _path_signal(query: str, file_path: str) -> float:
    """Score path-level match (module structure)."""
    query_lower = query.lower()
    module_path = file_path.replace("/", ".").replace("\\", ".").lower()
    segments = module_path.split(".")

    # Check if query terms appear in path segments
    query_terms = set(q for q in query_lower.split() if len(q) > 2)
    if not query_terms:
        return 0.0

    matches = sum(1 for seg in segments if _match_any(seg, query_terms))
    return min(matches / max(len(query_terms), 1), 1.0) * 0.5


def _match_any(segment: str, terms: set[str]) -> bool:
    """Check if any term matches a path segment."""
    for term in terms:
        if term == segment or segment.startswith(term) or term.startswith(segment):
            return True
    return False


def _symbol_signal(query: str, chunk: Chunk) -> float:
    """Score symbol definition presence in chunk."""
    import re
    from vortexa.search.ranking import is_symbol_query, _extract_symbol_name

    if not is_symbol_query(query):
        return 0.0

    symbol = _extract_symbol_name(query)
    if not symbol:
        return 0.0

    # Check if chunk defines the symbol
    # Match patterns: "class SymbolName", "def symbol_name", etc.
    pattern = re.compile(
        r"(?:class|def|fn|fun|func|function|struct|enum|trait|interface|type)\s+"
        + re.escape(symbol)
        + r"\b"
    )
    return 1.0 if pattern.search(chunk.content) else 0.0


def _graph_signal(query: str, chunk: Chunk, graph: KnowledgeGraph) -> float:
    """Score graph proximity to relevant nodes."""
    file_node = graph.find_file_node(chunk.file_path)
    if file_node is None:
        return 0.0

    # Count connections (more connections = more central = slightly higher score)
    out_count = len(graph.neighbors(file_node.id, direction="out"))
    in_count = len(graph.neighbors(file_node.id, direction="in"))

    # Normalize: assume 50+ connections is "very connected"
    total = min(out_count + in_count, 50) / 50.0
    return total * 0.3


def _import_signal(query: str, chunk: Chunk, graph: KnowledgeGraph) -> float:
    """Score import chain relevance."""
    file_node = graph.find_file_node(chunk.file_path)
    if file_node is None:
        return 0.0

    query_lower = query.lower()
    # Check if imported modules match query
    for edge in graph.neighbors(file_node.id, direction="out"):
        target = graph.get_node(edge.target)
        if target and target.label and query_lower in target.label.lower():
            return 0.8
        if target and target.file_path and query_lower in target.file_path.lower():
            return 0.6

    return 0.0


def _structural_signal(chunk: Chunk, graph: KnowledgeGraph) -> float:
    """Score structural importance (reference count)."""
    file_node = graph.find_file_node(chunk.file_path)
    if file_node is None:
        return 0.0

    # Number of incoming edges = reference count
    in_edges = graph.neighbors(file_node.id, direction="in")
    count = len(in_edges)
    if count == 0:
        return 0.0
    # Normalize: 20+ references = full score
    return min(count / 20.0, 1.0) * 0.2


# ── Weight persistence ────────────────────────────────────────────────────


def save_weights(config: VortexScoreConfig, path: Path) -> None:
    """Persist Vortex Score weights to disk."""
    path.write_text(json.dumps(config.as_dict(), indent=2))


def load_weights(path: Path) -> VortexScoreConfig | None:
    """Load Vortex Score weights from disk."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return VortexScoreConfig(
            embedding_weight=data.get("embedding", 1.0),
            filename_weight=data.get("filename", 0.8),
            path_weight=data.get("path", 0.6),
            symbol_weight=data.get("symbol", 1.2),
            graph_weight=data.get("graph", 0.5),
            import_weight=data.get("import", 0.4),
            bm25_idf_weight=data.get("bm25_idf", 0.7),
            structural_weight=data.get("structural", 0.3),
        )
    except Exception:
        logger.exception("Failed to load Vortex Score weights")
        return None
