"""Structural retrieval — import graph, call graph, and dependency traversal.

Boosts retrieval results based on structural relationships in the
repository knowledge graph: import proximity, test relationships,
reference density, and dependency chains.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vortexa.core.graph import KnowledgeGraph
    from vortexa.core.types import Chunk, EdgeType

logger = logging.getLogger(__name__)


def boost_import_proximity(
    scores: dict[Chunk, float],
    query_chunks: list[Chunk],
    graph: KnowledgeGraph,
    max_score: float,
) -> None:
    """Boost chunks whose files are import-neighbors of top-scoring files."""
    if not query_chunks or not max_score:
        return

    # Find the top files from the initial results
    top_files: set[str] = set()
    for chunk in query_chunks:
        top_files.add(chunk.file_path)

    # For each top file, find what it imports and what imports it
    related_files: dict[str, float] = {}
    for file_path in top_files:
        file_node = graph.find_file_node(file_path)
        if file_node is None:
            continue

        # Outgoing imports (this file imports X)
        for edge in graph.neighbors(file_node.id, direction="out"):
            target = graph.get_node(edge.target)
            if target and target.file_path and target.file_path != file_path:
                related_files[target.file_path] = max(
                    related_files.get(target.file_path, 0.0),
                    edge.weight * 0.3,
                )

        # Incoming imports (X imports this file)
        for edge in graph.neighbors(file_node.id, direction="in"):
            source = graph.get_node(edge.source)
            if source and source.file_path and source.file_path != file_path:
                related_files[source.file_path] = max(
                    related_files.get(source.file_path, 0.0),
                    edge.weight * 0.2,
                )

    # Apply boost to chunks in related files
    boost_unit = max_score * 0.15
    for chunk in list(scores):
        if chunk.file_path in related_files:
            scores[chunk] += boost_unit * related_files[chunk.file_path]


def boost_test_relationships(
    scores: dict[Chunk, float],
    graph: KnowledgeGraph,
    max_score: float,
) -> None:
    """Boost test files when source files are retrieved and vice versa."""
    if not max_score:
        return

    # Identify test vs source files in the results
    test_files: set[str] = set()
    source_files: set[str] = set()
    for chunk in scores:
        if _is_test_file(chunk.file_path):
            test_files.add(chunk.file_path)
        else:
            source_files.add(chunk.file_path)

    # Boost source files that have corresponding test files in results
    boost_unit = max_score * 0.2
    for source_file in source_files:
        expected_test = _get_expected_test_path(source_file)
        if expected_test in test_files:
            # Boost source and test
            for chunk in list(scores):
                if chunk.file_path == source_file:
                    scores[chunk] += boost_unit
                elif chunk.file_path == expected_test:
                    scores[chunk] += boost_unit


def boost_reference_density(
    scores: dict[Chunk, float],
    graph: KnowledgeGraph,
    max_score: float,
) -> None:
    """Boost chunks in files that are heavily referenced by other files."""
    if not max_score:
        return

    # Count incoming references per file
    ref_counts: Counter[str] = Counter()

    # More practical: count edges targeting each file node
    for node_id in list(graph._nodes.keys()):  # type: ignore[attr-defined]
        if node_id.startswith("file:"):
            in_count = len(graph.neighbors(node_id, direction="in"))
            ref_counts[node_id] = in_count

    if not ref_counts:
        return

    max_refs = max(ref_counts.values())
    if max_refs == 0:
        return

    boost_unit = max_score * 0.1
    for chunk in list(scores):
        file_node = graph.find_file_node(chunk.file_path)
        if file_node and file_node.id in ref_counts:
            density = ref_counts[file_node.id] / max_refs
            if density > 0.2:  # Only boost files with above-average references
                scores[chunk] += boost_unit * density


def compute_structural_boost(
    scores: dict[Chunk, float],
    top_chunks: list[Chunk],
    graph: KnowledgeGraph,
    max_score: float,
) -> None:
    """Apply all structural boost signals to scores (in-place)."""
    if graph.node_count == 0:
        return

    boost_import_proximity(scores, top_chunks, graph, max_score)
    boost_test_relationships(scores, graph, max_score)
    boost_reference_density(scores, graph, max_score)


# ── Helpers ───────────────────────────────────────────────────────────────


def _is_test_file(file_path: str) -> bool:
    """Check if a file path looks like a test file."""
    path = Path(file_path)
    stem = path.stem.lower()
    return (
        stem.startswith("test_")
        or stem.endswith("_test")
        or stem.endswith("_spec")
        or stem.endswith("tests")
    )


def _get_expected_test_path(source_file: str) -> str:
    """Get the expected test file path for a source file."""
    path = Path(source_file)
    stem = path.stem
    parent = path.parent

    # Common conventions: test_<name>.py, <name>_test.py
    test_path1 = parent / f"test_{stem}{path.suffix}"
    test_path2 = parent / f"{stem}_test{path.suffix}"

    # Also check in tests/ directory
    test_path3 = parent / "tests" / f"test_{stem}{path.suffix}"

    # Return the one most likely to exist (we just return the convention)
    return str(test_path1)
