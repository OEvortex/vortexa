"""Context expansion — automatic discovery of related code.

Given a set of retrieval results, automatically expands to include:
- Test files
- Imported modules
- Importers
- Callers and callees
- Sibling modules
- Related symbols

Returns a structured ContextPack ready for compression.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vortexa.core.graph import KnowledgeGraph
    from vortexa.core.types import Chunk, GraphNode, SearchResult, SymbolInfo

from vortexa.core.types import ContextPack

logger = logging.getLogger(__name__)


def expand_context(
    query: str,
    primary_results: list[SearchResult],
    graph: KnowledgeGraph,
    all_chunks: list[Chunk],
    depth: int = 1,
) -> ContextPack:
    """Expand from primary search results to a full ContextPack.

    :param query: Original search query.
    :param primary_results: Top-k search results from hybrid search.
    :param graph: Repository knowledge graph.
    :param all_chunks: All indexed chunks (for sibling extraction).
    :param depth: Graph expansion depth.
    :return: ContextPack with expanded context.
    """
    if not primary_results:
        return ContextPack(query=query, primary_chunks=primary_results)

    # Collect primary file paths
    primary_files: set[str] = set()
    for r in primary_results:
        primary_files.add(r.chunk.file_path)

    # Compute all fields
    test_files = _find_test_files(primary_files)
    imports_list, imported_by_list = _find_imports_importers(primary_files, graph)
    symbols = _find_symbols(primary_files, graph)
    callers, callees = _find_callers_callees(primary_files, graph)
    sibling_chunks = _find_siblings(primary_results, all_chunks)
    dependency_chain = _find_dependency_chain(primary_files, graph, depth)

    scores = [r.score for r in primary_results if r.score > 0]
    confidence = sum(scores) / len(scores) if scores else 0.0

    related_files = list(
        primary_files | set(imports_list) | set(imported_by_list) | set(test_files)
    )

    # Build trace with computed values
    trace = _build_trace(query, primary_files, test_files, imports_list,
                         imported_by_list, symbols, dependency_chain, confidence)

    pack = ContextPack(
        query=query,
        primary_chunks=primary_results,
        related_files=related_files,
        test_files=test_files,
        imports=imports_list,
        imported_by=imported_by_list,
        symbols=symbols,
        callers=callers,
        callees=callees,
        sibling_chunks=sibling_chunks,
        dependency_chain=dependency_chain,
        confidence=confidence,
        reasoning_trace=trace,
    )

    return pack


# ── Internal expansion strategies ─────────────────────────────────────────


def _find_test_files(primary_files: set[str]) -> list[str]:
    """Discover test files corresponding to primary files."""
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
                test_files.append(candidate_str)

    return test_files


def _find_imports_importers(
    primary_files: set[str],
    graph: KnowledgeGraph,
) -> tuple[list[str], list[str]]:
    """Discover imports and importers via knowledge graph."""
    imports: list[str] = []
    imported_by: list[str] = []

    for file_path in primary_files:
        file_node = graph.find_file_node(file_path)
        if file_node is None:
            continue

        # What this file imports
        for edge in graph.neighbors(file_node.id, direction="out"):
            target = graph.get_node(edge.target)
            if target and target.file_path and target.file_path not in primary_files:
                if target.file_path not in imports:
                    imports.append(target.file_path)

        # What imports this file
        for edge in graph.neighbors(file_node.id, direction="in"):
            source = graph.get_node(edge.source)
            if source and source.file_path and source.file_path not in primary_files:
                if source.file_path not in imported_by:
                    imported_by.append(source.file_path)

    return imports, imported_by


def _find_symbols(
    primary_files: set[str],
    graph: KnowledgeGraph,
) -> list[SymbolInfo]:
    """Find symbols defined in primary files. Skips file-level nodes."""
    symbols: list[SymbolInfo] = []
    seen: set[str] = set()
    skip_kinds = {"file"}

    for file_path in primary_files:
        for node in graph.find_nodes_in_file(file_path):
            if node.id not in seen and node.kind.value not in skip_kinds:
                seen.add(node.id)
                try:
                    from vortexa.core.types import SymbolInfo as SI, SymbolKind
                    symbol = SI(
                        name=node.label,
                        kind=SymbolKind(node.kind.value) if isinstance(node.kind.value, str) else node.kind,
                        file_path=node.file_path or "",
                        start_line=node.metadata.get("start_line", 0),
                        end_line=node.metadata.get("end_line", 0),
                        docstring=node.metadata.get("docstring"),
                        signature=node.metadata.get("signature"),
                    )
                    symbols.append(symbol)
                except Exception:
                    pass

    return symbols


def _find_callers_callees(
    primary_files: set[str],
    graph: KnowledgeGraph,
) -> tuple[list[SymbolInfo], list[SymbolInfo]]:
    """Find functions that call or are called by code in primary files."""
    callers: list[SymbolInfo] = []
    callees: list[SymbolInfo] = []

    for file_path in primary_files:
        for node in graph.find_nodes_in_file(file_path):
            # Find callers (edges pointing TO this node with type CALLS)
            for edge in graph.neighbors(node.id, direction="in"):
                source = graph.get_node(edge.source)
                if source and source.file_path != file_path:
                    callers.append(_to_symbol_info(source))
                    break

            # Find callees (edges pointing FROM this node with type CALLS)
            for edge in graph.neighbors(node.id, direction="out"):
                target = graph.get_node(edge.target)
                if target and target.file_path != file_path:
                    callees.append(_to_symbol_info(target))
                    break

    return callers, callees


def _find_siblings(
    primary_results: list[SearchResult],
    all_chunks: list[Chunk],
) -> list[Chunk]:
    """Find sibling chunks (same file, different ranges) of primary results."""
    siblings: list[Chunk] = []
    seen_ranges: set[tuple[str, int, int]] = {
        (r.chunk.file_path, r.chunk.start_line, r.chunk.end_line)
        for r in primary_results
    }

    for r in primary_results:
        file_path = r.chunk.file_path
        for chunk in all_chunks:
            if chunk.file_path != file_path:
                continue
            key = (chunk.file_path, chunk.start_line, chunk.end_line)
            if key not in seen_ranges:
                seen_ranges.add(key)
                siblings.append(chunk)

    return siblings[:20]  # Limit to prevent explosion


def _find_dependency_chain(
    primary_files: set[str],
    graph: KnowledgeGraph,
    depth: int,
) -> list[str]:
    """Find the dependency chain for primary files via graph expansion."""
    chain: list[str] = []
    visited: set[str] = set(primary_files)

    for file_path in primary_files:
        file_node = graph.find_file_node(file_path)
        if file_node is None:
            continue

        # Expand outward following IMPORTS edges
        for edge in graph.expand([file_node.id], depth=depth):
            target = graph.get_node(edge.target)
            if target and target.file_path and target.file_path not in visited:
                visited.add(target.file_path)
                chain.append(target.file_path)

    return chain


def _to_symbol_info(node: "GraphNode") -> SymbolInfo:
    """Convert a GraphNode to a SymbolInfo."""
    from vortexa.core.types import SymbolInfo as SI, SymbolKind
    return SI(
        name=node.label,
        kind=SymbolKind(node.kind.value) if isinstance(node.kind.value, str) else node.kind,
        file_path=node.file_path or "",
        start_line=node.metadata.get("start_line", 0),
        end_line=node.metadata.get("end_line", 0),
        docstring=node.metadata.get("docstring"),
        signature=node.metadata.get("signature"),
    )


def _build_trace(
    query: str,
    primary_files: set[str],
    test_files: list[str],
    imports_list: list[str],
    imported_by_list: list[str],
    symbols: list["SymbolInfo"],
    dependency_chain: list[str],
    confidence: float,
) -> list[str]:
    """Build a human-readable reasoning trace."""
    trace: list[str] = []
    trace.append(f"Query: \"{query}\"")
    trace.append(f"Primary files: {', '.join(sorted(primary_files)[:5])}")
    if test_files:
        trace.append(f"Related tests: {', '.join(test_files[:3])}")
    if imports_list:
        trace.append(f"Imported dependencies: {', '.join(imports_list[:5])}")
    if imported_by_list:
        trace.append(f"Imported by: {', '.join(imported_by_list[:3])}")
    if symbols:
        trace.append(f"Symbols found: {len(symbols)}")
    if dependency_chain:
        trace.append(f"Dependency chain: {len(dependency_chain)} files")
    trace.append(f"Confidence: {confidence:.3f}")
    return trace
