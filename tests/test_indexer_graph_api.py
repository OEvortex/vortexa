"""Tests for V2 indexer graph API + hybrid search enrichment.

Covers:
- query_graph on the indexer (delegates to KnowledgeGraph)
- get_god_nodes / get_graph_node / get_graph_neighbors / get_shortest_path
- _attach_hybrid_context enriches results with GraphContext when hybrid=True
"""

from __future__ import annotations

from unittest.mock import MagicMock

from vortexa.core.graph import KnowledgeGraph
from vortexa.core.indexer import CodebaseIndexer, SearchResultWithContext
from vortexa.core.types import (
    Chunk,
    EdgeType,
    GRAPH_STRUCTURAL_RELATIONS,
    GraphContext,
    GraphNode,
    SearchMode,
    SearchResult,
    SymbolKind,
)


def _stub_indexer(tmp_path) -> CodebaseIndexer:
    """Build a CodebaseIndexer that doesn't actually index anything.

    Returns one with chunks/graph pre-populated so we can exercise the
    graph API + hybrid enrichment paths without going through the full
    index pipeline (which needs an embedding model).
    """
    idx = CodebaseIndexer.__new__(CodebaseIndexer)
    idx.root = tmp_path
    idx.index_dir = tmp_path / ".jarvis" / "index"
    idx._graph_dir = idx.index_dir / "graph"
    idx._vortex_config = MagicMock()
    idx.session = MagicMock()
    idx.graph = KnowledgeGraph()
    idx.chunks = []
    idx.chunk_ids = []
    idx.file_hashes = {}
    idx.chunk_memo = {}
    idx._vector_store = None
    idx._bm25_index = None
    idx._file_index = None
    idx._function_index = None
    idx._symbol_index = None
    idx._file_index_data = []
    idx._function_index_data = []
    idx._symbol_index_data = []
    idx._memo_hits = 0
    idx._memo_misses = 0
    idx._embedder = None
    idx._model = MagicMock()
    return idx


def _make_result(file_path: str, score: float = 1.0) -> SearchResult:
    chunk = Chunk(
        content=f"def foo():\n    return 'hi from {file_path}'",
        file_path=file_path,
        start_line=1,
        end_line=2,
    )
    return SearchResult(chunk=chunk, score=score, source=SearchMode.HYBRID)


def test_indexer_query_graph_delegates_to_graph():
    """indexer.query_graph should call KnowledgeGraph.query_graph with normalised args."""
    idx = _stub_indexer(__import__("pathlib").Path("/tmp/vortexa_test_xyz"))
    idx.graph.add_node("class:Foo", SymbolKind.CLASS, "Foo", file_path="a.py")
    idx.graph.add_node("class:Bar", SymbolKind.CLASS, "Bar", file_path="b.py")
    idx.graph.add_edge("class:Foo", "class:Bar", EdgeType.CALLS)

    output = idx.query_graph("Foo", mode="BFS", depth=2)
    assert "Traversal:" in output
    assert "Foo" in output


def test_indexer_query_graph_invalid_mode_falls_back_to_bfs():
    idx = _stub_indexer(__import__("pathlib").Path("/tmp/vortexa_test_xyz"))
    idx.graph.add_node("class:Foo", SymbolKind.CLASS, "Foo", file_path="a.py")
    out = idx.query_graph("Foo", mode="garbage_mode", depth=2)
    # Should silently fall back to BFS
    assert "BFS" in out


def test_indexer_query_graph_clamps_depth():
    idx = _stub_indexer(__import__("pathlib").Path("/tmp/vortexa_test_xyz"))
    idx.graph.add_node("class:Foo", SymbolKind.CLASS, "Foo", file_path="a.py")
    # depth=99 should be clamped to 6 (max allowed)
    out = idx.query_graph("Foo", depth=99)
    assert "depth=6" in out


def test_indexer_get_god_nodes_returns_serialisable_list():
    idx = _stub_indexer(__import__("pathlib").Path("/tmp/vortexa_test_xyz"))
    idx.graph.add_node("class:Foo", SymbolKind.CLASS, "Foo", file_path="a.py")
    idx.graph.add_node("class:Bar", SymbolKind.CLASS, "Bar", file_path="b.py")
    idx.graph.add_edge("class:Foo", "class:Bar", EdgeType.CALLS)
    gods = idx.get_god_nodes(top_n=5)
    assert isinstance(gods, list)
    assert all(isinstance(n, dict) for n in gods)
    assert all("id" in n and "label" in n and "degree" in n for n in gods)


def test_indexer_get_graph_node_returns_dict_or_none():
    idx = _stub_indexer(__import__("pathlib").Path("/tmp/vortexa_test_xyz"))
    idx.graph.add_node(
        "class:Foo", SymbolKind.CLASS, "Foo", file_path="a.py",
        metadata={"start_line": 1, "end_line": 10},
    )
    found = idx.get_graph_node("Foo")
    assert found is not None
    assert found["label"] == "Foo"
    assert found["kind"] == "class"
    missing = idx.get_graph_node("nonexistent_symbol_zzz")
    assert missing is None


def test_indexer_get_graph_neighbors_returns_serialisable_list():
    idx = _stub_indexer(__import__("pathlib").Path("/tmp/vortexa_test_xyz"))
    idx.graph.add_node("class:Foo", SymbolKind.CLASS, "Foo", file_path="a.py")
    idx.graph.add_node("class:Bar", SymbolKind.CLASS, "Bar", file_path="b.py")
    idx.graph.add_edge("class:Foo", "class:Bar", EdgeType.CALLS)
    edges = idx.get_graph_neighbors("Foo")
    assert isinstance(edges, list)
    assert edges, "expected at least one edge"
    for e in edges:
        assert set(e.keys()) == {"source", "target", "relation", "direction"}


def test_indexer_get_shortest_path_returns_graph_path():
    from vortexa.core.types import GraphPath

    idx = _stub_indexer(__import__("pathlib").Path("/tmp/vortexa_test_xyz"))
    idx.graph.add_node("class:Foo", SymbolKind.CLASS, "Foo", file_path="a.py")
    idx.graph.add_node("class:Bar", SymbolKind.CLASS, "Bar", file_path="b.py")
    idx.graph.add_edge("class:Foo", "class:Bar", EdgeType.CALLS)
    path = idx.get_shortest_path("Foo", "Bar")
    assert isinstance(path, GraphPath)
    assert path.hops == 1


def test_attach_hybrid_context_enriches_results():
    """hybrid=True should attach a GraphContext to each result file."""
    idx = _stub_indexer(__import__("pathlib").Path("/tmp/vortexa_test_xyz"))
    # Build a small graph: Foo class in a.py with an import edge
    idx.graph.add_node(
        "class:Foo", SymbolKind.CLASS, "Foo",
        file_path="a.py",
    )
    idx.graph.add_node(
        "class:Bar", SymbolKind.CLASS, "Bar",
        file_path="b.py",
    )
    idx.graph.add_edge("class:Foo", "class:Bar", EdgeType.CALLS)
    # Add file node so file-level edges exist
    idx.graph.add_node("file:a.py", SymbolKind.FILE, "a.py", file_path="a.py")
    idx.graph.add_node("file:b.py", SymbolKind.FILE, "b.py", file_path="b.py")
    idx.graph.add_edge("file:a.py", "class:Foo", EdgeType.CONTAINS)
    idx.graph.add_edge("file:b.py", "class:Bar", EdgeType.CONTAINS)

    results = [_make_result("a.py")]
    enriched = idx._attach_hybrid_context(results, query="Foo")

    assert len(enriched) == 1
    assert isinstance(enriched[0], SearchResultWithContext)
    ctx = enriched[0].context
    assert isinstance(ctx, GraphContext)
    # Foo should be the key symbol for the a.py result file
    assert ctx.key_symbol == "Foo"


def test_attach_hybrid_context_skips_files_without_nodes():
    """Files with no graph nodes should pass through unenriched."""
    idx = _stub_indexer(__import__("pathlib").Path("/tmp/vortexa_test_xyz"))
    results = [_make_result("unknown.py")]
    enriched = idx._attach_hybrid_context(results, query="anything")
    # Result should pass through unchanged (no wrapper, no context)
    assert not isinstance(enriched[0], SearchResultWithContext)


def test_search_result_with_context_delegates_attributes():
    """SearchResultWithContext should expose the underlying SearchResult's attrs."""
    inner = _make_result("a.py", score=0.42)
    ctx = GraphContext(key_symbol="Foo", incoming=("Bar (imports)",), outgoing=())
    wrapped = SearchResultWithContext(result=inner, context=ctx)
    # Chunk / score / source all delegate
    assert wrapped.chunk.file_path == "a.py"
    assert wrapped.score == 0.42
    assert wrapped.source == SearchMode.HYBRID
    # Plus the context attribute is its own
    assert wrapped.context is ctx


def test_attach_hybrid_context_uses_structural_relations_filter():
    """The enrichment should only surface structural edges (call/import/etc),
    not type-annotation noise (references)."""
    idx = _stub_indexer(__import__("pathlib").Path("/tmp/vortexa_test_xyz"))
    idx.graph.add_node("class:Foo", SymbolKind.CLASS, "Foo", file_path="a.py")
    idx.graph.add_node("class:Bar", SymbolKind.CLASS, "Bar", file_path="b.py")
    idx.graph.add_node("class:Baz", SymbolKind.CLASS, "Baz", file_path="c.py")
    # Structural edge
    idx.graph.add_edge("class:Foo", "class:Bar", EdgeType.CALLS)
    # Non-structural edge (prose reference, not in GRAPH_STRUCTURAL_RELATIONS)
    idx.graph.add_edge("class:Foo", "class:Baz", EdgeType.REFERENCES)
    # Sanity check our filter set
    assert "calls" in GRAPH_STRUCTURAL_RELATIONS

    enriched = idx._attach_hybrid_context([_make_result("a.py")], query="Foo")
    ctx = enriched[0].context
    # Only the structural CALLS edge should appear in outgoing
    assert any("calls" in o for o in ctx.outgoing)
    assert not any("references" in o for o in ctx.outgoing)