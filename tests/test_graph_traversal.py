"""Tests for V2 knowledge graph traversal, scoring, and analysis.

Covers the new methods added in graph.py:
- score_nodes_against_query (IDF-weighted three-tier matching)
- pick_seeds (gap-ratio seed selection)
- bfs_traverse / dfs_traverse (with hub thresholding + relation filtering)
- god_nodes (excludes file-level hubs)
- find_node (code-before-document priority)
- get_node_info / get_neighbors / shortest_path_between
- query_graph (full BFS/DFS rendering)
"""

from __future__ import annotations

from vortexa.core.graph import KnowledgeGraph, build_graph_from_symbols
from vortexa.core.types import (
    EdgeType,
    GraphTraversalMode,
    ImportInfo,
    SymbolInfo,
    SymbolKind,
)


def _make_graph() -> KnowledgeGraph:
    """Build a small but realistic graph for traversal tests."""
    g = KnowledgeGraph()
    # File nodes
    g.add_node("file:a.py", SymbolKind.FILE, "a.py", file_path="a.py")
    g.add_node("file:b.py", SymbolKind.FILE, "b.py", file_path="b.py")
    g.add_node("file:c.py", SymbolKind.FILE, "c.py", file_path="c.py")
    g.add_node("file:d.py", SymbolKind.FILE, "d.py", file_path="d.py")

    # Symbol nodes (code abstractions)
    g.add_node(
        "class:Foo", SymbolKind.CLASS, "Foo",
        file_path="a.py",
        metadata={"start_line": 1, "end_line": 10},
    )
    g.add_node(
        "class:Bar", SymbolKind.CLASS, "Bar",
        file_path="b.py",
        metadata={"start_line": 1, "end_line": 10},
    )
    g.add_node(
        "func:helper", SymbolKind.FUNCTION, "helper",
        file_path="c.py",
        metadata={"start_line": 1, "end_line": 5},
    )
    g.add_node(
        "class:Noise", SymbolKind.CLASS, "Noise",
        file_path="d.py",
        metadata={"start_line": 1, "end_line": 10},
    )

    # Edges
    g.add_edge("file:a.py", "class:Foo", EdgeType.CONTAINS)
    g.add_edge("file:b.py", "class:Bar", EdgeType.CONTAINS)
    g.add_edge("file:c.py", "func:helper", EdgeType.CONTAINS)
    g.add_edge("file:d.py", "class:Noise", EdgeType.CONTAINS)

    # Code structure: Foo calls helper, Bar calls helper, Noise calls Foo
    g.add_edge("class:Foo", "func:helper", EdgeType.CALLS)
    g.add_edge("class:Bar", "func:helper", EdgeType.CALLS)
    g.add_edge("class:Noise", "class:Foo", EdgeType.CALLS)

    # Imports
    g.add_edge("file:a.py", "file:b.py", EdgeType.IMPORTS)
    g.add_edge("file:c.py", "file:d.py", EdgeType.IMPORTS)

    return g


def test_score_nodes_against_query_exact_match_wins():
    """An exact label match outscores a substring match."""
    g = _make_graph()
    scored = g.score_nodes_against_query("Foo")
    assert scored, "expected at least one scored node"
    top_id = scored[0][1]
    top_node = g.get_node(top_id)
    assert top_node is not None
    assert top_node.label == "Foo"
    assert top_id == "class:Foo"


def test_score_nodes_against_query_no_match_returns_empty():
    g = _make_graph()
    assert g.score_nodes_against_query("zzzz_no_such_symbol") == []


def test_pick_seeds_stops_at_gap():
    """pick_seeds should refuse additional seeds once score drops below the gap ratio."""
    # Manual scenario: one high-scoring seed, two far below it
    scored = [(1000.0, "a"), (1.0, "b"), (0.5, "c")]
    seeds = KnowledgeGraph().pick_seeds(scored, max_k=3, gap_ratio=0.2)
    # gap_ratio=0.2 → additional seeds must score ≥ 200.0
    # b scores 1.0 < 200.0 → only a is seeded
    assert seeds == ["a"]


def test_pick_seeds_includes_close_scores():
    """When scores are close together, multiple seeds are kept."""
    scored = [(100.0, "a"), (90.0, "b"), (10.0, "c")]
    seeds = KnowledgeGraph().pick_seeds(scored, max_k=3, gap_ratio=0.2)
    # b is 90% of a → above 20% gap → kept; c is 10% → dropped
    assert seeds == ["a", "b"]


def test_pick_seeds_empty_input():
    assert KnowledgeGraph().pick_seeds([]) == []


def test_bfs_traverse_follows_calls_edge():
    """BFS from Foo should reach helper in one hop via the CALLS edge."""
    g = _make_graph()
    visited, edges = g.bfs_traverse(["class:Foo"], depth=2)
    assert "func:helper" in visited
    # We expect at least the Foo → helper call edge in the result
    rels = [r for _, r, _ in edges]
    assert "calls" in rels


def test_bfs_traverse_respects_depth_limit():
    """With depth=1, BFS should not visit nodes 2 hops away."""
    g = _make_graph()
    visited, _ = g.bfs_traverse(["class:Foo"], depth=1)
    # helper is a direct call target (1 hop) → reached
    assert "func:helper" in visited
    # Noise → Foo (so from Foo the noise would be 1 hop, but Noise → Foo is the
    # recorded edge, not Foo → Noise). Use a graph where this is provable:
    # From file:a.py at depth 1, only file:a.py's direct neighbours are reached.
    g2 = _make_graph()
    visited2, _ = g2.bfs_traverse(["file:a.py"], depth=1)
    # a imports b directly → b is reachable; d requires a → b → ... → d
    assert "file:b.py" in visited2
    # d is reachable via a → b → ... depending on graph topology; at depth 1
    # from a, only direct neighbours of a (b via IMPORTS, class:Foo via CONTAINS)
    assert "file:d.py" not in visited2


def test_bfs_traverse_relation_filter():
    """relation_filter should restrict which edges are traversed."""
    g = _make_graph()
    # Only allow contains edges
    visited, _ = g.bfs_traverse(
        ["file:a.py"], depth=2, relation_filter={"contains"}
    )
    # contains edges: file → class. So from a.py at depth 1 we reach class:Foo.
    # Going through Foo → helper is a CALLS edge, filtered out.
    assert "class:Foo" in visited
    assert "func:helper" not in visited


def test_dfs_traverse_returns_visited():
    """DFS should also reach direct neighbours."""
    g = _make_graph()
    visited, edges = g.dfs_traverse(["class:Foo"], depth=2)
    assert "func:helper" in visited


def test_god_nodes_excludes_file_hubs():
    """File nodes should NOT appear in god_nodes even if they have many edges."""
    g = _make_graph()
    gods = g.god_nodes(top_n=10)
    labels = [n.label for n in gods]
    # file-level hubs are excluded
    assert not any(l.endswith(".py") for l in labels)
    # but real classes/functions surface
    assert "Foo" in labels or "Bar" in labels or "Noise" in labels or "helper" in labels


def test_god_nodes_respects_top_n():
    g = _make_graph()
    assert len(g.god_nodes(top_n=2)) <= 2


def test_god_nodes_empty_graph():
    g = KnowledgeGraph()
    assert g.god_nodes() == []


def test_find_node_exact_match_preferred():
    """An exact label match should be the first result."""
    g = _make_graph()
    matches = g.find_node("Foo")
    assert matches[0] == "class:Foo"


def test_find_node_returns_empty_for_no_match():
    g = _make_graph()
    assert g.find_node("zzzz_doesnt_exist") == []


def test_get_node_info_returns_details():
    g = _make_graph()
    info = g.get_node_info("Foo")
    assert info is not None
    assert info.label == "Foo"
    assert info.kind == "class"
    assert info.file_path == "a.py"
    assert info.degree >= 1


def test_get_node_info_returns_none_when_missing():
    g = _make_graph()
    assert g.get_node_info("nonexistent_symbol") is None


def test_get_neighbors_returns_both_directions():
    g = _make_graph()
    edges = g.get_neighbors("Foo")
    directions = {e.direction for e in edges}
    # Foo calls helper (out) and Noise calls Foo (in)
    assert "out" in directions
    assert "in" in directions


def test_get_neighbors_empty_for_unknown_node():
    g = _make_graph()
    assert g.get_neighbors("zzz_missing") == []


def test_shortest_path_between_finds_two_hop_path():
    """Path: class:Foo (caller) → func:helper (callee) via CALLS edge."""
    g = _make_graph()
    path = g.shortest_path_between("Foo", "helper")
    assert path is not None
    assert path.hops == 1
    assert len(path.segments) == 1
    src, rel, tgt = path.segments[0]
    assert "foo" in src
    assert rel == "calls"
    assert "helper" in tgt


def test_shortest_path_between_same_node():
    g = _make_graph()
    path = g.shortest_path_between("Foo", "Foo")
    assert path is not None
    assert path.hops == 0
    assert path.segments == ()


def test_shortest_path_returns_none_when_no_match():
    g = _make_graph()
    assert g.shortest_path_between("Foo", "zzz_nonexistent") is None


def test_query_graph_returns_text():
    g = _make_graph()
    output = g.query_graph("Foo")
    assert isinstance(output, str)
    assert "Traversal:" in output
    assert "Foo" in output


def test_query_graph_handles_no_match():
    g = _make_graph()
    output = g.query_graph("zzz_definitely_not_in_graph")
    assert output == "No matching nodes found."


def test_query_graph_respects_mode_and_depth():
    g = _make_graph()
    bfs_out = g.query_graph("Foo", mode=GraphTraversalMode.BFS, depth=2)
    dfs_out = g.query_graph("Foo", mode=GraphTraversalMode.DFS, depth=2)
    assert isinstance(bfs_out, str)
    assert isinstance(dfs_out, str)
    assert "BFS" in bfs_out
    assert "DFS" in dfs_out


def test_query_graph_truncates_at_token_budget():
    """Output should be capped when token_budget is very small."""
    g = _make_graph()
    output = g.query_graph("Foo", token_budget=50)
    # With a tight budget the output should be either the full short text
    # or include the truncation marker.
    assert len(output) <= 50 * 3 + 200  # allow some slack for header


def test_build_graph_from_symbols_smoke():
    """Existing graph builder should still work with the new module layout."""
    symbols = [
        ("a.py", [
            SymbolInfo(
                name="Foo", kind=SymbolKind.CLASS,
                file_path="a.py", start_line=1, end_line=10,
            ),
        ]),
    ]
    imports = [
        ("a.py", [
            ImportInfo(source_file="a.py", imported_module="b"),
        ]),
    ]
    files = ["a.py"]
    g = build_graph_from_symbols(symbols, imports, files)
    assert g.node_count >= 2
    assert any(n.label == "Foo" for n in g._nodes.values())