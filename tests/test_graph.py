"""Tests for the knowledge graph module."""

from vortexa.core.graph import KnowledgeGraph, build_graph_from_symbols
from vortexa.core.types import EdgeType, SymbolKind, SymbolInfo, ImportInfo


def test_add_node():
    graph = KnowledgeGraph()
    node = graph.add_node("file:test.py", SymbolKind.FILE, "test.py")
    assert graph.has_node("file:test.py")
    assert graph.node_count == 1
    assert node.label == "test.py"


def test_add_edge():
    graph = KnowledgeGraph()
    graph.add_node("file:a.py", SymbolKind.FILE, "a.py")
    graph.add_node("file:b.py", SymbolKind.FILE, "b.py")
    graph.add_edge("file:a.py", "file:b.py", EdgeType.IMPORTS)

    neighbors = graph.neighbors("file:a.py", direction="out")
    assert len(neighbors) == 1
    assert neighbors[0].target == "file:b.py"
    assert neighbors[0].type == EdgeType.IMPORTS


def test_expand():
    graph = KnowledgeGraph()
    graph.add_node("file:a.py", SymbolKind.FILE, "a.py")
    graph.add_node("file:b.py", SymbolKind.FILE, "b.py")
    graph.add_node("file:c.py", SymbolKind.FILE, "c.py")
    graph.add_edge("file:a.py", "file:b.py", EdgeType.IMPORTS)
    graph.add_edge("file:b.py", "file:c.py", EdgeType.IMPORTS)

    edges = graph.expand(["file:a.py"], depth=2)
    assert len(edges) == 2


def test_shortest_path():
    graph = KnowledgeGraph()
    graph.add_node("file:a.py", SymbolKind.FILE, "a.py")
    graph.add_node("file:b.py", SymbolKind.FILE, "b.py")
    graph.add_node("file:c.py", SymbolKind.FILE, "c.py")
    graph.add_edge("file:a.py", "file:b.py", EdgeType.IMPORTS)
    graph.add_edge("file:b.py", "file:c.py", EdgeType.IMPORTS)

    path = graph.shortest_path("file:a.py", "file:c.py")
    assert path is not None
    assert len(path) == 2


def test_find_nodes_by_name():
    graph = KnowledgeGraph()
    graph.add_node("class:MyClass", SymbolKind.CLASS, "MyClass")
    nodes = graph.find_nodes_by_name("MyClass")
    assert len(nodes) == 1


def test_find_nodes_in_file():
    graph = KnowledgeGraph()
    graph.add_node("file:test.py", SymbolKind.FILE, "test.py")
    graph.add_node("func:foo@test.py:1", SymbolKind.FUNCTION, "foo", file_path="test.py")
    nodes = graph.find_nodes_in_file("test.py")
    assert len(nodes) >= 1


def test_save_load(tmp_path):
    import shutil
    graph = KnowledgeGraph()
    graph.add_node("file:test.py", SymbolKind.FILE, "test.py")
    graph.add_node("func:foo@test.py:1", SymbolKind.FUNCTION, "foo", file_path="test.py")
    graph.add_edge("file:test.py", "func:foo@test.py:1", EdgeType.CONTAINS)

    save_dir = tmp_path / "graph"
    graph.save(save_dir)

    loaded = KnowledgeGraph.load(save_dir)
    assert loaded is not None
    assert loaded.node_count == 2
    assert loaded.find_nodes_by_name("foo") != []


def test_remove_node():
    graph = KnowledgeGraph()
    graph.add_node("file:test.py", SymbolKind.FILE, "test.py")
    graph.add_node("func:foo", SymbolKind.FUNCTION, "foo", file_path="test.py")
    graph.add_edge("file:test.py", "func:foo", EdgeType.CONTAINS)

    graph.remove_node("func:foo")
    assert not graph.has_node("func:foo")
    assert graph.node_count == 1


def test_build_from_symbols():
    symbols = [
        ("a.py", [
            SymbolInfo(name="foo", kind=SymbolKind.FUNCTION, file_path="a.py", start_line=1, end_line=5),
        ]),
    ]
    imports = [
        ("a.py", [
            ImportInfo(source_file="a.py", imported_module="os"),
        ]),
    ]
    files = ["a.py"]
    graph = build_graph_from_symbols(symbols, imports, files)
    assert graph.node_count >= 2  # file node + function node
