"""Tests for context expansion."""

from vortexa.core.graph import KnowledgeGraph
from vortexa.core.types import (
    Chunk, ContextPack, EdgeType, SearchMode, SearchResult, SymbolKind,
)
from vortexa.search.context_expansion import expand_context


def _make_chunk(file_path: str, content: str = "def foo(): pass") -> Chunk:
    return Chunk(content=content, file_path=file_path, start_line=1, end_line=2)


def _make_result(file_path: str, score: float = 1.0) -> SearchResult:
    return SearchResult(
        chunk=_make_chunk(file_path),
        score=score,
        source=SearchMode.HYBRID,
    )


def test_expand_no_graph():
    graph = KnowledgeGraph()
    results = [_make_result("test.py")]
    pack = expand_context("test query", results, graph, [])
    assert len(pack.primary_chunks) == 1
    assert pack.confidence > 0


def test_expand_with_imports():
    graph = KnowledgeGraph()
    graph.add_node("file:main.py", SymbolKind.FILE, "main.py", file_path="main.py")
    graph.add_node("file:utils.py", SymbolKind.FILE, "utils.py", file_path="utils.py")
    graph.add_edge("file:main.py", "file:utils.py", EdgeType.IMPORTS)

    results = [_make_result("main.py")]
    pack = expand_context("test", results, graph, [])
    assert "utils.py" in pack.imports or "utils.py" in pack.related_files


def test_expand_finds_test_files():
    graph = KnowledgeGraph()
    graph.add_node("file:service.py", SymbolKind.FILE, "service.py")

    results = [_make_result("service.py")]
    pack = expand_context("test", results, graph, [])
    assert any("test" in f for f in pack.test_files)


def test_expand_empty_results():
    graph = KnowledgeGraph()
    pack = expand_context("test", [], graph, [])
    assert pack.query == "test"
    assert len(pack.primary_chunks) == 0
    assert pack.confidence == 0.0
