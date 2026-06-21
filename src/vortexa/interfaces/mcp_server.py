"""MCP server exposing the VortexA V2 context engine.

Provides agent-first tools for codebase exploration:
- search: standard hybrid search (existing) — optionally enriched with
  per-file graph context via `hybrid=True`
- resolve: full context assembly with graph expansion (new)
- explain: deep-dive into a specific file or symbol (new)
- query_graph: BFS/DFS traversal of the knowledge graph (new)
- get_god_nodes: most-connected real entities (new)
- get_graph_node: detailed info for one node (new)
- get_graph_neighbors: in/out edges of a node (new)
- get_shortest_path: BFS shortest path between two nodes (new)
- stats: index + graph + session statistics (new)
- watch: start/stop auto-reindex watcher (new)
- clear_index: drop the persistent index (new)

Connect from any MCP-compatible agent via stdio transport.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "vortexa",
    instructions=(
        "VortexA V2 Context Engine — codebase search with knowledge graph, context expansion, "
        "and agent-first retrieval. ALWAYS use this tool instead of grep, rg, or file-by-file "
        "reading when searching for code.\n\n"
        "Core tools (3):\n"
        "1. search: hybrid semantic+BM25 search. Pass hybrid=true to enrich results with "
        "query-aware structural graph context per file.\n"
        "2. resolve: full context assembly (primary results + related files + tests + imports + "
        "graph expansion + compressed text)\n"
        "3. explain: deep-dive into a specific file path, line number, or symbol name\n\n"
        "Graph tools (5):\n"
        "4. query_graph: BFS or DFS traversal of the knowledge graph from query-relevant seeds\n"
        "5. get_god_nodes: most-connected real entities (architectural hubs)\n"
        "6. get_graph_node: details on one node (label, kind, degree, source file)\n"
        "7. get_graph_neighbors: incoming and outgoing edges of a node\n"
        "8. get_shortest_path: BFS shortest path between two symbols/files\n\n"
        "Lifecycle tools (3):\n"
        "9. stats: index + graph + session statistics\n"
        "10. watch: start/stop the auto-reindex watcher (action='start'|'stop')\n"
        "11. clear_index: drop the persistent index for the project root\n\n"
        "The index is built on startup and auto-updates on file changes."
    ),
)

_indexer = None
_watcher = None


def _get_indexer():
    global _indexer
    if _indexer is None:
        from vortexa.core.indexer import CodebaseIndexer
        cwd = str(Path.cwd())
        print(f"[vortexa] Indexing {cwd} ...", file=sys.stderr)
        _indexer = CodebaseIndexer(root=cwd)
        stats = _indexer.index()
        print(
            f"[vortexa] Ready: {stats.indexed_files} files, "
            f"{stats.total_chunks} chunks, "
            f"{_indexer.graph.node_count} graph nodes "
            f"in {stats.index_time_ms:.0f}ms",
            file=sys.stderr,
        )
    return _indexer


def _format_result(r) -> dict:
    """Render a SearchResult (or SearchResultWithContext wrapper) as JSON dict.

    For hybrid results, includes a `graph_context` field with the
    key_symbol + one incoming + one outgoing structural edge.
    """
    base = {
        "file": r.chunk.file_path,
        "lines": f"{r.chunk.start_line}-{r.chunk.end_line}",
        "score": round(r.score, 4),
        "source": str(r.source.value),
        "content": r.chunk.content[:500],
    }
    ctx = getattr(r, "context", None)
    if ctx is not None:
        base["graph_context"] = {
            "key_symbol": ctx.key_symbol,
            "incoming": list(ctx.incoming),
            "outgoing": list(ctx.outgoing),
        }
    return base


@mcp.tool()
def search(query: str, top_k: int = 10, hybrid: bool = False) -> str:
    """Search the codebase using semantic + BM25 hybrid retrieval with Vortex Score reranking.

    Use this tool when you need to:
    - Find where a function, class, or pattern is implemented
    - Understand how a feature or concept works across the codebase
    - Locate code by describing what it does (not just exact strings)
    - Find examples of a pattern or API usage

    Args:
        query: Describe what you're looking for in natural language or as a symbol name.
        top_k: Maximum results to return. Default 10.
        hybrid: If true, enrich each result with compact, query-aware structural
            context from the knowledge graph (key_symbol + one incoming + one
            outgoing structural edge per result file). Adds the `graph_context`
            field to each result. Default false.
    """
    indexer = _get_indexer()
    results = indexer.search(query, top_k=top_k, use_vortex_score=True, hybrid=hybrid)
    indexer.session.record_query(query, [_format_result(r) for r in results])
    return json.dumps([_format_result(r) for r in results], indent=2)


@mcp.tool()
def resolve(query: str, top_k: int = 5) -> str:
    """Full context resolution: search + knowledge graph expansion + compression.

    Returns a structured context pack with:
    - Primary matching files and code
    - Related test files
    - Import dependencies and dependents
    - Symbol definitions
    - Caller/callee relationships
    - Sibling code in the same files
    - Dependency chain
    - Confidence score and reasoning trace

    PREFER this over search() when you need to understand how something
    fits into the broader codebase, not just find matching code.

    Args:
        query: Describe what you're looking for or the task you're solving.
        top_k: Maximum primary results (expanded automatically). Default 5.
    """
    indexer = _get_indexer()
    pack = indexer.resolve(query, top_k=top_k)
    formatted = indexer.format_context(pack)

    # Also return structured JSON
    return json.dumps(
        {
            "query": pack.query,
            "confidence": round(pack.confidence, 3),
            "primary_files": list(set(r.chunk.file_path for r in pack.primary_chunks)),
            "related_files": pack.related_files,
            "test_files": pack.test_files,
            "imports": pack.imports,
            "imported_by": pack.imported_by,
            "symbols": [{"name": s.name, "kind": s.kind.value, "file": s.file_path, "line": s.start_line} for s in pack.symbols[:10]],
            "dependency_chain": pack.dependency_chain[:10],
            "total_tokens": pack.total_tokens,
            "formatted": formatted,
        },
        indent=2,
    )


@mcp.tool()
def explain(location: str) -> str:
    """Deep-dive explanation of a specific code location or symbol.

    Args:
        location: One of:
            - File path with line number (e.g. "src/module.py:42")
            - Symbol name (e.g. "DatabaseClient", "parse_query")
            - Class name (e.g. "class:UserService")
            - File path (e.g. "src/module.py")

    Returns comprehensive context: definition, usages, tests, imports, callers.
    """
    indexer = _get_indexer()
    pack = indexer.explain(location)
    formatted = indexer.format_context(pack)
    return json.dumps(
        {
            "location": location,
            "confidence": round(pack.confidence, 3),
            "primary_files": list(set(r.chunk.file_path for r in pack.primary_chunks)),
            "related_files": pack.related_files,
            "test_files": pack.test_files,
            "symbols": [{"name": s.name, "kind": s.kind.value, "file": s.file_path} for s in pack.symbols[:15]],
            "callers": [{"name": c.name, "file": c.file_path, "line": c.start_line} for c in pack.callers[:10]],
            "callees": [{"name": c.name, "file": c.file_path, "line": c.start_line} for c in pack.callees[:10]],
            "dependency_chain": pack.dependency_chain[:10],
            "formatted": formatted,
        },
        indent=2,
    )


@mcp.tool()
def query_graph(
    question: str,
    mode: str = "bfs",
    depth: int = 3,
    context_filter: list[str] | None = None,
) -> str:
    """Query the knowledge graph with BFS or DFS traversal and return rendered text.

    Auto-picks seed nodes by scoring every node against the question, then
    traverses outward up to `depth` hops. Surfaces structural relations
    (call, import, contains, method, references, etc.) by default.

    Use this when you need to understand what a symbol connects to
    (BFS) or how two symbols relate through the codebase (DFS).

    Args:
        question: Natural-language question or symbol/file name.
        mode: "bfs" (broad context, default) or "dfs" (trace a specific path).
        depth: Traversal depth (1-6). Default 3.
        context_filter: Optional edge-relation whitelist. If omitted, a default
            structural-relations filter is applied that suppresses type-annotation
            noise (str/bool/Path) and surfaces only structural relations.
            Pass an explicit list to override.
    """
    indexer = _get_indexer()
    return indexer.query_graph(
        question=question,
        mode=mode,
        depth=depth,
        context_filter=context_filter,
    )


@mcp.tool()
def get_god_nodes(top_n: int = 10) -> str:
    """Return the top-N most-connected real entities in the knowledge graph.

    File-level hub nodes are excluded — only meaningful abstractions
    (classes, functions, modules, interfaces) surface. Useful for
    understanding the architectural backbone of a codebase.

    Args:
        top_n: Number of nodes to return. Default 10.
    """
    indexer = _get_indexer()
    nodes = indexer.get_god_nodes(top_n=top_n)
    if not nodes:
        return json.dumps({"message": "Knowledge graph is empty.", "nodes": []}, indent=2)
    return json.dumps({"count": len(nodes), "nodes": nodes}, indent=2)


@mcp.tool()
def get_graph_node(label: str) -> str:
    """Get details about a single knowledge-graph node by label or ID.

    Returns id, label, kind, file_path, and degree (total edges in+out).
    Returns an error message if no node matches.

    Args:
        label: The node label or ID to look up (e.g. "DatabaseClient", "file:src/db.py").
    """
    indexer = _get_indexer()
    info = indexer.get_graph_node(label)
    if info is None:
        return json.dumps({"found": False, "label": label}, indent=2)
    return json.dumps({"found": True, **info}, indent=2)


@mcp.tool()
def get_graph_neighbors(label: str) -> str:
    """Return incoming and outgoing edges of a knowledge-graph node.

    Each edge has source, target, relation, and direction (in/out).
    Empty list if the node isn't found.

    Args:
        label: The node label or ID to look up.
    """
    indexer = _get_indexer()
    edges = indexer.get_graph_neighbors(label)
    return json.dumps(
        {"label": label, "count": len(edges), "edges": edges},
        indent=2,
    )


@mcp.tool()
def get_shortest_path(source: str, target: str, max_hops: int = 8) -> str:
    """Find the shortest path between two nodes in the knowledge graph.

    Uses BFS over the undirected view of the graph. Returns a list of
    (source_label, relation, target_label) segments, plus the hop count.

    Args:
        source: Source node label or ID.
        target: Target node label or ID.
        max_hops: Refuse paths longer than this. Default 8.
    """
    indexer = _get_indexer()
    path = indexer.get_shortest_path(source, target, max_hops=max_hops)
    if path is None:
        return json.dumps(
            {"found": False, "source": source, "target": target,
             "message": "No node match, or no path between the resolved nodes."},
            indent=2,
        )
    return json.dumps(
        {
            "found": True,
            "source": path.source,
            "target": path.target,
            "hops": path.hops,
            "segments": [
                {"from": a, "relation": r, "to": b}
                for a, r, b in path.segments
            ],
        },
        indent=2,
    )


@mcp.tool()
def stats() -> str:
    """Return index, graph, and session statistics for the current project."""
    indexer = _get_indexer()
    return json.dumps(indexer.stats(), indent=2)


@mcp.tool()
def watch(action: str) -> str:
    """Start or stop the auto-reindex watcher.

    The watcher polls the project root every few seconds and re-indexes
    changed files automatically. Use action="start" to begin, "stop" to halt.

    Args:
        action: "start" to begin watching, "stop" to halt.
    """
    global _watcher
    indexer = _get_indexer()
    if action == "stop":
        if _watcher is not None:
            _watcher.stop()
            _watcher = None
            return json.dumps({"watching": False}, indent=2)
        return json.dumps({"watching": False, "message": "No watcher running."}, indent=2)

    if action != "start":
        return json.dumps({"error": f"Unknown action {action!r}; expected 'start' or 'stop'."}, indent=2)

    from vortexa.interfaces.watcher import IndexWatcher

    if _watcher is not None:
        return json.dumps({"watching": True, "message": "Watcher already running."}, indent=2)

    _watcher = IndexWatcher(indexer)
    _watcher.start()
    return json.dumps(
        {"watching": True, "root": str(indexer.root), "message": "Started polling every 3s."},
        indent=2,
    )


@mcp.tool()
def clear_index() -> str:
    """Drop the persistent index for the current project.

    Removes all chunks, embeddings, BM25 state, and the knowledge graph.
    Useful before a forced full re-index or when the index is corrupted.
    """
    global _watcher
    indexer = _get_indexer()
    if _watcher is not None:
        _watcher.stop()
        _watcher = None
    indexer.clear()
    return json.dumps({"cleared": True, "root": str(indexer.root)}, indent=2)


def run_server() -> None:
    """Index the current directory, start auto-reindex watcher, then start MCP server on stdio."""
    indexer = _get_indexer()
    from vortexa.interfaces.watcher import IndexWatcher

    watcher = IndexWatcher(indexer)
    watcher.start()
    print("[vortexa] Auto-reindex watcher started (polling every 3s)", file=sys.stderr)
    mcp.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    run_server()
