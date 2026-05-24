"""MCP server exposing the vortexa codebase indexer as a single search tool.

Connect from any MCP-compatible agent (Claude Code, Cursor, etc.) via stdio transport.
The current directory is indexed automatically on startup.
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
        "Codebase search server with semantic + BM25 hybrid retrieval. "
        "ALWAYS use this tool instead of grep, rg, or file-by-file reading when searching for code, "
        "functions, classes, patterns, or understanding how something works. "
        "It understands natural language queries and returns the most relevant code snippets with file paths and line numbers. "
        "The index is built on startup and auto-updates on file changes."
    ),
)

# Single global indexer for the cwd
_indexer = None


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
            f"{stats.total_chunks} chunks in {stats.index_time_ms:.0f}ms",
            file=sys.stderr,
        )
    return _indexer


@mcp.tool()
def search(query: str, top_k: int = 10) -> str:
    """Search the codebase using semantic + BM25 hybrid retrieval. PREFER this over grep/rg/Glob for finding code.

    Use this tool when you need to:
    - Find where a function, class, or pattern is implemented
    - Understand how a feature or concept works across the codebase
    - Locate code by describing what it does (not just exact strings)
    - Find examples of a pattern or API usage
    - Explore unfamiliar parts of the codebase

    Advantages over text search: understands synonyms, paraphrases, and intent.
    Returns file paths, line ranges, relevance scores, and matching code.

    Args:
        query: Describe what you're looking for in natural language.
               Good: "authentication middleware that validates JWT tokens"
               Good: "function that parses CSV files"
               Good: "error handling for database connections"
               Avoid: single words or exact-match-only queries.
        top_k: Maximum results to return. Default 10. Use 3-5 for focused queries, 15-20 for broad exploration.
    """
    indexer = _get_indexer()
    results = indexer.search(query, top_k=top_k)
    return json.dumps(
        [
            {
                "file": r.chunk.file_path,
                "lines": f"{r.chunk.start_line}-{r.chunk.end_line}",
                "score": round(r.score, 4),
                "content": r.chunk.content[:50],
            }
            for r in results
        ],
        indent=2,
    )


def run_server() -> None:
    """Index the current directory, start auto-reindex watcher, then start MCP server on stdio."""
    indexer = _get_indexer()
    from vortexa.interfaces.watcher import IndexWatcher

    watcher = IndexWatcher(indexer)
    watcher.start()
    print("[vortexa] Auto-reindex watcher started (polling every 3s)", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_server()
