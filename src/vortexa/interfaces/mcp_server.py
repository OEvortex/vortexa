"""MCP server exposing the vortexa codebase indexer.

Connect from any MCP-compatible agent (Claude Code, Cursor, etc.) via stdio
transport. The current directory is indexed automatically on startup.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from fastmcp import Context, FastMCP  # ty: ignore[unresolved-import, unused-ignore-comment]

from vortexa import __version__
from vortexa.core.progress import IndexProgress, ProgressCallback
from vortexa.interfaces.progress import IndexProgressBar

mcp = FastMCP(
    "vortexa",
    version=__version__,
    instructions=(
        "Codebase search server with semantic + BM25 hybrid retrieval. "
        "Use search for discovery, resolve for feature-level graph context, and "
        "explain for known files or symbols. Prefer these tools over grep, rg, "
        "Glob, or broad file-by-file reading when exploring codebases. Call "
        "status to check whether indexing is still running or ready, and "
        "reindex only after large external changes. Results include file paths, "
        "line ranges, scores, source, language, and code snippets. The index is "
        "built on startup with a realtime progress bar on stderr and auto-updates "
        "on file changes."
    ),
)

# Single global indexer for the server root.
_indexer = None
_index_attempted = False
_init_lock = threading.RLock()

_server_root: Path | None = None
_server_model_id: str | None = None
_server_include_text = False
_show_progress = True


def _resolve_server_root(root: str | Path | None) -> Path:
    configured = root or os.environ.get("VORTEXA_ROOT") or Path.cwd()
    return Path(configured).expanduser().resolve()


def _new_progress_bar(root: Path) -> IndexProgressBar:
    return IndexProgressBar(title=f"Indexing {root}", enabled=_show_progress)


def _get_indexer(progress: ProgressCallback | None = None, *, ensure_indexed: bool = True):
    """Create and lazily index the server's repository once."""
    global _indexer, _index_attempted
    with _init_lock:
        if _indexer is None:
            from vortexa.core.indexer import CodebaseIndexer

            root = _resolve_server_root(_server_root)
            _indexer = CodebaseIndexer(root=root, model_id=_server_model_id)

        if ensure_indexed and not _index_attempted:
            bar = progress
            if bar is None and _show_progress:
                bar = _new_progress_bar(_indexer.root)
            stats = _indexer.index(
                include_text_files=_server_include_text,
                progress=bar,
            )
            _index_attempted = True
            print(
                f"[vortexa] Ready: {stats.indexed_files} files, "
                f"{stats.total_chunks} chunks in {stats.index_time_ms:.0f}ms",
                file=sys.stderr,
            )
        return _indexer


def _progress_message(event: IndexProgress) -> str:
    if event.detail:
        return f"{event.message or event.phase}: {event.detail}"
    return event.message or event.phase


def _tool_progress(ctx: Context, loop: asyncio.AbstractEventLoop):
    bar = _new_progress_bar(_resolve_server_root(_server_root))

    def report(event: IndexProgress) -> None:
        bar(event)
        total = event.total if event.total > 0 else None
        try:
            asyncio.run_coroutine_threadsafe(
                ctx.report_progress(
                    progress=event.current,
                    total=total,
                    message=_progress_message(event),
                ),
                loop,
            )
        except RuntimeError:
            # The event loop can close while a final callback is in flight.
            pass

    return report


def _stats_payload(indexer, stats) -> dict:
    return {
        "root": str(indexer.root),
        "index_dir": str(indexer.index_dir),
        "model": indexer.model_id,
        "indexed_files": stats.indexed_files,
        "total_chunks": stats.total_chunks,
        "languages": stats.languages,
        "changed_files": stats.changed_files,
        "removed_files": stats.removed_files,
        "memo_hits": stats.memo_hits,
        "memo_misses": stats.memo_misses,
        "embedded_chunks": stats.embedded_chunks,
        "up_to_date": stats.up_to_date,
        "index_time_ms": stats.index_time_ms,
        "errors": list(stats.errors),
    }


@mcp.tool()
def search(query: str, top_k: int = 10, min_score: float = 0.0) -> dict:
    """Search the codebase using semantic + BM25 hybrid retrieval. PREFER this over grep/rg/Glob for finding code.

    Use this tool when you need to:
    - Find where a function, class, or pattern is implemented
    - Understand how a feature or concept works across the codebase
    - Locate code by describing what it does (not just exact strings)
    - Find examples of a pattern or API usage
    - Explore unfamiliar parts of the codebase

    Advantages over text search: understands synonyms, paraphrases, and intent.
    Returns file paths, line ranges, relevance scores, source, and matching code.

    Args:
        query: Describe what you're looking for in natural language.
               Good: "authentication middleware that validates JWT tokens"
               Good: "function that parses CSV files"
               Good: "error handling for database connections"
               Avoid: single words or exact-match-only queries.
        top_k: Maximum results to return. Default 10. Use 3-5 for focused queries, 15-20 for broad exploration.
        min_score: Drop results below this score. Results with zero relevance are always removed.
    """
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if top_k > 100:
        raise ValueError("top_k must be at most 100")
    if min_score < 0:
        raise ValueError("min_score must be zero or greater")

    indexer = _get_indexer()
    results = indexer.search(query, top_k=top_k, min_score=min_score)
    return {
        "query": query,
        "count": len(results),
        "results": [
            {
                "file": result.chunk.file_path,
                "lines": f"{result.chunk.start_line}-{result.chunk.end_line}",
                "start_line": result.chunk.start_line,
                "end_line": result.chunk.end_line,
                "language": result.chunk.language,
                "score": round(result.score, 4),
                "source": result.source.value,
                "content": result.chunk.content[:500],
            }
            for result in results
        ],
    }


@mcp.tool()
def resolve(query: str, top_k: int = 5) -> dict:
    """Resolve a feature or concept with tests, imports, callers, and related files.

    Use this after search when you need graph context: related files, tests,
    imports, callers/callees, and a compact dependency chain.
    """
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if top_k > 100:
        raise ValueError("top_k must be at most 100")

    indexer = _get_indexer()
    pack = indexer.resolve(query, top_k=top_k)
    return {
        "query": pack["query"],
        "confidence": pack["confidence"],
        "primary_files": sorted({r.chunk.file_path for r in pack["primary_chunks"]}),
        "related_files": pack["related_files"],
        "test_files": pack["test_files"],
        "imports": pack["imports"],
        "imported_by": pack["imported_by"],
        "symbols": pack["symbols"][:20],
        "callers": pack["callers"][:15],
        "callees": pack["callees"][:15],
        "dependency_chain": pack["dependency_chain"][:10],
        "formatted": indexer.format_context(pack),
    }


@mcp.tool()
def explain(location: str) -> dict:
    """Explain a file, file:line location, or symbol with surrounding code context.

    Use this when you already know the file or symbol name and need its
    definition, usages, tests, imports, and caller/callee relationships.
    """
    if not location or not location.strip():
        raise ValueError("location must not be empty")
    indexer = _get_indexer()
    pack = indexer.explain(location)
    return {
        "location": location,
        "confidence": pack["confidence"],
        "primary_files": sorted({r.chunk.file_path for r in pack["primary_chunks"]}),
        "related_files": pack["related_files"],
        "test_files": pack["test_files"],
        "symbols": pack["symbols"][:20],
        "callers": pack["callers"][:15],
        "callees": pack["callees"][:15],
        "dependency_chain": pack["dependency_chain"][:10],
        "formatted": indexer.format_context(pack),
    }


@mcp.tool()
def status() -> dict:
    """Report whether the index is ready, still indexing, empty, or failed.

    This never builds an index. Use it to decide whether to wait, search, or
    call reindex.
    """
    if _indexer is not None and _indexer.is_indexing:
        return {
            "version": __version__,
            "status": "indexing",
            "root": str(_indexer.root),
            "model": _indexer.model_id,
            "is_indexing": True,
            "indexed_files": _indexer.last_index_stats.indexed_files
            if _indexer.last_index_stats
            else 0,
            "total_chunks": _indexer.last_index_stats.total_chunks
            if _indexer.last_index_stats
            else 0,
            "last_index_error": _indexer.last_index_error,
        }
    indexer = _get_indexer(ensure_indexed=False)
    return {"version": __version__, **indexer.status()}


@mcp.tool()
async def reindex(force: bool = False, include_text: bool | None = None, ctx: Context = None) -> dict:
    """Rebuild or incrementally update the index, with realtime progress.

    Use ``force=True`` only when the incremental watcher may have missed changes
    or after switching model configuration. Otherwise the incremental update is
    fast and preserves cached vectors.
    """
    global _index_attempted
    loop = asyncio.get_running_loop()
    progress = _tool_progress(ctx, loop) if ctx is not None else None
    with _init_lock:
        indexer = _get_indexer(ensure_indexed=False)
        stats = await asyncio.to_thread(
            indexer.index,
            force=force,
            include_text_files=include_text,
            progress=progress,
        )
        _index_attempted = True
    return {
        "version": __version__,
        "status": "ready",
        **_stats_payload(indexer, stats),
    }


def run_server(
    root: str | Path | None = None,
    *,
    model_id: str | None = None,
    include_text: bool = False,
    watch: bool = True,
    show_progress: bool = True,
) -> None:
    """Index the repository, start the watcher, then serve MCP on stdio."""
    global _server_root, _server_model_id, _server_include_text, _show_progress

    _server_root = _resolve_server_root(root)
    _server_model_id = model_id
    _server_include_text = include_text
    _show_progress = show_progress

    indexer = None
    try:
        indexer = _get_indexer()
    except Exception as exc:
        print(f"[vortexa] Indexing failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    if watch and indexer is not None and _index_attempted:
        from vortexa.interfaces.watcher import IndexWatcher

        watcher = IndexWatcher(indexer, include_text_files=include_text)
        watcher.start()
        print("[vortexa] Auto-reindex watcher started (polling every 3s)", file=sys.stderr)

    mcp.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    run_server()
