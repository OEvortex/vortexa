"""Command-line interface for vortexa.

Subcommands:
  serve               Start the MCP server (default when no args).
  search QUERY        Hybrid semantic+BM25 search. Alias for -q.
  resolve QUERY       Full context resolution with graph expansion.
  explain LOCATION    Deep-dive into a file path, file:line, or symbol name.

Backward compatibility:
  vortexa -q TEXT    Alias for `vortexa search TEXT`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from vortexa.core.indexer import CodebaseIndexer

logger = logging.getLogger(__name__)


# ── Root / environment resolution (unchanged) ──────────────────────────────


def _json_path_from_text(text: str) -> str | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    for key in (
        "working_directory",
        "cwd",
        "current_directory",
        "root",
        "project_root",
        "workspace_root",
    ):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _resolve_root(environment_details: str | None, explicit_root: str | None) -> Path:
    if explicit_root:
        root = Path(explicit_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"root is not a directory: {root}")
        return root

    if not environment_details:
        return Path.cwd().resolve()

    details = environment_details.strip()
    if not details:
        return Path.cwd().resolve()

    direct_path = Path(details).expanduser()
    if direct_path.is_dir():
        return direct_path.resolve()

    json_path = _json_path_from_text(details)
    if json_path:
        candidate = Path(json_path).expanduser()
        if candidate.is_dir():
            return candidate.resolve()

    for line in details.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = key.strip().lower().replace(" ", "_")
        if normalized_key in {
            "working_directory",
            "cwd",
            "current_directory",
            "root",
            "project_root",
            "workspace_root",
        }:
            candidate = Path(value.strip()).expanduser()
            if candidate.is_dir():
                return candidate.resolve()

    return Path.cwd().resolve()


# ── Formatters ─────────────────────────────────────────────────────────────


def _format_search_result(result) -> dict[str, object]:
    out = {
        "file": result.chunk.file_path,
        "lines": f"{result.chunk.start_line}-{result.chunk.end_line}",
        "score": round(result.score, 4),
        "source": result.source.value,
        "content": result.chunk.content[:500],
    }
    ctx = getattr(result, "context", None)
    if ctx is not None:
        out["graph_context"] = {
            "key_symbol": ctx.key_symbol,
            "incoming": list(ctx.incoming),
            "outgoing": list(ctx.outgoing),
        }
    return out


def _print_search_plain(results) -> None:
    for result in results:
        ctx = getattr(result, "context", None)
        header = (
            f"{result.chunk.file_path}:{result.chunk.start_line}-{result.chunk.end_line} "
            f"score={result.score:.4f} source={result.source.value}"
        )
        if ctx is not None:
            header += f"  symbol={ctx.key_symbol}"
            if ctx.incoming:
                header += f"  ←{ctx.incoming[0]}"
            if ctx.outgoing:
                header += f"  →{ctx.outgoing[0]}"
        print(header)
        print(result.chunk.content[:500].rstrip())
        print()


def _format_resolve_pack(pack, formatted: str | None = None) -> dict[str, object]:
    out: dict[str, object] = {
        "query": pack.query,
        "confidence": round(pack.confidence, 3),
        "primary_files": sorted(set(r.chunk.file_path for r in pack.primary_chunks)),
        "related_files": pack.related_files,
        "test_files": pack.test_files,
        "imports": pack.imports,
        "imported_by": pack.imported_by,
        "symbols": [
            {"name": s.name, "kind": s.kind.value, "file": s.file_path,
             "line": s.start_line}
            for s in pack.symbols[:15]
        ],
        "callers": [
            {"name": c.name, "file": c.file_path, "line": c.start_line}
            for c in pack.callers[:10]
        ],
        "callees": [
            {"name": c.name, "file": c.file_path, "line": c.start_line}
            for c in pack.callees[:10]
        ],
        "dependency_chain": pack.dependency_chain[:10],
        "total_tokens": pack.total_tokens,
    }
    if formatted is not None:
        out["formatted"] = formatted
    return out


def _print_resolve_plain(pack, formatted: str | None = None) -> None:
    if formatted:
        print(formatted)
        print()

    print(f"confidence={pack.confidence:.3f}  "
          f"primary={len(pack.primary_chunks)}  "
          f"related={len(pack.related_files)}  "
          f"tests={len(pack.test_files)}  "
          f"imports={len(pack.imports)}  "
          f"symbols={len(pack.symbols)}")

    primary_files = sorted(set(r.chunk.file_path for r in pack.primary_chunks))
    if primary_files:
        print(f"primary files: {', '.join(primary_files)}")
    if pack.test_files:
        print(f"tests: {', '.join(t.split('/')[-1] for t in pack.test_files[:5])}")
    if pack.imports:
        print(f"imports: {', '.join(i.split('/')[-1] for i in pack.imports[:5])}")
    if pack.imported_by:
        print(f"imported_by: {', '.join(i.split('/')[-1] for i in pack.imported_by[:5])}")
    if pack.symbols:
        names = [s.name for s in pack.symbols[:8]]
        print(f"symbols: {' '.join(names)}{'...' if len(pack.symbols) > 8 else ''}")
    if pack.dependency_chain:
        print(f"dep_chain: {' -> '.join(Path(p).name for p in pack.dependency_chain[:6])}")


def _format_explain_pack(pack, location: str) -> dict[str, object]:
    return {
        "location": location,
        "confidence": round(pack.confidence, 3),
        "primary_files": sorted(set(r.chunk.file_path for r in pack.primary_chunks)),
        "related_files": pack.related_files,
        "test_files": pack.test_files,
        "symbols": [
            {"name": s.name, "kind": s.kind.value, "file": s.file_path,
             "line": s.start_line, "signature": s.signature}
            for s in pack.symbols[:20]
        ],
        "callers": [
            {"name": c.name, "file": c.file_path, "line": c.start_line}
            for c in pack.callers[:15]
        ],
        "callees": [
            {"name": c.name, "file": c.file_path, "line": c.start_line}
            for c in pack.callees[:15]
        ],
        "dependency_chain": pack.dependency_chain[:10],
        "total_tokens": pack.total_tokens,
    }


def _print_explain_plain(pack, location: str) -> None:
    primary_files = sorted(set(r.chunk.file_path for r in pack.primary_chunks))
    print(f"=== explain: {location} ===")
    print(f"confidence={pack.confidence:.3f}")
    if primary_files:
        print(f"primary files: {', '.join(primary_files)}")
    if pack.symbols:
        print()
        print("symbols:")
        for s in pack.symbols[:10]:
            sig = f" — {s.signature}" if s.signature else ""
            print(f"  {s.kind.value} {s.name}  {s.file_path}:{s.start_line}{sig}")
    if pack.test_files:
        print()
        print(f"tests: {', '.join(pack.test_files[:5])}")
    if pack.imports:
        print(f"imports: {', '.join(Path(i).name for i in pack.imports[:5])}")
    if pack.callers:
        print()
        print("callers:")
        for c in pack.callers[:5]:
            print(f"  {c.name}  {c.file_path}:{c.start_line}")
    if pack.callees:
        print()
        print("callees:")
        for c in pack.callees[:5]:
            print(f"  {c.name}  {c.file_path}:{c.start_line}")


# ── Indexer helper (shared by subcommands) ─────────────────────────────────


def _make_indexer(args, parser) -> CodebaseIndexer:
    """Build and optionally index. Returns the indexer."""
    try:
        root = _resolve_root(args.environment_details, args.root)
    except ValueError as exc:
        parser.error(str(exc))
        raise SystemExit(2) from exc  # unreachable, satisfies the type checker

    indexer = CodebaseIndexer(root=root)

    if getattr(args, "no_index", False):
        indexer._load_state()
    else:
        include_text = bool(getattr(args, "include_text", False))
        force = bool(getattr(args, "force", False))
        stats = indexer.index(force=force, include_text_files=include_text)
        print(
            f"[vortexa] Ready: {stats.indexed_files} files, "
            f"{stats.total_chunks} chunks in {stats.index_time_ms:.0f}ms",
            file=sys.stderr,
        )

    return indexer


# ── Subcommand handlers ─────────────────────────────────────────────────────


def cmd_serve(args, parser) -> int:
    from vortexa.interfaces.mcp_server import run_server
    run_server()
    return 0


def cmd_search(args, parser) -> int:
    if not args.query:
        parser.error("query is required for `search`")
    if args.top_k <= 0:
        parser.error("--top-k must be greater than 0")
    if args.alpha is not None and not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between 0.0 and 1.0")

    indexer = _make_indexer(args, parser)
    results = indexer.search(
        args.query,
        top_k=args.top_k,
        alpha=args.alpha,
        use_vortex_score=True,
        hybrid=bool(getattr(args, "hybrid", False)),
    )

    if args.plain:
        _print_search_plain(results)
    else:
        print(json.dumps([_format_search_result(r) for r in results], indent=2))
    return 0


def cmd_resolve(args, parser) -> int:
    if not args.query:
        parser.error("query is required for `resolve`")
    if args.top_k <= 0:
        parser.error("--top-k must be greater than 0")

    indexer = _make_indexer(args, parser)
    pack = indexer.resolve(args.query, top_k=args.top_k)
    formatted = indexer.format_context(pack)

    if args.plain:
        _print_resolve_plain(pack, formatted)
    else:
        out = _format_resolve_pack(pack)
        out["formatted"] = formatted
        print(json.dumps(out, indent=2))
    return 0


def cmd_explain(args, parser) -> int:
    if not args.location:
        parser.error("location is required for `explain`")

    indexer = _make_indexer(args, parser)
    pack = indexer.explain(args.location)
    formatted = indexer.format_context(pack)

    if args.plain:
        _print_explain_plain(pack, args.location)
        if formatted:
            print()
            print("formatted context:")
            print(formatted)
    else:
        out = _format_explain_pack(pack, args.location)
        out["formatted"] = formatted
        print(json.dumps(out, indent=2))
    return 0


# ── Legacy -q parser (backward compat for `vortexa -q "auth"`) ─────────────


def _build_legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vortexa",
        description="Codebase indexing and semantic search engine.",
    )
    parser.add_argument("-q", "--query", metavar="TEXT", help="search query; quote multi-word queries")
    parser.add_argument(
        "environment_details",
        nargs="?",
        help="optional root path, JSON environment details, or Kilo environment details",
    )
    parser.add_argument("--root", help="codebase root to index and search; overrides environment_details")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--include-text", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-index", action="store_true")
    parser.add_argument("--plain", action="store_true")
    return parser


def run_legacy_query(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Original -q behavior — search without Vortex Score rerank.

    Preserved verbatim for backward compatibility with the cf7dadc CLI.
    """
    if not args.query:
        raise SystemExit("query is required when using -q/--query")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be greater than 0")
    if args.alpha is not None and not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be between 0.0 and 1.0")

    try:
        root = _resolve_root(args.environment_details, args.root)
    except ValueError as exc:
        parser.error(str(exc))
    indexer = CodebaseIndexer(root=root)

    if args.no_index:
        indexer._load_state()
    else:
        stats = indexer.index(force=args.force, include_text_files=args.include_text)
        print(
            f"[vortexa] Ready: {stats.indexed_files} files, "
            f"{stats.total_chunks} chunks in {stats.index_time_ms:.0f}ms",
            file=sys.stderr,
        )

    query = args.query
    results = indexer.search(query, top_k=args.top_k, alpha=args.alpha)
    if args.plain:
        _print_search_plain(results)
    else:
        print(json.dumps([_format_search_result(r) for r in results], indent=2))
    return 0


# ── Subcommand parser (new shape) ──────────────────────────────────────────


def _add_common_index_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "environment_details",
        nargs="?",
        help="optional root path, JSON environment details, or Kilo environment details",
    )
    p.add_argument(
        "--root",
        help="codebase root to index and search; overrides environment_details",
    )
    p.add_argument(
        "--include-text",
        action="store_true",
        help="include text files such as .md, .json, and .yaml in the index",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="force a full re-index before searching",
    )
    p.add_argument(
        "--no-index",
        action="store_true",
        help="skip indexing and search the existing index only",
    )
    p.add_argument(
        "--plain",
        action="store_true",
        help="print human-readable results instead of JSON",
    )


def _build_subcommand_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vortexa",
        description="Codebase indexing and semantic search engine.",
    )
    subparsers = parser.add_subparsers(
        title="subcommands",
        dest="subcommand",
        metavar="<command>",
    )

    # serve ────────────────────────────────────────────────────────────────
    subparsers.add_parser(
        "serve",
        help="start the MCP server on stdio (default when no args)",
    )

    # search ──────────────────────────────────────────────────────────────
    p_search = subparsers.add_parser(
        "search",
        help="hybrid semantic+BM25 search with Vortex Score reranking",
        description="Search the codebase using semantic + BM25 hybrid retrieval. "
                    "Returns ranked file:line matches with optional graph context.",
    )
    p_search.add_argument("query", help="search query; quote multi-word queries")
    p_search.add_argument("--top-k", type=int, default=10,
                          help="maximum results (default: 10)")
    p_search.add_argument("--alpha", type=float, default=None,
                          help="semantic weight 0.0–1.0 (default: adaptive)")
    p_search.add_argument("--hybrid", action="store_true",
                          help="enrich each result with query-aware graph context")
    _add_common_index_args(p_search)

    # resolve ─────────────────────────────────────────────────────────────
    p_resolve = subparsers.add_parser(
        "resolve",
        help="full context resolution with graph expansion",
        description="Search + knowledge graph expansion + compression. "
                    "Returns primary results, related files, tests, imports, "
                    "symbols, callers/callees, and dependency chain.",
    )
    p_resolve.add_argument("query", help="search query or task description")
    p_resolve.add_argument("--top-k", type=int, default=5,
                           help="maximum primary results (default: 5)")
    _add_common_index_args(p_resolve)

    # explain ─────────────────────────────────────────────────────────────
    p_explain = subparsers.add_parser(
        "explain",
        help="deep-dive into a file path, file:line, or symbol name",
        description="Find the symbol/file and return definition, usages, tests, "
                    "imports, and caller/callee relationships.",
    )
    p_explain.add_argument(
        "location",
        help="file path with line number (e.g. 'src/module.py:42'), symbol name "
             "(e.g. 'DatabaseClient'), or file path (e.g. 'src/module.py')",
    )
    _add_common_index_args(p_explain)

    return parser


_SUBCOMMAND_HANDLERS = {
    "search": cmd_search,
    "resolve": cmd_resolve,
    "explain": cmd_explain,
    "serve": cmd_serve,
}


# ── Entry point ────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv if argv is not None else sys.argv[1:])

    # Backward-compat: -q / --query shortcut (pre-subcommand CLI shape).
    # These flags are only meaningful in legacy mode, so we route here
    # before the subcommand parser sees them.
    if raw_argv and ("-q" in raw_argv or "--query" in raw_argv):
        parser = _build_legacy_parser()
        args = parser.parse_args(raw_argv)
        return run_legacy_query(args, parser)

    # No args → start MCP server (preserves the default behaviour).
    if not raw_argv:
        from vortexa.interfaces.mcp_server import run_server
        run_server()
        return 0

    # Subcommand dispatch
    parser = _build_subcommand_parser()

    # Special-case `serve` and `serve --help` to keep the existing UX.
    if raw_argv[0] == "serve":
        if raw_argv in (["serve", "-h"], ["serve", "--help"]):
            print("usage: vortexa serve\n\nstart the MCP server")
            return 0
        from vortexa.interfaces.mcp_server import run_server
        run_server()
        return 0

    args = parser.parse_args(raw_argv)
    handler = _SUBCOMMAND_HANDLERS.get(args.subcommand)
    if handler is None:
        parser.print_help()
        return 0
    return handler(args, parser)


if __name__ == "__main__":
    raise SystemExit(main())