"""Command-line interface for vortexa."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from vortexa.core.indexer import CodebaseIndexer, _index_dir_for_root
from vortexa.core.types import GraphContext

logger = logging.getLogger(__name__)


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


def _format_result(result) -> dict[str, object]:
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


def _print_plain(results) -> None:
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
        "query": pack["query"],
        "confidence": pack["confidence"],
        "primary_files": sorted({r.chunk.file_path for r in pack["primary_chunks"]}),
        "related_files": pack["related_files"],
        "test_files": pack["test_files"],
        "imports": pack["imports"],
        "imported_by": pack["imported_by"],
        "symbols": [
            {"name": s["name"], "kind": s["kind"], "file": s["file"], "line": s["line"]}
            for s in pack["symbols"][:15]
        ],
        "callers": [
            {"name": c["name"], "file": c["file"], "line": c["line"]}
            for c in pack["callers"][:10]
        ],
        "callees": [
            {"name": c["name"], "file": c["file"], "line": c["line"]}
            for c in pack["callees"][:10]
        ],
        "dependency_chain": pack["dependency_chain"][:10],
        "total_tokens": pack["total_tokens"],
    }
    if formatted is not None:
        out["formatted"] = formatted
    return out


def _print_resolve_plain(pack, formatted: str | None = None) -> None:
    if formatted:
        print(formatted)
        print()

    print(
        f"confidence={pack['confidence']:.3f}  "
        f"primary={len(pack['primary_chunks'])}  "
        f"related={len(pack['related_files'])}  "
        f"tests={len(pack['test_files'])}  "
        f"imports={len(pack['imports'])}  "
        f"symbols={len(pack['symbols'])}"
    )

    primary_files = sorted({r.chunk.file_path for r in pack["primary_chunks"]})
    if primary_files:
        print(f"primary files: {', '.join(primary_files)}")
    if pack["test_files"]:
        print(f"tests: {', '.join(t.split('/')[-1] for t in pack['test_files'][:5])}")
    if pack["imports"]:
        print(f"imports: {', '.join(i.split('/')[-1] for i in pack['imports'][:5])}")
    if pack["imported_by"]:
        print(f"imported_by: {', '.join(i.split('/')[-1] for i in pack['imported_by'][:5])}")
    if pack["symbols"]:
        names = [s["name"] for s in pack["symbols"][:8]]
        print(f"symbols: {' '.join(names)}{'...' if len(pack['symbols']) > 8 else ''}")
    if pack["dependency_chain"]:
        print(f"dep_chain: {' -> '.join(Path(p).name for p in pack['dependency_chain'][:6])}")


def _format_explain_pack(pack, location: str) -> dict[str, object]:
    return {
        "location": location,
        "confidence": pack["confidence"],
        "primary_files": sorted({r.chunk.file_path for r in pack["primary_chunks"]}),
        "related_files": pack["related_files"],
        "test_files": pack["test_files"],
        "symbols": [
            {"name": s["name"], "kind": s["kind"], "file": s["file"], "line": s["line"]}
            for s in pack["symbols"][:20]
        ],
        "callers": [
            {"name": c["name"], "file": c["file"], "line": c["line"]}
            for c in pack["callers"][:15]
        ],
        "callees": [
            {"name": c["name"], "file": c["file"], "line": c["line"]}
            for c in pack["callees"][:15]
        ],
        "dependency_chain": pack["dependency_chain"][:10],
        "total_tokens": pack["total_tokens"],
    }


def _print_explain_plain(pack, location: str) -> None:
    primary_files = sorted({r.chunk.file_path for r in pack["primary_chunks"]})
    print(f"=== explain: {location} ===")
    print(f"confidence={pack['confidence']:.3f}")
    if primary_files:
        print(f"primary files: {', '.join(primary_files)}")
    if pack["symbols"]:
        print()
        print("symbols:")
        for s in pack["symbols"][:10]:
            sig = f" — {s.get('signature', '')}" if s.get("signature") else ""
            print(f"  {s['kind']} {s['name']}  {s['file']}:{s['line']}{sig}")
    if pack["test_files"]:
        print()
        print(f"tests: {', '.join(pack['test_files'][:5])}")
    if pack["imports"]:
        print(f"imports: {', '.join(Path(i).name for i in pack['imports'][:5])}")
    if pack["callers"]:
        print()
        print("callers:")
        for c in pack["callers"][:5]:
            print(f"  {c['name']}  {c['file']}:{c['line']}")
    if pack["callees"]:
        print()
        print("callees:")
        for c in pack["callees"][:5]:
            print(f"  {c['name']}  {c['file']}:{c['line']}")


def _attach_graph_context(self, results):
    graph = self._build_repo_graph()
    enriched = []
    for r in results:
        ctx = None
        if graph is not None:
            nid = f"file:{r.chunk.file_path}"
            if nid in graph.nodes:
                incoming = []
                outgoing = []
                for nb in graph.neighbors(nid, direction="in")[:1]:
                    incoming.append(nb.split("::")[-1] if "::" in nb else nb.split(":")[-1] if ":" in nb else nb)
                for nb in graph.neighbors(nid, direction="out")[:1]:
                    outgoing.append(nb.split("::")[-1] if "::" in nb else nb.split(":")[-1] if ":" in nb else nb)
                ctx = GraphContext(key_symbol="", incoming=tuple(incoming), outgoing=tuple(outgoing))
        if ctx is not None:
            class ResultWithContext:
                def __init__(self, result, context):
                    self._result = result
                    self.context = context
                    self.chunk = result.chunk
                    self.score = result.score
                    self.source = result.source
                def __getattr__(self, name):
                    return getattr(self._result, name)
            enriched.append(ResultWithContext(r, ctx))
        else:
            enriched.append(r)
    return enriched


def run_query(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
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
    _MODEL_ALIASES = {
        "mini": "VTXAI/vtx-embed-7M",
        "nano": "VTXAI/vtx-embed-1M",
    }

    def _resolve_model_id(model_arg: str | None) -> str | None:
        if model_arg is None:
            return None
        if model_arg in _MODEL_ALIASES:
            return _MODEL_ALIASES[model_arg]
        return model_arg

    model_id = _resolve_model_id(args.model)
    indexer = CodebaseIndexer(root=root, model_id=model_id)

    has_index = _has_existing_index(root)

    if args.no_index:
        if not has_index:
            raise SystemExit(
                f"No index found for {root}. Run without --no-index to build one."
            )
        indexer._load_state()
        print(f"[vortexa] Loaded existing index ({len(indexer.chunks)} chunks)", file=sys.stderr)
    elif args.force:
        stats = indexer.index(force=True, include_text_files=args.include_text)
        print(
            f"[vortexa] Re-indexed: {stats.indexed_files} files, "
            f"{stats.total_chunks} chunks in {stats.index_time_ms:.0f}ms",
            file=sys.stderr,
        )
    else:
        if not has_index:
            stats = indexer.index(include_text_files=args.include_text)
            print(
                f"[vortexa] Indexed: {stats.indexed_files} files, "
                f"{stats.total_chunks} chunks in {stats.index_time_ms:.0f}ms",
                file=sys.stderr,
            )
        else:
            indexer._load_state()
            before = len(indexer.chunks)
            stats = indexer.index(include_text_files=args.include_text)
            after = len(indexer.chunks)
            if after == before and stats.memo_misses == 0:
                print(
                    f"[vortexa] Index up to date ({after} chunks, no changes)",
                    file=sys.stderr,
                )
            else:
                added = after - before
                label = f"+{added} chunks" if added else f"{stats.memo_misses} updated"
                print(
                    f"[vortexa] Updated: {stats.indexed_files} files, "
                    f"{after} chunks ({label}) in {stats.index_time_ms:.0f}ms",
                    file=sys.stderr,
                )

    query = args.query
    results = indexer.search(query, top_k=args.top_k, alpha=args.alpha)
    if args.plain:
        _print_plain(results)
    else:
        print(json.dumps([_format_result(result) for result in results], indent=2))
    return 0


def _has_existing_index(root: Path) -> bool:
    """Check if a persistent index already exists for this root."""
    index_dir = _index_dir_for_root(root)
    return (index_dir / "state.lmdb").exists()


def _make_indexer(args, parser):
    try:
        root = _resolve_root(args.environment_details, args.root)
    except ValueError as exc:
        parser.error(str(exc))
        raise SystemExit(2) from exc  # unreachable, satisfies the type checker

    _MODEL_ALIASES = {
        "mini": "VTXAI/vtx-embed-7M",
        "nano": "VTXAI/vtx-embed-1M",
    }

    model_arg = getattr(args, "model", None)
    if model_arg in _MODEL_ALIASES:
        model_arg = _MODEL_ALIASES[model_arg]

    indexer = CodebaseIndexer(root=root, model_id=model_arg)

    if getattr(args, "no_index", False):
        if not _has_existing_index(root):
            raise SystemExit(
                f"No index found for {root}. Run without --no-index to build one."
            )
        indexer._load_state()
        print(f"[vortexa] Loaded existing index ({len(indexer.chunks)} chunks)", file=sys.stderr)
    else:
        include_text = bool(getattr(args, "include_text", False))
        force = bool(getattr(args, "force", False))
        if force:
            stats = indexer.index(force=True, include_text_files=include_text)
            print(
                f"[vortexa] Re-indexed: {stats.indexed_files} files, "
                f"{stats.total_chunks} chunks in {stats.index_time_ms:.0f}ms",
                file=sys.stderr,
            )
        elif _has_existing_index(root):
            indexer._load_state()
            before = len(indexer.chunks)
            stats = indexer.index(include_text_files=include_text)
            after = len(indexer.chunks)
            if after == before and stats.memo_misses == 0:
                print(
                    f"[vortexa] Index up to date ({after} chunks, no changes)",
                    file=sys.stderr,
                )
            else:
                added = after - before
                label = f"+{added} chunks" if added else f"{stats.memo_misses} updated"
                print(
                    f"[vortexa] Updated: {stats.indexed_files} files, "
                    f"{after} chunks ({label}) in {stats.index_time_ms:.0f}ms",
                    file=sys.stderr,
                )
        else:
            stats = indexer.index(include_text_files=include_text)
            print(
                f"[vortexa] Indexed: {stats.indexed_files} files, "
                f"{stats.total_chunks} chunks in {stats.index_time_ms:.0f}ms",
                file=sys.stderr,
            )

    return indexer


def _add_common_index_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "environment_details",
        nargs="?",
        help="optional root path, JSON environment details, or text containing a root path to index and search (defaults to current directory)",
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
    p.add_argument(
        "--model",
        metavar="ID",
        default=None,
        help="embedding model ID (default: VTXAI/vtx-embed-7M, alias: mini, nano)",
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

    subparsers.add_parser(
        "serve",
        help="start the MCP server on stdio (default when no args)",
    )

    p_search = subparsers.add_parser(
        "search",
        help="hybrid semantic+BM25 search with optional graph context",
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
    results = indexer.search(args.query, top_k=args.top_k, alpha=args.alpha)
    if getattr(args, "hybrid", False):
        results = _attach_graph_context(indexer, results)

    if args.plain:
        _print_plain(results)
    else:
        print(json.dumps([_format_result(r) for r in results], indent=2))
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


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv if argv is not None else sys.argv[1:])

    if raw_argv and ("-q" in raw_argv or "--query" in raw_argv):
        parser = argparse.ArgumentParser(
            prog="vortexa",
            description="Codebase indexing and semantic search engine.",
        )
        parser.add_argument("-q", "--query", metavar="TEXT", help="search query; quote multi-word queries")
        parser.add_argument(
            "environment_details",
            nargs="?",
            help="optional root path, JSON environment details, or text containing a root path to index and search (defaults to current directory)",
        )
        parser.add_argument("--root", help="codebase root to index and search; overrides environment_details")
        parser.add_argument("--top-k", type=int, default=10)
        parser.add_argument("--alpha", type=float, default=None)
        parser.add_argument("--include-text", action="store_true")
        parser.add_argument("--force", action="store_true")
        parser.add_argument("--no-index", action="store_true")
        parser.add_argument("--plain", action="store_true")
        parser.add_argument("--model", metavar="ID", default=None)
        args = parser.parse_args(raw_argv)
        return run_query(args, parser)

    if not raw_argv:
        from vortexa.interfaces.mcp_server import run_server
        run_server()
        return 0

    parser = _build_subcommand_parser()

    if raw_argv[0] == "serve":
        if raw_argv in (["serve", "-h"], ["serve", "--help"]):
            print("usage: vortexa serve\n\nstart the MCP server")
            return 0
        from vortexa.interfaces.mcp_server import run_server
        run_server()
        return 0

    if raw_argv[0] == "embed":
        return _run_embed(raw_argv[1:])

    args = parser.parse_args(raw_argv)
    handler = {
        "search": cmd_search,
        "resolve": cmd_resolve,
        "explain": cmd_explain,
        "serve": cmd_serve,
    }.get(args.subcommand)
    if handler is None:
        parser.print_help()
        return 0
    return handler(args, parser)


def _build_embed_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vortexa embed",
        description="Encode text strings into dense vector embeddings using VTX-embed models.",
    )
    parser.add_argument(
        "texts",
        nargs="*",
        help="text strings to encode (quote multi-word strings)",
    )
    parser.add_argument(
        "--model",
        metavar="ID",
        default="mini",
        help="embedding model ID or alias (mini, nano). Default: mini",
    )
    return parser


def _run_embed(argv: list[str]) -> int:
    parser = _build_embed_parser()
    args = parser.parse_args(argv)

    if not args.texts:
        raise SystemExit("at least one text string is required for embedding")

    from vortexa.core.inference import VortexEmbedInference

    engine = VortexEmbedInference(args.model)
    vecs = engine.encode(args.texts)

    print(json.dumps({
        "model": engine.model_id,
        "dim": engine.dim,
        "count": len(args.texts),
        "embeddings": [v.tolist() for v in vecs],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
