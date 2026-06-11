"""Command-line interface for vortexa."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from vortexa.core.indexer import CodebaseIndexer, _index_dir_for_root

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
    return {
        "file": result.chunk.file_path,
        "lines": f"{result.chunk.start_line}-{result.chunk.end_line}",
        "score": round(result.score, 4),
        "source": result.source.value,
        "content": result.chunk.content[:500],
    }


def _print_plain(results) -> None:
    for result in results:
        print(
            f"{result.chunk.file_path}:{result.chunk.start_line}-{result.chunk.end_line} "
            f"score={result.score:.4f} source={result.source.value}"
        )
        print(result.chunk.content[:500].rstrip())
        print()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vortexa",
        description="Codebase indexing and semantic search engine.",
    )
    parser.add_argument(
        "-q",
        "--query",
        metavar="TEXT",
        help="search query; quote multi-word queries",
    )
    parser.add_argument(
        "environment_details",
        nargs="?",
        help="optional root path, JSON environment details, or text containing a root path to index and search (defaults to current directory)",
    )
    parser.add_argument(
        "--root",
        help="codebase root to index and search; overrides environment_details",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="maximum number of results to return (default: 10)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="semantic weight from 0.0 to 1.0; defaults to adaptive weighting",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="include text files such as .md, .json, and .yaml in the index",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="force a full re-index before searching (ignores existing index)",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="skip indexing entirely and search the existing index (errors if none)",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="print human-readable results instead of JSON",
    )
    return parser


def _has_existing_index(root: Path) -> bool:
    """Check if a persistent index already exists for this root."""
    index_dir = _index_dir_for_root(root)
    return (index_dir / "state.lmdb").exists()


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
    indexer = CodebaseIndexer(root=root)

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


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv if argv is not None else sys.argv[1:])
    if not raw_argv or raw_argv[0] == "serve":
        if raw_argv in (["serve", "-h"], ["serve", "--help"]):
            print("usage: vortexa serve\n\nstart the MCP server")
            return 0
        from vortexa.interfaces.mcp_server import run_server

        run_server()
        return 0

    parser = _build_parser()
    args = parser.parse_args(raw_argv)
    return run_query(args, parser)


if __name__ == "__main__":
    raise SystemExit(main())
