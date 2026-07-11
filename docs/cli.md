# CLI Reference

The `vortexa` command is the command-line front end to the context engine. It
exposes four entry points:

- `serve` — start the MCP server (default when no arguments are given),
- `search` — hybrid search with Vortex Score reranking,
- `resolve` — full context resolution with graph expansion,
- `explain` — deep-dive into a file path, `file:line`, or symbol name,
- `-q` / `--query` — legacy search shortcut (backward compatible).

All subcommands share a set of root-resolution rules for picking the codebase
root, and a set of common indexing flags.

## Root resolution

The codebase root is chosen in this order:

1. `--root PATH` if provided (must be a directory).
2. `environment_details` positional argument, which may be:
   - a direct directory path,
   - a JSON object with a `working_directory` / `cwd` / `root` / `project_root` /
     `workspace_root` key,
   - Kilo-style `Key: value` text where a known key names a directory.
3. The current working directory.

```bash
vortexa search "auth middleware" --root /path/to/project
vortexa resolve "auth middleware" "Working directory: /path/to/project"
```

## Output format

By default every command prints **JSON** to stdout (logs go to stderr). Pass
`--plain` for human-readable text.

## `search`

Hybrid semantic + BM25 search, re-ranked with the Vortex Score.

```bash
vortexa search "authentication middleware that validates JWT tokens" \
    --top-k 5 --alpha 0.6 --hybrid --plain
```

| Flag | Default | Description |
|------|---------|-------------|
| `query` (positional) | — | Search query; quote multi-word queries. |
| `--top-k` | `10` | Maximum results to return. |
| `--alpha` | `None` | Semantic weight `0.0`–`1.0`; `None` = adaptive. |
| `--hybrid` | off | Enrich each result with query-aware graph context. |
| `--root` | — | Codebase root (overrides positional root). |
| `--include-text` | off | Include `.md`/`.json`/`.yaml`/etc. in the index. |
| `--force` | off | Force a full re-index before searching. |
| `--no-index` | off | Search the existing index only (no indexing). |
| `--plain` | off | Print human-readable results instead of JSON. |

JSON output shape:

```json
[
  {
    "file": "src/auth/middleware.py",
    "lines": "12-48",
    "score": 0.892,
    "source": "hybrid",
    "content": "def validate_jwt(token: str) -> User: ...",
    "graph_context": {
      "key_symbol": "validate_jwt",
      "incoming": ["AuthRouter (calls)"],
      "outgoing": ["decode_token (calls)"]
    }
  }
]
```

## `resolve`

Full context resolution: search + knowledge-graph expansion + compression.

```bash
vortexa resolve "CSV parser implementation" --top-k 5 --plain
```

| Flag | Default | Description |
|------|---------|-------------|
| `query` (positional) | — | Search query or task description. |
| `--top-k` | `5` | Maximum primary results (expanded automatically). |
| `--root` / `--include-text` / `--force` / `--no-index` / `--plain` | — | Same as `search`. |

JSON output includes `primary_files`, `related_files`, `test_files`, `imports`,
`imported_by`, `symbols`, `callers`, `callees`, `dependency_chain`,
`confidence`, `total_tokens`, and a `formatted` block ready for an LLM.

## `explain`

Deep-dive into a location. Accepts a `file:line`, a symbol name, or a file path.

```bash
vortexa explain "src/auth/jwt.py:42"
vortexa explain "JWTValidator"
vortexa explain "src/auth/jwt.py"
```

| Flag | Default | Description |
|------|---------|-------------|
| `location` (positional) | — | `file:line`, symbol name, `class:Name`, or file path. |
| `--root` / `--include-text` / `--force` / `--no-index` / `--plain` | — | Same as `search`. |

JSON output includes `primary_files`, `related_files`, `test_files`, `symbols`,
`callers`, `callees`, `dependency_chain`, `confidence`, `total_tokens`, and
`formatted`.

## `serve`

Start the MCP server on stdio. Equivalent to `python -m vortexa.interfaces.mcp_server`.

```bash
vortexa serve
# also: vortexa-serve
```

On startup it auto-indexes the cwd and starts the auto-reindex watcher.

## Legacy `-q`

For backward compatibility with the original CLI shape:

```bash
vortexa -q "authentication middleware that validates JWT tokens"
vortexa -q "CSV parser implementation" /path/to/project
```

| Flag | Default | Description |
|------|---------|-------------|
| `-q`, `--query` | — | Search query; quote multi-word queries. |
| `environment_details` (positional) | — | Root path, JSON, or Kilo-style env details. |
| `--root` | — | Codebase root (overrides `environment_details`). |
| `--top-k` | `10` | Maximum results. |
| `--alpha` | `None` | Semantic weight `0.0`–`1.0`. |
| `--include-text` | off | Include text files in the index. |
| `--force` | off | Force a full re-index. |
| `--no-index` | off | Search existing index only. |
| `--plain` | off | Print human-readable results. |

> The legacy `-q` path does not apply the full Vortex Score rerank (it uses the
> base hybrid search). Prefer the `search` subcommand for reranked results.

## Next steps

- Plug vortexa into an agent via the [MCP Server](mcp-server.md).
- Browse the full [Python API Reference](python-api.md).
