# MCP Server

vortexa ships a built-in **Model Context Protocol (MCP)** server that exposes the
entire V2 context engine as agent-friendly tools. It uses stdio transport and is
compatible with Claude Code, Cursor, and any MCP-capable client.

## Starting the server

```bash
# Auto-indexes the current directory, then serves on stdio
python -m vortexa.interfaces.mcp_server

# Or via the installed entry point
vortexa serve
```

On startup the server:

1. Indexes the current working directory.
2. Prints stats to stderr (`[vortexa] Ready: ...`).
3. Starts an auto-reindex `IndexWatcher` (native FS events, polling fallback).
4. Listens for MCP requests on stdio.

The server's `instructions` tell the agent to **always prefer vortexa over
`grep`/`rg`/manual file reading** when searching code.

## Tools

The server exposes **11 tools** across three groups.

### Core search & context (3)

#### `search`

```text
search(query: str, top_k: int = 10, hybrid: bool = False) -> str
```

Hybrid semantic + BM25 search with Vortex Score reranking. Returns a JSON array
of results (`file`, `lines`, `score`, `source`, `content`, and `graph_context`
when `hybrid=true`).

- `query` — what you're looking for, in natural language or as a symbol name.
- `top_k` — maximum results. Default `10`.
- `hybrid` — enrich each result with compact, query-aware structural graph
  context per file (`key_symbol` + one incoming + one outgoing structural edge).

#### `resolve`

```text
resolve(query: str, top_k: int = 5) -> str
```

Full context resolution: search + knowledge-graph expansion + compression.
Returns a structured `ContextPack` (primary files, related files, test files,
imports, imported-by, symbols, callers, callees, dependency chain, confidence)
plus a `formatted` block. Prefer this over `search` when you need to understand
how something fits into the broader codebase.

#### `explain`

```text
explain(location: str) -> str
```

Deep-dive into a code location. `location` is one of:

- file path with line: `"src/module.py:42"`
- symbol name: `"DatabaseClient"`, `"parse_query"`
- class name: `"class:UserService"`
- file path: `"src/module.py"`

Returns definition, usages, tests, imports, and caller/callee context.

### Knowledge graph (5)

#### `query_graph`

```text
query_graph(question: str, mode: str = "bfs", depth: int = 3,
            context_filter: list[str] | None = None) -> str
```

BFS or DFS traversal from query-relevant seeds, rendered as text. Auto-picks
seeds by scoring every node against the question. `mode` is `"bfs"` (broad
context) or `"dfs"` (trace a path). `depth` is `1..6`. `context_filter` is an
optional relation whitelist; when omitted, a default structural-relations filter
suppresses type-annotation noise (`str`/`bool`/`Path` edges).

#### `get_god_nodes`

```text
get_god_nodes(top_n: int = 10) -> str
```

The top-N most-connected *real* entities (file-level hub nodes excluded) — useful
for understanding a codebase's architectural backbone.

#### `get_graph_node`

```text
get_graph_node(label: str) -> str
```

Details for one node by label or id: `id`, `label`, `kind`, `file_path`, `degree`.

#### `get_graph_neighbors`

```text
get_graph_neighbors(label: str) -> str
```

Incoming and outgoing edges of a node: `{source, target, relation, direction}`.

#### `get_shortest_path`

```text
get_shortest_path(source: str, target: str, max_hops: int = 8) -> str
```

BFS shortest path between two nodes (matched by label) over the undirected graph.
Returns hop count and `(source_label, relation, target_label)` segments.

### Lifecycle (3)

#### `stats`

```text
stats() -> str
```

Index + graph + session statistics for the current project.

#### `watch`

```text
watch(action: str) -> str
```

Start (`action="start"`) or stop (`action="stop"`) the auto-reindex watcher.

#### `clear_index`

```text
clear_index() -> str
```

Drop the persistent index for the current project root (chunks, embeddings, BM25
state, and knowledge graph). Stops the watcher first.

## Integration with Claude Code / Cursor

Add vortexa to your MCP configuration. For Cursor, edit `~/.cursor/mcp.json`;
for Claude Code, add it to the `mcp_servers` block.

```json
{
  "mcpServers": {
    "vortexa": {
      "command": "python",
      "args": ["-m", "vortexa.interfaces.mcp_server"],
      "cwd": "/path/to/your/project"
    }
  }
}
```

Use an absolute `command` (e.g. the path to your `python` or `vortexa` binary) if
`python` is not on the agent's `PATH`. Set `cwd` to the project you want indexed.

### Example agent workflow

1. Agent receives: *"How do we validate JWT tokens?"*
2. Agent calls `resolve("where is JWT validation implemented?", top_k=5)` → gets
   primary files, tests, importers, callers, callees, and a compressed context.
3. Agent calls `get_shortest_path("class:JWTValidator", "file:src/api/users.py")`
   → discovers the call chain.
4. Agent calls `explain("src/auth/jwt.py:42")` → reads the exact definition.

All of this is one or two round-trips instead of dozens of `grep` + file reads.

## Troubleshooting

- **Model download fails / hub is unreachable.** The default embedder is fetched
  from the Hugging Face Hub on first run. Pre-download it, or pass a custom
  `Embedder` via the Python API.
- **Watcher noise / high CPU.** Set `VORTEXA_FORCE_POLLING=1` to force polling,
  or use the `watch` tool to stop the watcher.
- **No results.** Ensure the index was built (`stats` shows `indexed_files > 0`).
  Use `search` with `alpha=0.0` to fall back to pure keyword matching.

## Next steps

- Understand the [Knowledge Graph & Scoring](knowledge-graph.md) behind these tools.
- Use the same engine from the [CLI](cli.md) or [Python API](python-api.md).
