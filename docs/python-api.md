# Python API Reference

This page documents the public Python surface of vortexa, centered on
`CodebaseIndexer`. All examples assume:

```python
from vortexa.core.indexer import CodebaseIndexer

indexer = CodebaseIndexer(root=".")
indexer.index()
```

## `CodebaseIndexer`

The main orchestrator. Construct it, call `index()`, then search / resolve /
traverse.

### Constructor

```python
CodebaseIndexer(
    root: str | Path,
    model: Encoder | Embedder | None = None,
    model_id: str = "VTXAI/Vortex-Embed-4.7M",
    index_dir: str | Path | None = None,
    chunk_config: ChunkConfig | None = None,
)
```

| Argument | Description |
|----------|-------------|
| `root` | Project root to index and search. |
| `model` | Optional `Embedder` or `Encoder` override. When an `Embedder` is passed, it is used directly; otherwise the object is treated as a plain encoder. |
| `model_id` | Hugging Face model id for the default `LF4Embedder` when `model` is `None`. |
| `index_dir` | Override the persistent index directory (defaults to `<root>/.jarvis/index`). |
| `chunk_config` | `ChunkConfig` controlling chunk size / overlap. |

### `index(...)`

```python
indexer.index(
    force: bool = False,
    include_text_files: bool = False,
    chunk_config: ChunkConfig | None = None,
) -> IndexStats
```

Walks the project, parses, chunks, embeds, builds the knowledge graph, and
persists state. Incremental by default — unchanged files (by content hash) are
skipped. Returns an `IndexStats`.

| Argument | Description |
|----------|-------------|
| `force` | Ignore memoization and re-embed everything. |
| `include_text_files` | Include `.md`, `.json`, `.yaml`, `.toml`, etc. in the index. |
| `chunk_config` | Override chunking for this run. |

### Search

#### `search(...)`

```python
indexer.search(
    query: str,
    top_k: int = 10,
    alpha: float | None = None,
    use_vortex_score: bool = False,
    hybrid: bool = False,
) -> list[SearchResult]
```

| Argument | Description |
|----------|-------------|
| `query` | Natural-language query or symbol name. |
| `top_k` | Number of results to return. |
| `alpha` | Semantic weight in `[0, 1]`. `None` → adaptive. `0.0` → pure BM25, `1.0` → pure semantic. |
| `use_vortex_score` | Re-rank with the full weighted Vortex Score. |
| `hybrid` | Enrich each result with a `GraphContext` (key symbol + 1 incoming + 1 outgoing structural edge). |

Returns `list[SearchResult]`, or `list[SearchResultWithContext]` when `hybrid=True`
(both expose `chunk`, `score`, `source`, and `context`).

#### `search_by_path(query, top_k=10)`

Pure path-based retrieval — no embeddings, no BM25. Useful for "find the file
that handles payments" style queries.

#### `find_symbol(name, top_k=10)`

Find code that defines a symbol by name. Tries the knowledge graph first, then
falls back to BM25-biased search.

#### `find_related(chunk_idx, top_k=5)`

Find chunks semantically similar to an already-indexed chunk (by index).

### Knowledge graph & context

#### `resolve(query, top_k=5) -> ContextPack`

The primary V2 API. Runs hybrid search with Vortex Score reranking, expands the
results into a `ContextPack`, and compresses it to an ~8000-token budget.

#### `explain(code_location) -> ContextPack`

Deep-dive into a location. Accepts `file:line` (`src/auth/jwt.py:42`), a symbol
name (`JWTValidator`), or a file path (`src/auth/jwt.py`).

#### `query_graph(question, mode="bfs", depth=3, context_filter=None, token_budget=2000) -> str`

Render a BFS/DFS traversal of the knowledge graph as text. `mode` is `"bfs"`
(broad context) or `"dfs"` (trace a path). `depth` is clamped to `1..6`.

#### `get_god_nodes(top_n=10) -> list[dict]`

The top-N most-connected *real* entities (file-level hub nodes are excluded).

#### `get_graph_node(label) -> dict | None`

Details for one node by label or id: `id`, `label`, `kind`, `file_path`, `degree`.

#### `get_graph_neighbors(label) -> list[dict]`

Incoming and outgoing edges for a node: `{source, target, relation, direction}`.

#### `get_shortest_path(source, target, max_hops=8) -> GraphPath | None`

Shortest path between two nodes (matched by label), as a `GraphPath`.

### Watch mode

The watcher lives in `interfaces.watcher`; the MCP server manages one
automatically. From Python:

```python
from vortexa.interfaces.watcher import IndexWatcher

watcher = IndexWatcher(indexer, poll_interval=3.0)
watcher.start()
# ... files change, auto-reindex happens ...
watcher.stop()
```

With `force_polling=True`, or `VORTEXA_FORCE_POLLING=1`, native FS events are
bypassed in favor of `(mtime_ns, size)` polling.

### Management

#### `stats() -> dict`

```python
{
    "indexed_files": 127,
    "total_chunks": 843,
    "languages": {"python": 45, ...},
    "graph_nodes": 2104,
    "graph_edges": 3820,
    "memo_hits": 120,
    "memo_misses": 723,
    "session_queries": 3,
}
```

#### `clear()`

Deletes the persistent index, graph, and session memory, and resets in-memory
state.

#### `format_context(pack) -> str`

Render a `ContextPack` as agent-ready text.

## Key types

All live in `vortexa.core.types`.

### `Chunk`

```python
@dataclass(frozen=True, slots=True)
class Chunk:
    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str | None = None
    lineage: Lineage | None = None
    chunk_hash: str | None = None
```

`chunk.location` returns `"file.py:start-end"`.

### `SearchResult` / `SearchResultWithContext`

```python
@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk: Chunk
    score: float
    source: SearchMode   # SEMANTIC | BM25 | HYBRID | PATH | SYMBOL
```

When `hybrid=True`, results are wrapped in `SearchResultWithContext` which also
carries a `context: GraphContext` (and delegates `chunk`/`score`/`source`).

### `ContextPack`

```python
@dataclass(frozen=True, slots=True)
class ContextPack:
    query: str
    primary_chunks: list[SearchResult]
    related_files: list[str]
    test_files: list[str]
    imports: list[str]
    imported_by: list[str]
    symbols: list[SymbolInfo]
    callers: list[SymbolInfo]
    callees: list[SymbolInfo]
    sibling_chunks: list[Chunk]
    dependency_chain: list[str]
    confidence: float
    reasoning_trace: list[str]
    total_tokens: int
```

### `SymbolInfo` / `ImportInfo`

```python
@dataclass(frozen=True, slots=True)
class SymbolInfo:
    name: str
    kind: SymbolKind
    file_path: str
    start_line: int
    end_line: int
    docstring: str | None = None
    parent: str | None = None
    signature: str | None = None

@dataclass(frozen=True, slots=True)
class ImportInfo:
    source_file: str
    imported_module: str
    imported_names: list[str]
    is_relative: bool = False
    alias: str | None = None
```

### `GraphContext` / `GraphPath`

```python
@dataclass(frozen=True, slots=True)
class GraphContext:
    key_symbol: str
    incoming: tuple[str, ...] = ()
    outgoing: tuple[str, ...] = ()

@dataclass(frozen=True, slots=True)
class GraphPath:
    source: str
    target: str
    hops: int
    segments: tuple[tuple[str, str, str], ...]  # (source_label, relation, target_label)
```

### `ChunkConfig`

```python
@dataclass(frozen=True, slots=True)
class ChunkConfig:
    chunk_size: int = 1500          # characters
    min_chunk_size: int | None = None  # defaults to chunk_size // 2
    chunk_overlap: int = 200        # characters
    language: str | None = None
```

### `IndexStats`

```python
@dataclass(frozen=True, slots=True)
class IndexStats:
    indexed_files: int = 0
    total_chunks: int = 0
    languages: dict[str, int] = field(default_factory=dict)
    memo_hits: int = 0
    memo_misses: int = 0
    index_time_ms: float = 0
```

### Enums

- `SearchMode`: `HYBRID`, `SEMANTIC`, `BM25`, `PATH`, `SYMBOL`.
- `SymbolKind`: `FILE`, `CLASS`, `FUNCTION`, `METHOD`, `VARIABLE`, `INTERFACE`,
  `STRUCT`, `ENUM`, `TRAIT`, `TYPE`, `MODULE`, `NAMESPACE`, `UNKNOWN`.
- `EdgeType`: `IMPORTS`, `CALLS`, `USES`, `EXTENDS`, `IMPLEMENTS`, `TESTS`,
  `REFERENCES`, `CONTAINS`, `DEFINED_BY`, `SIBLING`.
- `GraphTraversalMode`: `BFS`, `DFS`.

## Next steps

- Explore the [Knowledge Graph & Scoring](knowledge-graph.md).
- Use vortexa from the [CLI](cli.md) or the [MCP Server](mcp-server.md).
