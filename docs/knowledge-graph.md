# Knowledge Graph & Scoring

This page describes the structural knowledge graph, the context-expansion engine,
the Vortex Score reranker, and session memory — the pieces that make vortexa
agent-first rather than just a vector search.

## Knowledge graph

During indexing, `core.graph.build_graph_from_symbols` builds a per-repo graph:

- **File nodes** — one `file:` node per indexed file.
- **Symbol nodes** — one node per class, function, method, struct, enum, trait,
  type, interface, module, or namespace, labeled by name.
- **Edges** between them.

### Node and edge model

Node kinds (`SymbolKind`):

```text
FILE, CLASS, FUNCTION, METHOD, VARIABLE, INTERFACE, STRUCT, ENUM,
TRAIT, TYPE, MODULE, NAMESPACE, UNKNOWN
```

Edge types (`EdgeType`):

```text
IMPORTS, CALLS, USES, EXTENDS, IMPLEMENTS, TESTS,
REFERENCES, CONTAINS, DEFINED_BY, SIBLING
```

The knowledge graph is an in-memory adjacency list (`core.graph.KnowledgeGraph`)
with derived indexes (by kind, by name, by file) for O(1) traversal, and it is
persisted to LMDB so it survives restarts.

### Traversal

- **`bfs_traverse` / `dfs_traverse`** — expand from seed nodes up to `depth`
  hops. High-degree "hub" nodes (above the p99 degree, floored at 50) are not
  expanded *through* as transit, which keeps traversal focused on the
  query-relevant neighborhood instead of fanning out through infrastructure noise.
- **`shortest_path_between`** — BFS over the undirected view; the MCP
  `get_shortest_path` tool caps at `max_hops=8` to keep agent context compact.
- **Query-aware seeding** — `score_nodes_against_query` ranks every node by a
  three-tier match (exact > prefix > substring) weighted by per-term IDF, so rare
  identifiers outrank common words like `error`. `pick_seeds` then keeps only the
  dominant matches.

### Structural-relation filtering

AST extractors can create noisy edges (e.g. `str`/`bool`/`Path` type annotations).
`GRAPH_STRUCTURAL_RELATIONS` and `GRAPH_DEFAULT_RELATION_FILTER` (in
`core.types`) whitelist only real code structure — `call`, `import`, `contains`,
`extends`, `implements`, `references`, etc. — so hybrid search enrichment and
graph queries surface meaningful relationships.

## Context expansion

`search.context_expansion.expand_context` turns a set of primary results into a
`ContextPack`:

| Field | How it is discovered |
|-------|----------------------|
| `primary_chunks` | Top-k hybrid search results. |
| `test_files` | `test_<stem>` / `<stem>_test` siblings of primary files. |
| `imports` | Outgoing `IMPORTS` edges from primary files. |
| `imported_by` | Incoming `IMPORTS` edges into primary files. |
| `symbols` | Symbol nodes defined in the primary files (file nodes excluded). |
| `callers` | Symbols with an incoming `CALLS` edge into a primary symbol. |
| `callees` | Symbols with an outgoing `CALLS` edge from a primary symbol. |
| `sibling_chunks` | Other chunks in the same file (capped at 20). |
| `dependency_chain` | Files reachable via `IMPORTS` edges up to `depth`. |
| `confidence` | Mean of positive primary scores. |
| `reasoning_trace` | Human-readable trace of the expansion. |

`resolve()` runs expansion and then `context_compressor.compress` to fit the
pack within a token budget (default 8000 tokens), followed by
`format_for_agent` for a ready-to-paste block.

## Vortex Score

`search.vortex_score.compute_vortex_score` fuses **eight normalized signals**
`[0, 1]` into a single score, normalized by the sum of weights:

| Signal | Weight (default) | What it captures |
|--------|------------------|------------------|
| `embedding` | `1.0` | Semantic similarity to the query. |
| `filename` | `0.8` | Filename match (path scorer). |
| `path` | `0.6` | Directory/module-path match. |
| `symbol` | `1.2` | Does the chunk define the queried symbol? |
| `graph` | `0.5` | Graph proximity / centrality of the file node. |
| `import` | `0.4` | Imported / importer module matches the query. |
| `bm25_idf` | `0.7` | BM25 lexical score (normalized to `[0, 1]`). |
| `structural` | `0.3` | Incoming-reference (importance) count. |

Defaults are defined in `VortexScoreConfig` (`core.types`) and persisted to
`vortex_weights.json` under the index directory. Because the config is loaded on
index, you can tune the weights and re-run `resolve`/`search(use_vortex_score=True)`
to observe the effect.

Enable it with:

```python
results = indexer.search("auth middleware", top_k=5, use_vortex_score=True)
pack = indexer.resolve("how are JWT tokens validated?", top_k=5)  # always uses it
```

The graph-only signals are skipped when the graph is empty (e.g. before the first
index), so the score degrades gracefully to embedding + BM25 + lexical signals.

## Session memory

`storage.session_memory.SessionMemory` tracks, per session:

- the queries an agent has run,
- the files/symbols it has visited,

so that recently-touched code can be boosted in recall across MCP turns. It is
recorded automatically by `CodebaseIndexer.search` and by the MCP `search` tool,
persisted under `.jarvis/index/session`, and reset by `indexer.clear()`.

## Putting it together

```python
pack = indexer.resolve("how are JWT tokens validated?", top_k=5)

# Primary matches
print([r.chunk.file_path for r in pack.primary_chunks])

# Who depends on this code?
print("imported by:", pack.imported_by)

# How is it called?
print("callers:", [c.name for c in pack.callers])
print("callees:", [c.name for c in pack.callees])

# Drop-in context for an LLM prompt
print(indexer.format_context(pack))
```

## Next steps

- See the [Python API Reference](python-api.md) for method signatures.
- Explore the [Embedding Models](embeddings.md) that power the semantic signal.
