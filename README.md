# vortexa

Codebase indexing and semantic search engine with hybrid (semantic + BM25) retrieval. Works standalone or as an MCP server for AI agents.

## Features

- **Semantic search** — Find code by describing what it does in natural language
- **Hybrid retrieval** — Combines dense embeddings with BM25 keyword scoring
- **AST-aware chunking** — Respects function/class boundaries via tree-sitter
- **Incremental indexing** — Only re-indexes changed files using content hashing
- **Persistent storage** — LMDB-backed vector store with memoized embeddings
- **Live watch mode** — Polls for file changes and auto-re-indexes
- **MCP server** — Expose as a tool for any MCP-compatible agent (Claude Code, Cursor, etc.)

## Installation

```bash
# Core (semantic + BM25 search, line-based chunking)
pip install vortexa

# Full (includes model2vec, sentence-transformers, tree-sitter)
pip install "vortexa[full]"

# Extras (MCP server, live watcher)
pip install "vortexa[full]" fastmcp
```

## Quick start

```python
from vortexa.core.indexer import CodebaseIndexer
from vortexa.core.types import ChunkConfig

indexer = CodebaseIndexer(root=".")
stats = indexer.index()
print(f"Indexed {stats.indexed_files} files, {stats.total_chunks} chunks")

results = indexer.search_with_lineage("authentication middleware", top_k=5)
for r in results:
    print(f"{r.chunk.file_path}:{r.chunk.start_line}  score={r.score:.3f}")
    print(r.chunk.content[:200])
    print()
```

## CLI / MCP Server

```bash
# Start as an MCP server (stdio transport, auto-indexes cwd)
python -m vortexa.interfaces.mcp_server
```

### Usage with AI agents

**Claude Code / Cursor** (or any MCP-compatible agent):

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "vortexa": {
      "command": "python",
      "args": ["-m", "vortexa.interfaces.mcp_server"]
    }
  }
}
```

The server exposes a single `search` tool that accepts a natural language query and returns the most relevant code snippets with file paths, line numbers, and relevance scores.

```
Search: "database connection pool implementation"
→ src/db/pool.py:45-78  score=0.892
  class ConnectionPool:
      def __init__(self, min_connections=5, max_connections=20):
          ...

Search: "how does the heartbeat system work"
→ core/heartbeat.py:12-60  score=0.876
  class HeartbeatScheduler:
      ...
```

## Python API

### Indexing

```python
from vortexa.core.indexer import CodebaseIndexer

# Basic indexing
indexer = CodebaseIndexer(root="/path/to/project")
stats = indexer.index()
# Stats: indexed_files, total_chunks, languages, memo_hits, memo_misses

# With custom chunk config
from vortexa.core.types import ChunkConfig
indexer = CodebaseIndexer(
    root=".",
    chunk_config=ChunkConfig(chunk_size=100, chunk_overlap=10)
)
stats = indexer.index(force=False, include_text_files=True)
```

### Searching

```python
# Hybrid search (semantic + BM25, auto-weighted)
results = indexer.search_with_lineage("error handling", top_k=10)

# Pure semantic search
results = indexer.search_with_lineage("CSV parser", top_k=5, alpha=1.0)

# Pure BM25 keyword search
results = indexer.search_with_lineage("parse csv", top_k=5, alpha=0.0)

# Symbol lookup
results = indexer.find_symbol("ConnectionPool", top_k=5)

# Related chunks (by index)
results = indexer.find_related(chunk_idx=3, top_k=5)
```

### Index stats & management

```python
stats = indexer.stats()      # {indexed_files, total_chunks, languages, ...}
indexer.clear()              # Delete the persistent index
indexer.index(force=True)    # Force full re-index
```

### Live watch mode

```python
from vortexa.interfaces.watcher import IndexWatcher

watcher = IndexWatcher(indexer, poll_interval=3.0)
watcher.start()   # Background thread, auto-reindexes on changes
# ... do work ...
watcher.stop()
```

## Architecture

```
vortexa/
├── core/
│   ├── indexer.py       # CodebaseIndexer — main orchestrator
│   ├── chunking.py      # AST-aware + line-based chunking
│   ├── embedding.py     # Embedding models (Model2Vec, SentenceTransformers)
│   ├── language.py      # Language detection & file extension mapping
│   └── types.py         # Shared data types (Chunk, ChunkConfig, IndexStats, ...)
├── storage/
│   ├── vector_store.py  # LMDB-backed persistent vector store
│   ├── bm25.py          # BM25 keyword index
│   └── walker.py        # File system walker with .gitignore support
├── search/
│   ├── search.py        # Hybrid search orchestrator
│   ├── ranking.py       # Result ranking & symbol detection
│   └── tokens.py        # Identifier tokenization & splitting
└── interfaces/
    ├── mcp_server.py    # MCP server (stdio)
    └── watcher.py       # Live file watching & auto-reindex
```

No top-level `__init__.py` — vortexa is a namespace package (PEP 420). Import directly from subpackages.

## Dependencies

| Package | Required | Used for |
|---------|----------|----------|
| `numpy` | Yes | Vector operations, embeddings |
| `lmdb` | Yes | Persistent vector/chunk storage |
| `pathspec` | Yes | `.gitignore` pattern matching |
| `model2vec` | Optional | Fast static embeddings |
| `sentence-transformers` | Optional | Transformer-based embeddings |
| `tree-sitter-language-pack` | Optional | AST-aware chunking |
| `fastmcp` | Optional | MCP server |

## License

MIT
