# vortexa Documentation

Welcome to the vortexa documentation. vortexa is an **agent-first codebase
context engine**: a knowledge graph, hybrid semantic + keyword search, automatic
context expansion, a weighted Vortex Score, session memory, and an MCP server —
all over a persistent, incremental index.

## Start here

- **New to vortexa?** Read [Getting Started](getting-started.md) for installation,
  a quick start, configuration, and the on-disk project layout.
- **Want the big picture?** See [Architecture](architecture.md) for the data flow,
  module layout, indexing pipeline, and persistence model.

## Reference

| Guide | What it covers |
|-------|----------------|
| [Getting Started](getting-started.md) | Installation, quick start, configuration, project layout, languages. |
| [Architecture](architecture.md) | System design, layers, indexing pipeline, persistence (LMDB). |
| [Python API Reference](python-api.md) | `CodebaseIndexer` methods and the public dataclasses/types. |
| [CLI Reference](cli.md) | The `search`, `resolve`, `explain`, and `serve` subcommands and the legacy `-q` mode. |
| [MCP Server](mcp-server.md) | The 11 MCP tools and how to wire vortexa into Claude Code / Cursor. |
| [Knowledge Graph & Scoring](knowledge-graph.md) | Graph model, edge types, context expansion, the Vortex Score, session memory. |
| [Embedding Models](embeddings.md) | The default LF4 model and how to plug in Model2Vec / SentenceTransformers. |
| [Contributing](contributing.md) | Development setup, running tests, project conventions. |

## At a glance

```python
from vortexa.core.indexer import CodebaseIndexer

indexer = CodebaseIndexer(root=".")
indexer.index()

# Agent-ready context in one call:
pack = indexer.resolve("where is JWT validation implemented?", top_k=5)
print(indexer.format_context(pack))
```

Or from the terminal:

```bash
vortexa resolve "where is JWT validation implemented?" --top-k 5
vortexa serve   # MCP server on stdio
```
