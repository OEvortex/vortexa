# Vortexa Documentation

Vortexa is a codebase indexing and semantic search engine designed for AI
agents and developers. It builds a persistent, hybrid search index over source
code using dense embeddings (Vortex-Embed), sparse BM25 keyword matching, and
adaptive fusion ranking.

## Quick Links

- [Inference API](inference.md) — Embed, search, and retrieve with Vortex-Embed
  models as a standalone inference engine.
- [Model Configuration](models.md) — Available models, aliases, and how to
  select them.
- [CLI Search](../README.md#cli-search) — Command-line search usage.
- [Python API](../README.md#python-api) — Library usage for indexing and
  search.
- [Architecture](../README.md#architecture) — System design overview.
- [Changelog](../CHANGELOG.md) — Version history and changes.

---

## Overview

Vortexa ships with two main APIs:

| API | Purpose | Entry Point |
|-----|---------|-------------|
| **Indexer** | Codebase indexing, chunking, hybrid search | `vortexa.core.indexer.CodebaseIndexer` |
| **Inference** | Standalone text embedding for VTX-Embed models | `vortexa.core.inference.VortexEmbedInference` |

## Installation

```bash
pip install vortexa
```

Install optional dependencies for full functionality:

```bash
pip install "vortexa[full]"     # all embedders + tree-sitter
pip install "vortexa[mcp]"      # MCP server support
```

## Quick Start

### Python — Indexing and Search

```python
from vortexa.core.indexer import CodebaseIndexer

indexer = CodebaseIndexer(root="/path/to/project")
indexer.index()
results = indexer.search("authentication middleware")

for r in results[:5]:
    print(f"{r.file_path}:{r.chunk.start_line} score={r.score:.4f}")
```

### Python — Inference (Embedding)

```python
from vortexa.core.inference import VortexEmbedInference

# Load the mini model (default)
model = VortexEmbedInference("mini")

# Encode a single string
vec = model.encode("hello world")
print(vec.shape)  # (1, 256)

# Encode a batch
vecs = model.encode(["hello", "world"])
print(vecs.shape)  # (2, 256)

# Truncate to 128 dimensions (Matryoshka)
vec_small = model.encode("hello world", dim=128)
print(vec_small.shape)  # (1, 128)

# Model info
print(model.model_id)   # VTXAI/vtx-embed-7M
print(model.dim)        # 256
```

### CLI — Embedding

```bash
# Encode text with the default (mini) model
vortexa embed "hello world"

# Encode with the nano model
vortexa embed "hello world" --model nano

# Truncate to 64 dimensions
vortexa embed "hello world" --model nano --dim 64

# Batch encode from file
vortexa embed -f queries.txt --model mini
```

### CLI — Search

```bash
vortexa -q "authentication middleware" /path/to/project
vortexa -q "CSV parser" --top-k 20 --model nano /path/to/project
```