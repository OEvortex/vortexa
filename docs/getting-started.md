# Getting Started

This guide gets you from zero to a working vortexa index in a few minutes.

## Requirements

- Python **3.10+** (the package is tested against 3.10–3.14).
- ~50 MB of disk for the default embedding model and the on-disk index.
- An internet connection on first run (the default embedding model is downloaded
  from the Hugging Face Hub and cached locally).

## Installation

The base install includes everything you need: hybrid search, the default
`VTXAI/Vortex-Embed-4.7M` embedding model, and the MCP server.

```bash
pip install vortexa
```

For AST-aware chunking (tree-sitter) and alternative embedding backends, install
the `full` extra:

```bash
pip install "vortexa[full]"
```

`full` adds:

- `tree-sitter-language-pack` — precise function/class boundary chunking and
  multi-language symbol extraction.
- `model2vec` — static-embedding alternative.
- `sentence-transformers` — transformer-based dense embeddings.

> The `mcp` extra is retained for backwards compatibility but is equivalent to
> the base install, because `fastmcp` is a required dependency.

Verify the install:

```bash
vortexa --help
python -c "import vortexa; print('ok')"
```

## Quick start

### 1. Index a codebase

```python
from vortexa.core.indexer import CodebaseIndexer

indexer = CodebaseIndexer(root=".")
stats = indexer.index()

print(f"Indexed {stats.indexed_files} files, {stats.total_chunks} chunks")
print(f"Graph: {indexer.graph.node_count} nodes, {indexer.graph.edge_count} edges")
print(f"Index time: {stats.index_time_ms:.0f} ms")
```

`indexer.index()` walks the project, parses each file, builds the knowledge
graph, embeds chunks, and writes everything to a persistent store under
`.jarvis/index` (see [Project layout](#project-layout)).

### 2. Search

```python
results = indexer.search("CSV parser implementation", top_k=5)
for r in results:
    print(f"{r.chunk.file_path}:{r.chunk.start_line}  score={r.score:.3f}")
    print(f"  {r.chunk.content[:150].strip()}")
```

### 3. Resolve agent-ready context

```python
pack = indexer.resolve("how are JWT tokens validated?", top_k=5)
print(indexer.format_context(pack))
```

### From the CLI

```bash
vortexa search "authentication middleware that validates JWT tokens" --plain
vortexa resolve "CSV parser implementation" --top-k 5
vortexa serve
```

## Configuration

### CodebaseIndexer options

```python
CodebaseIndexer(
    root=".",                       # project root to index (Path or str)
    model=None,                     # optional Embedder/Encoder override
    model_id="VTXAI/Vortex-Embed-4.7M",  # default embedder on Hugging Face Hub
    index_dir=None,                 # defaults to <root>/.jarvis/index
    chunk_config=None,              # ChunkConfig override
)
```

### Chunking

Chunking is controlled by `ChunkConfig`. Sizes are measured in **characters**,
not lines.

```python
from vortexa.core.types import ChunkConfig

ChunkConfig(
    chunk_size=1500,        # target size per chunk (default 1500)
    chunk_overlap=200,      # overlap between adjacent chunks (default 200)
    min_chunk_size=None,    # defaults to chunk_size // 2
    language=None,          # optional language hint
)
```

When `tree-sitter` is available for a language, vortexa splits at function / class
/ block boundaries and then merges adjacent pieces up to `chunk_size`, applying
`chunk_overlap` between them. Without tree-sitter it falls back to a line-based
splitter.

### Environment variables

| Variable | Effect |
|----------|--------|
| `VORTEXA_FORCE_POLLING=1` | Force the watcher to use polling instead of native FS events. |
| `HF_HUB_DISABLE_SYMLINKS_WARNING=1` | Silences a Hugging Face symlink warning (set automatically by the MCP server). |

### Indexing options

```python
indexer.index(
    force=False,             # re-embed everything (ignores memoization)
    include_text_files=False,# include .md/.json/.yaml/etc. in the index
    chunk_config=None,       # override chunking for this run
)
```

## Project layout

vortexa writes all persistent state under `<root>/.jarvis/index`:

```text
.jarvis/index/
├── state.lmdb            # chunk vectors, BM25 state, file hashes, chunk memo
├── bm25/                  # BM25 index (persisted)
├── graph/                # knowledge graph (LMDB)
│   └── graph.lmdb
├── file/                  # file-level vector index + meta.json
├── function/              # function-level vector index + meta.json
├── symbol/                # symbol-level vector index + meta.json
├── session/               # session memory (queries + visited symbols)
└── vortex_weights.json    # tunable Vortex Score weights
```

All of it is regenerated automatically; delete `.jarvis/` (or call
`indexer.clear()`) to start fresh.

## Supported languages

vortexa maps **100+ file extensions** to languages and parses **35+ languages**
with tree-sitter (`[full]` install) for accurate symbol extraction. Supported
families include: Python, JavaScript / TypeScript / TSX / JSX, Go, Rust, Ruby,
Java, Kotlin, Swift, C / C++ / C#, Scala, PHP, Lua, Elixir, Erlang, Haskell,
OCaml, Dart, Zig, Nim, Julia, SQL, HTML / CSS / SCSS / LESS, Vue, Svelte, Astro,
Markdown, YAML, JSON, TOML, Bash / Zsh / Fish, Dockerfile, GraphQL, Solidity,
Elm, Clojure, Groovy, Racket, Common Lisp, Fortran, Pascal, Nix, and more.

Without tree-sitter, vortexa still indexes every mapped language using
line-based chunking; only the AST-aware symbol extraction is degraded.

## Next steps

- Learn the [Architecture](architecture.md).
- Browse the full [Python API Reference](python-api.md).
- Wire vortexa into your agent via the [MCP Server](mcp-server.md).
