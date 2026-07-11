# Contributing

Thanks for your interest in improving vortexa. This guide covers the development
setup, running tests, and project conventions.

## Development setup

vortexa uses [uv](https://github.com/astral-sh/uv) for environment management, but
any virtualenv works.

```bash
# Clone
git clone https://github.com/OEvortex/vortexa
cd vortexa

# Create a venv and install (editable, with full extras)
python -m venv .venv
source .venv/bin/activate
pip install -e ".[full]"
```

`[full]` pulls in `model2vec`, `sentence-transformers`, and
`tree-sitter-language-pack` so the AST-aware chunking and alternative embedders
are exercised locally.

## Project structure

```text
src/vortexa/
├── core/        # indexer, chunking, parser, graph, embedding, language, types
├── storage/     # vector_store, bm25, session_memory, walker
├── search/      # search, ranking, path_scorer, vortex_score, structural,
│               #   context_expansion, context_compressor, tokens
└── interfaces/  # cli, mcp_server, watcher
tests/           # pytest suite (mirrors src layout)
docs/            # documentation (see index.md)
```

## Running tests

```bash
pytest                 # run the full suite
pytest tests/test_graph.py -q   # a single file
pytest -k "test_parser"  # by name
```

The suite covers indexing, parsing (multi-language), the knowledge graph and its
traversal, context expansion, path scoring, session memory, the CLI, and the
watcher. All tests currently pass on a clean checkout.

## Code style

- **Formatter / linter:** Ruff. Configuration lives in `pyproject.toml`
  (line length 100, target `py310`). Selectors: `E`, `W`, `F`, `I`, `B`, `C4`,
  `UP`; `E501` and `B008` are ignored.
- **Typing:** the codebase targets Python 3.10+ and uses `from __future__ import
  annotations`. Keep public functions typed.
- **Dataclasses:** shared data types are frozen, `slots=True` dataclasses in
  `core.types`.
- **Imports:** within a module, prefer lazy imports for optional/heavy
  dependencies (e.g. `model2vec`, `sentence_transformers`, `torch`) so the base
  install stays light.

Run checks locally before opening a PR:

```bash
ruff check src tests
ruff format --check src tests
```

## Adding a feature

1. **Core retrieval change** → update `core/` and the relevant `search/` module.
2. **New agent capability** → expose it on `CodebaseIndexer`, then add an MCP
   tool in `interfaces/mcp_server.py` and a CLI subcommand in
   `interfaces/cli.py` if applicable.
3. **Docs** → update `README.md` and the matching page under `docs/`.
4. **Tests** → add or extend a `tests/test_*.py` module.

## Persistence & the `.jarvis` directory

During development you will create `.jarvis/index` in whatever root you index.
It is git-ignored. Delete it (or call `indexer.clear()`) to force a clean
re-index when changing chunking or embedding models.

## Releasing

- Version is defined once in `pyproject.toml` (`[project] version`).
- Update `CHANGELOG`/README highlights alongside version bumps.
- Build & publish: `python -m build` then upload to PyPI (maintainers only).

## Code of conduct

Be respectful and constructive. Keep changes focused, tested, and documented.

## Next steps

- Read the [Architecture](architecture.md) to understand the layers.
- See the [Python API Reference](python-api.md) for the public surface.
