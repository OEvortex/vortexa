---
name: vortexa
description: Use Vortexa to index and semantically search a local codebase, resolve feature context, and explain symbols or files. Prefer Vortexa MCP tools or the vortexa CLI when exploring unfamiliar repositories, tracing implementations, finding callers and tests, or locating code by intent rather than exact text.
license: Apache-2.0
compatibility: Requires Vortexa 0.3.5 or newer. The first model load may require network access to Hugging Face; the MCP server uses the current working directory.
metadata:
  author: VortexAI
  version: "0.3.5"
---

# Vortexa

Vortexa is a hybrid codebase retrieval engine. It combines native LF2/LF4
semantic embeddings with BM25, AST-aware chunking, and repository graph
context. Use it before broad file-by-file reading when the task involves
finding, understanding, or tracing code.

## When to use

- Find a function, class, pattern, or feature by describing its purpose.
- Understand how a feature works across files.
- Locate callers, callees, imports, tests, and related files.
- Explain a file, file:line location, or symbol.
- Search an unfamiliar repository with natural-language queries.

## Preferred workflow

1. Establish the repository root. For MCP, use the server's current working
   directory. For the CLI, pass the root as the positional argument or use
   `--root`.
2. Use the `search` tool or `vortexa search` for discovery.
3. Use `resolve` for a feature-level investigation that needs graph context,
   imports, callers, tests, or related files.
4. Use `explain` for a known file, `file:line`, or symbol.
5. Read only the returned source ranges needed to verify the answer.
6. Re-index with `--force` when working with changes that the watcher or
   incremental index has not noticed.

## MCP tools

When the Vortexa MCP server is configured, prefer these tools over `grep`,
`rg`, `Glob`, and broad file reads:

- `search(query, top_k=10)` — hybrid semantic + BM25 code search.
- `resolve(query, top_k=5)` — search plus graph expansion and context
  compression.
- `explain(location)` — deep-dive into a file, file:line, or symbol.

Use focused queries such as "where are JWT tokens validated?" rather than
single words such as `auth`. Start with 3–5 results for a focused question and
increase `top_k` for broad exploration.

## CLI equivalents

```bash
# Search; indexes the repository when needed
vortexa search "where are access tokens refreshed?" --root /path/to/repo

# Resolve a feature and its related tests, imports, callers, and callees
vortexa resolve "request retry and backoff" --root /path/to/repo

# Explain a known location or symbol
vortexa explain "src/auth.py:42" --root /path/to/repo

# Search using the legacy query form
vortexa -q "database transaction cleanup" /path/to/repo
```

Useful options:

- `--top-k N` controls the number of results.
- `--plain` prints human-readable results instead of JSON.
- `--include-text` includes Markdown, JSON, YAML, and similar text files.
- `--no-index` searches the existing index without re-indexing.
- `--force` rebuilds the index before searching.

## Embedding models

`mini` and `nano` use native LF2 2-bit checkpoints by default. They are fast
and small enough for normal use. Use `mini-full` or `nano-full` when the
original LF4 checkpoint is explicitly required. The `-f/--fast` flag remains
supported as a compatibility flag.

Do not silently switch model families while comparing results: embeddings
from different dimensions or quantization families are not interchangeable
in an existing vector index. Re-index before changing models.

## Output interpretation

For each search result, inspect:

- `file` and `lines` — the precise source location.
- `score` and `source` — ranking and whether the result came from semantic,
  lexical, or hybrid retrieval.
- `content` — the matching chunk, truncated for transport; read the file for
  complete implementation context.

For `resolve`, prioritize primary files first, then inspect the returned tests,
imports, callers, callees, and related files. Treat graph expansion as a
navigation aid, not proof that every neighboring symbol is relevant.

## Safety and efficiency

- Do not use Vortexa as a substitute for executing tests or reading the exact
  source before making a destructive change.
- Avoid asking for very large `top_k` values unless exploring broadly.
- Keep model IDs consistent within one index and comparison run.
- If the server reports that an index is stale, use `--force` or allow the
  watcher to finish before relying on retrieval results.
