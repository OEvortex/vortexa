# Release Notes

## vortexa 0.3.7 — 2026-10-26

### Fixed

**File walker now respects the repository-root `.gitignore` and `.git/info/exclude` when indexing a subdirectory.**

Previously `walk_files()` only loaded ignore files (`*.gitignore`, `*.indexignore`) from inside the directory being indexed. When you pointed Vortexa at a subdirectory — common for agents that index a single package or module — the repository root's ignore rules were silently dropped. Files the root `.gitignore` excluded (build artifacts, logs, generated code) would leak into the index and search results.

Now the walker walks upward from the indexed root to find the git repository root and applies every ignore file between the repo root and the indexed root, ordered outermost-first so closer files take precedence (matching git's last-match-wins semantics). This also honours `.git/info/exclude`, the per-repo local ignore file.

- Repo-root `.gitignore` patterns apply when indexing a subdirectory.
- `.git/info/exclude` is honoured.
- A closer `.gitignore` can re-include files the repo root ignored (`!` negations).
- Standalone directories (no `.git` ancestor) behave exactly as before.

### Tests

Added `tests/test_walker.py` with 15 tests covering basic globs, negations, anchored patterns, recursive globs, nested ignore files, `.indexignore`, default-ignored directories, symlinks, repo-root gitignore, `.git/info/exclude`, and subdirectory override. Walker output is also verified against `git ls-files --exclude-standard` on 22 differential cases.

### Changed

- Bumped package version to `0.3.7`.

## vortexa 0.3.6 — 2026-10-17

See the [CHANGELOG.md](CHANGELOG.md) for the full 0.3.6 history.