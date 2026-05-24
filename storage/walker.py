"""File discovery with .gitignore support."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from pathspec import GitIgnoreSpec


@dataclass(frozen=True)
class IgnoreSpec:
    base: Path
    spec: GitIgnoreSpec


_DEFAULT_IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git/",
        ".hg/",
        ".svn/",
        "__pycache__/",
        "node_modules/",
        ".venv/",
        "venv/",
        ".tox/",
        ".mypy_cache/",
        ".pytest_cache/",
        ".ruff_cache/",
        ".cache/",
        ".semble/",
        ".next/",
        "dist/",
        "build/",
        ".eggs/",
        ".jarvis/",
    }
)


def _load_ignore_for_dir(directory: Path) -> GitIgnoreSpec | None:
    """Load gitignore and .indexignore for a directory."""
    gitignore = directory / ".gitignore"
    indexignore = directory / ".indexignore"

    lines = []
    if gitignore.is_file():
        lines.extend(gitignore.read_text(encoding="utf-8", errors="ignore").splitlines())
    if indexignore.is_file():
        lines.extend(indexignore.read_text(encoding="utf-8", errors="ignore").splitlines())
    if lines:
        return GitIgnoreSpec.from_lines(lines)
    return None


def walk_files(root: Path, extensions: Sequence[str]) -> Iterator[Path]:
    """Yield files under root matching extensions, skipping ignored paths.

    Directories matching DEFAULT_IGNORED_DIRS are always skipped.
    If the root contains a .gitignore, its patterns are also honoured.

    :param root: Root directory to walk.
    :param extensions: List of file extensions to match.
    :yield: Path to each file under root matching the criteria.
    """
    extensions_set = frozenset(extensions)
    dir_patterns = sorted(_DEFAULT_IGNORED_DIRS)
    base_spec = GitIgnoreSpec.from_lines(dir_patterns, backend="simple")
    s = IgnoreSpec(base=root, spec=base_spec)
    yield from _walk(root, [s], extensions_set)


def _is_ignored(path: Path, specs: list[IgnoreSpec]) -> tuple[bool, bool]:
    """Check if a path is ignored by any of the provided ignore specs."""
    is_dir = path.is_dir()
    ignored = False
    found = False
    for ignore_spec in specs:
        try:
            relative = path.relative_to(ignore_spec.base)
        except ValueError:
            continue

        relative_str = relative.as_posix()
        if is_dir:
            relative_str += "/"

        for pattern in ignore_spec.spec.patterns:
            if pattern.include is None:
                continue

            if pattern.match_file(relative_str) is not None:
                ignored = pattern.include
                pat = pattern.pattern
                found = not ignored and isinstance(pat, str) and bool(Path(pat.rstrip("/")).suffix)

    return ignored, found


def _walk(
    directory: Path,
    inherited_specs: list[IgnoreSpec],
    extensions: frozenset[str],
) -> Iterator[Path]:
    """Recursive function for walking files under a directory."""
    spec = _load_ignore_for_dir(directory)
    if spec is not None:
        inherited_specs = [*inherited_specs, IgnoreSpec(base=directory, spec=spec)]

    try:
        items = sorted(directory.iterdir())
    except PermissionError:
        return

    for item in items:
        if item.is_symlink():
            continue
        is_ignored, found = _is_ignored(item, inherited_specs)
        if is_ignored:
            continue

        if item.is_dir():
            yield from _walk(item, inherited_specs, extensions)
        elif item.is_file() and (found or item.suffix.lower() in extensions):
            yield item
