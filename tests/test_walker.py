"""Tests for the file walker's .gitignore / .indexignore support."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from vortexa.storage.walker import walk_files


def _build(tmp: str, files: list[str]) -> None:
    """Create files under *tmp* (parents are created as needed)."""
    for f in files:
        p = Path(tmp) / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")


def _git_unignored(tmp: str, extensions: list[str]) -> set[str]:
    """Return the set of git-unignored files under *tmp* matching *extensions*."""
    out = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard"],
        cwd=tmp,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return {f for f in out if Path(f).suffix in extensions}


@pytest.mark.parametrize(
    "lines, files",
    [
        # Basic globbing
        (["*.log"], ["a.log", "b.py", "sub/c.log", "sub/d.py"]),
        # Negation re-includes a file
        (["*.log", "!keep.log"], ["a.log", "keep.log", "b.py"]),
        # Anchored directory pattern
        (["build/"], ["build/x.py", "main.py"]),
        # Recursive glob
        (["**/gen/"], ["gen/a.py", "sub/gen/b.py", "main.py"]),
        # Subdirectory glob
        (["sub/*.py"], ["sub/x.py", "sub/deep/y.py", "other/z.py"]),
        # Nested .gitignore in a subdirectory
        (["*.log"], ["a.log", "sub/b.log", "sub/c.py"]),
        # Comments and blank lines
        (["# comment", "", "*.tmp"], ["a.tmp", "b.py"]),
        # Negated pattern with a suffix
        (["*.py", "!sub/keep.py"], ["a.py", "sub/keep.py", "sub/deep/y.py"]),
    ],
)
def test_walker_matches_git_unignored(lines: list[str], files: list[str]) -> None:
    """Walker output must equal `git ls-files --exclude-standard`."""
    extensions = [".py", ".log", ".tmp"]
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / ".gitignore").write_text("\n".join(lines) + "\n")
        _build(tmp, files)
        subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)

        expected = _git_unignored(tmp, extensions)
        actual = {str(p.relative_to(tmp)) for p in walk_files(Path(tmp), extensions)}

        assert actual == expected, (
            f"patterns={lines}\n  expected={sorted(expected)}\n  actual={sorted(actual)}"
        )


def test_nested_gitignore_is_honoured() -> None:
    """A .gitignore in a subdirectory applies only to that subtree."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / ".gitignore").write_text("*.log\n")
        (root / "src").mkdir()
        (root / "src" / ".gitignore").write_text("secret.py\n")
        (root / "src" / "secret.py").write_text("x")
        (root / "src" / "main.py").write_text("x")
        (root / "src" / "a.log").write_text("x")

        result = {
            str(p.relative_to(root)) for p in walk_files(root, [".py", ".log"])
        }
        assert result == {"src/main.py"}


def test_indexignore_is_honoured() -> None:
    """A .indexignore file is treated like a .gitignore."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / ".indexignore").write_text("*.log\n")
        (root / "a.log").write_text("x")
        (root / "b.py").write_text("x")

        result = {str(p.relative_to(root)) for p in walk_files(root, [".py", ".log"])}
        assert result == {"b.py"}


def test_default_ignored_dirs_are_skipped() -> None:
    """Hard-coded default directories are always skipped."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for d in (".git", "__pycache__", ".venv", "node_modules", "dist"):
            (root / d).mkdir()
            (root / d / "x.py").write_text("x")
        (root / "main.py").write_text("x")

        result = {str(p.relative_to(root)) for p in walk_files(root, [".py"])}
        assert result == {"main.py"}


def test_symlinks_are_skipped() -> None:
    """Symlinks are not followed."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "real.py").write_text("x")
        (root / "link.py").symlink_to(root / "real.py")

        result = {str(p.relative_to(root)) for p in walk_files(root, [".py"])}
        assert result == {"real.py"}


def test_repo_root_gitignore_applies_to_subdirectory() -> None:
    """Indexing a subdirectory still honours the repository root .gitignore."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        (repo / ".gitignore").write_text("*.log\n")
        sub = repo / "backend"
        sub.mkdir()
        (sub / "main.py").write_text("x")
        (sub / "debug.log").write_text("x")
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)

        result = {str(p.relative_to(repo)) for p in walk_files(sub, [".py", ".log"])}
        assert result == {"backend/main.py"}


def test_git_info_exclude_applies_to_subdirectory() -> None:
    """A repository's .git/info/exclude is honoured when indexing a subdir."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        sub = repo / "backend"
        sub.mkdir()
        (sub / "main.py").write_text("x")
        (sub / "secret.py").write_text("x")
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        (repo / ".git" / "info" / "exclude").write_text("secret.py\n")

        result = {str(p.relative_to(repo)) for p in walk_files(sub, [".py"])}
        assert result == {"backend/main.py"}


def test_subdirectory_gitignore_overrides_repo_root() -> None:
    """A closer .gitignore can re-include files the repo root ignored."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        (repo / ".gitignore").write_text("*.log\n")
        sub = repo / "backend"
        sub.mkdir()
        (sub / ".gitignore").write_text("!important.log\n")
        (sub / "important.log").write_text("x")
        (sub / "debug.log").write_text("x")
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)

        result = {str(p.relative_to(repo)) for p in walk_files(sub, [".py", ".log"])}
        assert result == {"backend/important.log"}