"""Extension-to-language mapping for code files."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

_EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".mts": "typescript",
    ".cts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".swift": "swift",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".lua": "lua",
    ".r": "r",
    ".R": "r",
    ".scala": "scala",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hrl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".mli": "ocaml_interface",
    ".dart": "dart",
    ".zig": "zig",
    ".nim": "nim",
    ".v": "v",
    ".jl": "julia",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "zsh",
    ".fish": "fish",
    ".ps1": "powershell",
    ".psd1": "powershell",
    ".psm1": "powershell",
    ".sql": "sql",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".less": "less",
    ".vue": "vue",
    ".svelte": "svelte",
    ".astro": "astro",
    ".json": "json",
    ".json5": "json5",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".xsl": "xml",
    ".xslt": "xml",
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "rst",
    ".tex": "latex",
    ".dockerfile": "dockerfile",
    ".tf": "terraform",
    ".tfvars": "terraform",
    ".proto": "proto",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".prisma": "prisma",
    ".sol": "solidity",
    ".elm": "elm",
    ".clj": "clojure",
    ".cljc": "clojure",
    ".cljs": "clojure",
    ".groovy": "groovy",
    ".gradle": "groovy",
    ".rkt": "racket",
    ".lisp": "commonlisp",
    ".cl": "commonlisp",
    ".f": "fortran",
    ".f90": "fortran",
    ".pas": "pascal",
    ".pp": "puppet",
    ".cmake": "cmake",
    ".makefile": "make",
    ".mk": "make",
    ".conf": "nginx",
    ".ini": "ini",
    ".cfg": "ini",
    ".bat": "batch",
    ".cmd": "batch",
    ".vim": "vim",
    ".nu": "nushell",
    ".typst": "typst",
    ".nix": "nix",
    ".dhall": "dhall",
}

_DOC_LANGUAGES: frozenset[str] = frozenset(
    {
        "json",
        "json5",
        "yaml",
        "toml",
        "markdown",
        "rst",
        "latex",
        "html",
        "xml",
        "csv",
        "ini",
        "properties",
    }
)

_LANGUAGE_TO_EXTENSION: dict[str, list[str]] = defaultdict(list)
for _ext, _lang in _EXTENSION_TO_LANGUAGE.items():
    _LANGUAGE_TO_EXTENSION[_lang].append(_ext)


def detect_language(path: Path | str) -> str | None:
    """Detect the programming language of a file from its extension."""
    p = Path(path) if isinstance(path, str) else path
    return _EXTENSION_TO_LANGUAGE.get(p.suffix.lower())


def get_extensions(include_text_files: bool = False) -> list[str]:
    """Return a sorted list of supported file extensions."""
    if include_text_files:
        return sorted(_EXTENSION_TO_LANGUAGE.keys())
    return sorted(
        ext
        for ext, lang in _EXTENSION_TO_LANGUAGE.items()
        if lang not in _DOC_LANGUAGES
    )
