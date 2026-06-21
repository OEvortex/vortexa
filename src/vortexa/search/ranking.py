"""Result ranking and boosting for the codebase indexer.

Handles:
- Symbol definition boosting (boost chunks that define queried symbols)
- File path penalties (test files, __init__.py, compat dirs, etc.)
- Multi-chunk file promotion (files with multiple relevant chunks)
- File saturation decay (prevent single files from dominating)
- Alpha auto-detection (symbol queries vs natural language)
- Structural boost (import proximity, test relationships, reference density)
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable
from pathlib import Path

from vortexa.core.types import Chunk
from vortexa.search.tokens import split_identifier

# Symbol-lookup queries: namespace-qualified, leading-underscore, or containing
# uppercase/underscore. Plain lowercase words (e.g. "session") are NL, not symbols.
_SYMBOL_QUERY_RE = re.compile(
    r"^(?:"
    r"[A-Za-z_][A-Za-z0-9_]*(?:(?:::|\\|->|\.)[A-Za-z_][A-Za-z0-9_]*)+"  # namespace-qualified
    r"|_[A-Za-z0-9_]*"  # leading underscore
    r"|[A-Za-z][A-Za-z0-9]*[A-Z_][A-Za-z0-9_]*"  # contains uppercase or underscore
    r"|[A-Z][A-Za-z0-9]*"  # starts with uppercase
    r")$"
)

# CamelCase/camelCase identifiers embedded in a NL query
_EMBEDDED_SYMBOL_RE = re.compile(
    r"\b(?:"
    r"[A-Z][a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*"  # PascalCase
    r"|[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]+"  # camelCase
    r")\b"
)

_EMBEDDED_STEM_MIN_LEN = 4
_EMBEDDED_SYMBOL_BOOST_SCALE = 0.5

_DEFINITION_KEYWORDS = (
    "class",
    "module",
    "defmodule",
    "def",
    "interface",
    "struct",
    "enum",
    "trait",
    "type",
    "func",
    "function",
    "object",
    "abstract class",
    "data class",
    "fn",
    "fun",
    "package",
    "namespace",
    "protocol",
    "record",
    "typedef",
)

_DEFINITION_KEYWORD_BODY = "|".join(re.escape(kw) for kw in _DEFINITION_KEYWORDS)
_KEYWORD_PREFIX = r"(?:^|(?<=\s))(?:"

_DEFINITION_BOOST_MULTIPLIER = 3.0
_STEM_BOOST_MULTIPLIER = 1.0
_FILE_COHERENCE_BOOST_FRAC = 0.2

# Path penalties
_TEST_FILE_RE = re.compile(
    r"(?:^|/)(?:"
    r"test_[^/]*\.py"
    r"|[^/]*_test\.py"
    r"|[^/]*_test\.go"
    r"|[^/]*Tests?\.java"
    r"|[^/]*\.test\.[jt]sx?"
    r"|[^/]*\.spec\.[jt]sx?"
    r"|test_helpers?[^/]*\.\w+"
    r")$"
)
_TEST_DIR_RE = re.compile(r"(?:^|/)(?:tests?|__tests__|spec|testing)(?:/|$)")
_COMPAT_DIR_RE = re.compile(r"(?:^|/)(?:compat|_compat|legacy)(?:/|$)")
_EXAMPLES_DIR_RE = re.compile(r"(?:^|/)(?:_?examples?|docs?_src)(?:/|$)")
_TYPE_DEFS_RE = re.compile(r"\.d\.ts$")

_STRONG_PENALTY = 0.3
_MODERATE_PENALTY = 0.5
_MILD_PENALTY = 0.7

_REEXPORT_FILENAMES = frozenset({"__init__.py", "package-info.java"})
_FILE_SATURATION_THRESHOLD = 1
_FILE_SATURATION_DECAY = 0.5

_STOPWORDS = frozenset(
    "a an and are as at be by do does for from has have how if in is it not of on or the to was"
    " what when where which who why with".split()
)

# Alpha auto-detection
_ALPHA_SYMBOL = 0.3  # lean BM25 for exact keyword matching
_ALPHA_NL = 0.5  # balanced semantic + BM25


def is_symbol_query(query: str) -> bool:
    """Return True if the query looks like a bare symbol or namespace-qualified identifier."""
    return _SYMBOL_QUERY_RE.match(query.strip()) is not None


def resolve_alpha(query: str, alpha: float | None) -> float:
    """Return the blending weight for semantic scores, auto-detecting from query type."""
    if alpha is not None:
        return alpha
    return _ALPHA_SYMBOL if is_symbol_query(query) else _ALPHA_NL


def apply_query_boost(
    combined_scores: dict[Chunk, float],
    query: str,
    all_chunks: list[Chunk],
) -> dict[Chunk, float]:
    """Apply query-type boosts to candidate scores."""
    if not combined_scores:
        return combined_scores

    max_score = max(combined_scores.values())
    boosted = dict(combined_scores)

    if is_symbol_query(query):
        _boost_symbol_definitions(boosted, query, max_score, all_chunks)
    else:
        _boost_stem_matches(boosted, query, max_score)
        _boost_embedded_symbols(boosted, query, max_score, all_chunks)

    return boosted


def boost_multi_chunk_files(scores: dict[Chunk, float]) -> None:
    """Promote files with multiple high-scoring chunks by boosting their top chunk (in-place)."""
    if not scores:
        return

    max_score = max(scores.values())
    if max_score == 0.0:
        return

    file_sum: dict[str, float] = {}
    best_chunk: dict[str, Chunk] = {}
    for chunk, score in scores.items():
        file_path = chunk.file_path
        file_sum[file_path] = file_sum.get(file_path, 0.0) + score
        if file_path not in best_chunk or score > scores[best_chunk[file_path]]:
            best_chunk[file_path] = chunk

    max_file_sum = max(file_sum.values())
    boost_unit = max_score * _FILE_COHERENCE_BOOST_FRAC
    for file_path, chunk in best_chunk.items():
        scores[chunk] += boost_unit * file_sum[file_path] / max_file_sum


def rerank_topk(
    scores: dict[Chunk, float],
    top_k: int,
    *,
    penalise_paths: bool = True,
) -> list[tuple[Chunk, float]]:
    """Select top-k results with optional file-path penalties and file-saturation decay."""
    if not scores:
        return []

    penalty_cache: dict[str, float] = {}
    penalised: dict[Chunk, float] = {}
    for chunk, score in scores.items():
        if penalise_paths:
            if chunk.file_path not in penalty_cache:
                penalty_cache[chunk.file_path] = _file_path_penalty(chunk.file_path)
            penalised[chunk] = score * penalty_cache[chunk.file_path]
        else:
            penalised[chunk] = score

    ranked = sorted(penalised, key=lambda c: -penalised[c])

    file_selected: dict[str, int] = {}
    selected: list[tuple[float, Chunk]] = []
    min_selected = float("+inf")

    for chunk in ranked:
        pen_score = penalised[chunk]

        if len(selected) >= top_k and pen_score <= min_selected:
            break

        already_selected = file_selected.get(chunk.file_path, 0)
        eff_score = pen_score
        if already_selected >= _FILE_SATURATION_THRESHOLD:
            excess = already_selected - _FILE_SATURATION_THRESHOLD + 1
            eff_score *= _FILE_SATURATION_DECAY**excess

        selected.append((eff_score, chunk))
        file_selected[chunk.file_path] = already_selected + 1

        if len(selected) >= top_k:
            min_selected = min(s for s, _ in selected)

    selected.sort(key=lambda t: -t[0])
    return [(chunk, score) for score, chunk in selected[:top_k]]


# ── Internal helpers ──────────────────────────────────────────────────


def _file_path_penalty(file_path: str) -> float:
    """Return a combined multiplicative penalty for all applicable path patterns."""
    normalised = file_path.replace("\\", "/")
    penalty = 1.0
    if _TEST_FILE_RE.search(normalised) is not None or _TEST_DIR_RE.search(normalised) is not None:
        penalty *= _STRONG_PENALTY
    if Path(file_path).name in _REEXPORT_FILENAMES:
        penalty *= _MODERATE_PENALTY
    if _COMPAT_DIR_RE.search(normalised):
        penalty *= _STRONG_PENALTY
    if _EXAMPLES_DIR_RE.search(normalised):
        penalty *= _STRONG_PENALTY
    if _TYPE_DEFS_RE.search(normalised):
        penalty *= _MILD_PENALTY
    return penalty


def _extract_symbol_name(query: str) -> str:
    """Extract the final identifier from a possibly namespace-qualified query."""
    for separator in ("::", "\\", "->", "."):
        if separator in query:
            return query.rsplit(separator, 1)[-1]
    return query.strip()


@functools.lru_cache(maxsize=256)
def _definition_pattern(symbol_name: str) -> re.Pattern[str]:
    escaped = re.escape(symbol_name)
    ns_prefix = r"(?:[A-Za-z_][A-Za-z0-9_]*(?:\.|::))*"
    suffix = r")\s+" + ns_prefix + escaped + r"(?:\s|[<({:\[;]|$)"
    return re.compile(_KEYWORD_PREFIX + _DEFINITION_KEYWORD_BODY + suffix, re.MULTILINE)


def _chunk_defines_symbol(chunk: Chunk, symbol_name: str) -> bool:
    """Return True if the chunk contains a definition of symbol_name."""
    return _definition_pattern(symbol_name).search(chunk.content) is not None


def _stem_matches(stem: str, name: str) -> bool:
    """Return True if stem matches name (exact, snake_case-normalised, or plural)."""
    stem_norm = stem.replace("_", "")
    return stem == name or stem_norm == name or stem.rstrip("s") == name or stem_norm.rstrip("s") == name


def _definition_tier(chunk: Chunk, names: set[str], boost_unit: float) -> float:
    """Return the boost amount for a chunk that defines one of names (0.0 if none match)."""
    if not any(_chunk_defines_symbol(chunk, name) for name in names):
        return 0.0
    stem = Path(chunk.file_path).stem.lower()
    return boost_unit * (1.5 if any(_stem_matches(stem, name.lower()) for name in names) else 1.0)


def _scan_non_candidates(
    boosted: dict[Chunk, float],
    names: set[str],
    boost_unit: float,
    all_chunks: list[Chunk],
    stem_ok: Callable[[str], bool],
) -> None:
    """Boost non-candidate chunks whose lowercased file stem satisfies stem_ok (in-place)."""
    for chunk in all_chunks:
        if chunk in boosted:
            continue
        if not stem_ok(Path(chunk.file_path).stem.lower()):
            continue
        if tier := _definition_tier(chunk, names, boost_unit):
            boosted[chunk] = tier


def _boost_symbol_definitions(
    boosted: dict[Chunk, float],
    query: str,
    max_score: float,
    all_chunks: list[Chunk],
) -> None:
    """Boost chunks that define the queried symbol."""
    symbol_name = _extract_symbol_name(query)
    names = {symbol_name}
    if symbol_name != query.strip():
        names.add(query.strip())

    boost_unit = max_score * _DEFINITION_BOOST_MULTIPLIER

    for chunk in list(boosted):
        if tier := _definition_tier(chunk, names, boost_unit):
            boosted[chunk] += tier

    _scan_non_candidates(
        boosted,
        names,
        boost_unit,
        all_chunks,
        lambda stem: _stem_matches(stem, symbol_name.lower()),
    )


def _boost_embedded_symbols(
    boosted: dict[Chunk, float],
    query: str,
    max_score: float,
    all_chunks: list[Chunk],
) -> None:
    """Boost chunks defining CamelCase/camelCase symbols embedded in NL queries."""
    names = set(_EMBEDDED_SYMBOL_RE.findall(query))
    if not names:
        return

    boost_unit = max_score * _DEFINITION_BOOST_MULTIPLIER * _EMBEDDED_SYMBOL_BOOST_SCALE

    for chunk in list(boosted):
        if tier := _definition_tier(chunk, names, boost_unit):
            boosted[chunk] += tier

    symbols_lower = frozenset(s.lower() for s in names)
    for chunk in all_chunks:
        if chunk in boosted:
            continue
        stem = Path(chunk.file_path).stem.lower()
        stem_norm = stem.replace("_", "")
        if not any(
            stem == symbol_lower
            or stem_norm == symbol_lower
            or (len(stem) >= _EMBEDDED_STEM_MIN_LEN and symbol_lower.startswith(stem))
            or (len(stem_norm) >= _EMBEDDED_STEM_MIN_LEN and symbol_lower.startswith(stem_norm))
            for symbol_lower in symbols_lower
        ):
            continue
        if tier := _definition_tier(chunk, names, boost_unit):
            boosted[chunk] = tier


def _count_keyword_matches(keywords: set[str], parts: set[str]) -> int:
    """Count query keywords that match path parts, allowing prefix overlap (min 3 chars)."""
    exact = keywords & parts
    if len(exact) == len(keywords):
        return len(exact)
    n_matches = len(exact)
    for keyword in keywords - exact:
        for part in parts:
            shorter, longer = (keyword, part) if len(keyword) <= len(part) else (part, keyword)
            if len(shorter) >= 3 and longer.startswith(shorter):
                n_matches += 1
                break
    return n_matches


def _boost_stem_matches(
    boosted: dict[Chunk, float],
    query: str,
    max_score: float,
) -> None:
    """Boost chunks whose file paths match NL query keywords."""
    keywords = {
        word.lower()
        for word in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", query)
        if len(word) > 2 and word.lower() not in _STOPWORDS
    }
    if not keywords:
        return

    boost = max_score * _STEM_BOOST_MULTIPLIER
    path_cache: dict[str, set[str]] = {}
    for chunk in list(boosted):
        if chunk.file_path not in path_cache:
            path = Path(chunk.file_path)
            parts: set[str] = set(split_identifier(path.stem))
            if path.parent.name and path.parent.name not in (".", "/", ".."):
                parts.update(split_identifier(path.parent.name))
            path_cache[chunk.file_path] = parts
        n_matches = _count_keyword_matches(keywords, path_cache[chunk.file_path])
        if n_matches > 0:
            match_ratio = n_matches / len(keywords)
            if match_ratio >= 0.10:
                boosted[chunk] += boost * match_ratio
