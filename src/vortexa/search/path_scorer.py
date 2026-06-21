"""Path-based retrieval and scoring for code search.

The filename and directory structure are first-class retrieval signals,
not just ranking hints. This module provides path-only retrieval and
path-aware scoring features.
"""

from __future__ import annotations

import re
from pathlib import Path

from vortexa.search.tokens import split_identifier

_STOPWORDS = frozenset(
    "a an and are as at be by do does for from has have how if in is it not of on or the to was"
    " what when where which who why with".split()
)


def path_score(query: str, file_path: str) -> float:
    """Score how well a file path matches a query based on path features.

    Returns a normalized score in [0.0, 1.0].
    """
    path = Path(file_path)
    scores: list[float] = []

    # 1. Filename exact match
    stem = path.stem.lower()
    query_lower = query.lower().strip()
    if stem == query_lower:
        scores.append(1.0)
    elif stem.rstrip("s") == query_lower or query_lower.rstrip("s") == stem:
        scores.append(0.9)
    elif query_lower in stem:
        scores.append(0.7)
    elif stem in query_lower:
        scores.append(0.5)

    # 2. Directory name match
    for part in path.parts:
        part_stem = Path(part).stem.lower()
        if part_stem == query_lower:
            scores.append(0.8)
        elif query_lower in part_stem or part_stem in query_lower:
            scores.append(0.4)

    # 3. Identifier-split match
    path_tokens = _get_path_tokens(file_path)
    query_tokens = _tokenize_query(query)
    if query_tokens and path_tokens:
        matches = query_tokens & path_tokens
        if matches:
            ratio = len(matches) / len(query_tokens)
            scores.append(min(ratio, 1.0) * 0.8)

    # 4. Module path match (package.subpackage.module pattern)
    module_path = str(path).replace("/", ".").replace("\\", ".").rstrip(".py").lower()
    if query_lower in module_path:
        scores.append(0.6)

    return max(scores) if scores else 0.0


def path_retrieve(query: str, all_files: list[str], top_k: int = 10) -> list[tuple[str, float]]:
    """Pure path-based retrieval: find files matching query by path only.

    No embeddings, no BM25 — just filename and directory structure matching.
    """
    scored: list[tuple[str, float]] = []
    for file_path in all_files:
        score = path_score(query, file_path)
        if score > 0.0:
            scored.append((file_path, score))

    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


def filename_features(query: str) -> list[str]:
    """Extract path-like features from a query string.

    E.g., "Fix OAuth bug in auth_service.py" → ["auth_service", "auth_service.py", "oauth"]
    """
    features: list[str] = []

    # Extract explicit filenames (e.g. "auth_service.py")
    file_refs = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*\.[a-z]+", query)
    features.extend(f.lower() for f in file_refs)

    # Extract module-like segments (e.g. "auth_service", "AuthService")
    identifiers = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", query)
    for ident in identifiers:
        if len(ident) > 2 and ident.lower() not in _STOPWORDS:
            features.append(ident.lower())
            # Split and add sub-tokens
            for token in split_identifier(ident):
                if len(token) > 2:
                    features.append(token.lower())

    return list(set(features))


def _get_path_tokens(file_path: str) -> set[str]:
    """Extract all meaningful tokens from a file path."""
    path = Path(file_path)
    tokens: set[str] = set()

    # Filename stem
    stem = path.stem.lower()
    tokens.add(stem)
    tokens.update(t.lower() for t in split_identifier(stem) if len(t) > 2)

    # Directory parts
    for part in path.parts:
        part_stem = Path(part).stem.lower()
        if part_stem not in (".", "..", ""):
            tokens.add(part_stem)
            tokens.update(t.lower() for t in split_identifier(part_stem) if len(t) > 2)

    # Module path
    module = str(path).replace("/", ".").replace("\\", ".").lower()
    for segment in module.split("."):
        segment = segment.strip()
        if segment and len(segment) > 2:
            tokens.add(segment)

    return tokens


def _tokenize_query(query: str) -> set[str]:
    """Tokenize a query into a set of meaningful tokens."""
    tokens: set[str] = set()
    for ident in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", query):
        lower = ident.lower()
        if len(lower) > 2 and lower not in _STOPWORDS:
            tokens.add(lower)
            tokens.update(t.lower() for t in split_identifier(ident) if len(t) > 2)
    return tokens
