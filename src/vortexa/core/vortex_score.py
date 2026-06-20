"""Vortex Score — multi-signal retrieval ranking for VortexA v2.

Combines:
  - Embedding similarity (dense)
  - Filename match
  - Path match
  - Symbol overlap
  - Graph proximity
  - Import relationship
  - IDF (BM25-style)
  - Structural signals

All weights are tunable. Default weights chosen for code-search.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class VortexScoreWeights:
    """Weights for combining retrieval signals."""
    embedding: float = 0.40
    filename: float = 0.15
    path: float = 0.10
    symbol: float = 0.15
    graph: float = 0.10
    import_rel: float = 0.05
    idf: float = 0.05
    structural: float = 0.0  # reserved for future

    def as_array(self) -> np.ndarray:
        return np.array([
            self.embedding, self.filename, self.path, self.symbol,
            self.graph, self.import_rel, self.idf, self.structural,
        ])

    def to_dict(self) -> dict:
        return {
            "embedding": self.embedding, "filename": self.filename,
            "path": self.path, "symbol": self.symbol,
            "graph": self.graph, "import_rel": self.import_rel,
            "idf": self.idf, "structural": self.structural,
        }


def tokenize(s: str) -> List[str]:
    """Simple tokenizer: split on non-alphanumeric, lowercase."""
    return re.findall(r"[a-z0-9_]+", s.lower())


def filename_score(query: str, file_path: str) -> float:
    """Score based on filename token overlap.

    Matches "axis" in "axis.py" or "test_axis.py" for query "axis".
    """
    fname = file_path.split("/")[-1].lower()
    fname_tokens = set(re.findall(r"[a-z0-9_]+", fname.replace(".", " ").replace("_", " ")))
    query_tokens = set(tokenize(query))
    if not fname_tokens or not query_tokens:
        return 0.0
    overlap = len(fname_tokens & query_tokens) / len(query_tokens)
    return overlap


def path_score(query: str, file_path: str) -> float:
    """Score based on directory path token overlap.

    Matches "test" in "/path/to/test/file.py" for query "test".
    """
    parts = file_path.lower().split("/")[:-1]  # exclude filename
    path_tokens = set()
    for p in parts:
        path_tokens.update(re.findall(r"[a-z0-9_]+", p.replace("_", " ")))
    query_tokens = set(tokenize(query))
    if not path_tokens or not query_tokens:
        return 0.0
    overlap = len(path_tokens & query_tokens) / len(query_tokens)
    return min(overlap, 1.0)


def symbol_score(query: str, file_symbols: Set[str], chunk_text: str = "") -> float:
    """Score based on symbol name overlap with query."""
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    score = 0.0
    for sym in file_symbols:
        sym_tokens = set(tokenize(sym.split("::")[-1].replace("_", " ")))
        if not sym_tokens:
            continue
        overlap = len(sym_tokens & query_tokens) / len(query_tokens)
        if overlap > 0:
            score = max(score, overlap)
    # Also check chunk text
    if chunk_text:
        chunk_tokens = set(tokenize(chunk_text))
        chunk_overlap = len(chunk_tokens & query_tokens) / max(len(query_tokens), 1)
        score = max(score, chunk_overlap * 0.5)  # lower weight
    return min(score, 1.0)


def idf_score(query: str, file_idf: Dict[str, float], file_path: str) -> float:
    """Score based on IDF (BM25-style) of query terms in the file.

    file_idf: {term: idf_weight} for terms in the query.
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0
    score = sum(file_idf.get(t, 0.0) for t in query_tokens) / max(len(query_tokens), 1)
    return min(score / 5.0, 1.0)  # normalize


def graph_score(file_path: str, seed_files: Set[str],
                adjacency: Dict[str, Set[str]], max_hops: int = 2) -> float:
    """Score based on graph distance from seed files.

    Files that are 1-2 hops from a seed file get a bonus.
    """
    if not seed_files or file_path in seed_files:
        return 0.0
    # BFS from seed files
    visited = set(seed_files)
    frontier = set(seed_files)
    for hop in range(1, max_hops + 1):
        new_frontier = set()
        for n in frontier:
            for nb in adjacency.get(n, set()):
                if nb not in visited:
                    visited.add(nb)
                    new_frontier.add(nb)
        frontier = new_frontier
        if file_path in visited:
            return 1.0 / hop
    return 0.0


def import_score(file_path: str, seed_files: Set[str],
                 imports: Dict[str, Set[str]]) -> float:
    """Score based on import relationships with seed files.

    If file_path imports something that a seed file exports, or vice versa,
    give a small bonus.
    """
    if not seed_files or file_path in seed_files:
        return 0.0
    score = 0.0
    file_imports = imports.get(file_path, set())
    for seed in seed_files:
        seed_imports = imports.get(seed, set())
        # This file imports something the seed file also imports (similar domain)
        if file_imports & seed_imports:
            score += 0.1
        # This file imports the seed (depends on it)
        if seed in file_imports:
            score += 0.3
    return min(score, 1.0)


def vortex_score(
    embedding_sim: float,
    query: str,
    file_path: str,
    *,
    file_symbols: Optional[Set[str]] = None,
    chunk_text: str = "",
    file_idf: Optional[Dict[str, float]] = None,
    seed_files: Optional[Set[str]] = None,
    adjacency: Optional[Dict[str, Set[str]]] = None,
    imports: Optional[Dict[str, Set[str]]] = None,
    weights: Optional[VortexScoreWeights] = None,
) -> Tuple[float, Dict[str, float]]:
    """Compute the Vortex Score for a single (file, query) pair.

    Returns (final_score, component_scores) for debugging.
    """
    w = weights or VortexScoreWeights()
    components = {
        "embedding": float(np.clip(embedding_sim, 0, 1)),
        "filename": filename_score(query, file_path),
        "path": path_score(query, file_path),
        "symbol": symbol_score(query, file_symbols or set(), chunk_text),
        "graph": graph_score(file_path, seed_files or set(), adjacency or {}),
        "import_rel": import_score(file_path, seed_files or set(), imports or {}),
        "idf": idf_score(query, file_idf or {}, file_path),
        "structural": 0.0,
    }
    weight_array = np.array([
        w.embedding, w.filename, w.path, w.symbol,
        w.graph, w.import_rel, w.idf, w.structural,
    ])
    score_array = np.array([
        components["embedding"], components["filename"], components["path"],
        components["symbol"], components["graph"], components["import_rel"],
        components["idf"], components["structural"],
    ])
    # Normalize weights
    if weight_array.sum() > 0:
        weight_array = weight_array / weight_array.sum()
    final = float(np.dot(weight_array, score_array))
    return final, components
