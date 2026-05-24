"""BM25 sparse index using bm25s for lexical code search.

Uses bm25s (numpy-based) instead of whoosh for ~10x faster index builds.
Provides persistent BM25 indexing with identifier-aware tokenization.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from vortexa.core.types import Chunk
from vortexa.search.tokens import enrich_for_bm25, tokenize


class BM25Index:
    """BM25 sparse index for code chunk search using bm25s."""

    def __init__(self, index_dir: Path | None = None) -> None:
        self._index_dir = index_dir
        self._index = None
        self._corpus_ids: list[str] = []
        self._tokenized_corpus = None

    def build(self, chunks: list[Chunk], chunk_ids: list[str], persist_dir: Path | None = None) -> None:
        """Build a BM25 index from chunks.

        :param chunks: List of Chunk objects to index.
        :param chunk_ids: Corresponding chunk IDs.
        :param persist_dir: Optional directory to persist the index.
        """
        import bm25s

        self._corpus_ids = chunk_ids
        index_dir = persist_dir or self._index_dir
        if index_dir is None:
            raise ValueError("Either index_dir or persist_dir must be provided")

        # Tokenize corpus
        tokenized_corpus = []
        for chunk in chunks:
            enriched = enrich_for_bm25(chunk.content, chunk.file_path)
            tokens = tokenize(enriched)
            tokenized_corpus.append(tokens)

        # Build index — pass tokenized corpus directly (list of list of tokens)
        retriever = bm25s.BM25()
        retriever.index(tokenized_corpus)
        self._index = retriever
        self._index_dir = index_dir

        # Persist
        if index_dir:
            self._save(index_dir, chunk_ids)

    def _save(self, index_dir: Path, chunk_ids: list[str]) -> None:
        """Persist the BM25 index to disk."""
        index_dir.mkdir(parents=True, exist_ok=True)
        save_dir = index_dir / "bm25_data"
        if save_dir.exists():
            shutil.rmtree(save_dir)
        assert self._index is not None
        self._index.save(str(save_dir))
        # Save chunk ID mapping
        (save_dir / "corpus_ids.json").write_text(json.dumps(chunk_ids))

    def load(self, index_dir: Path) -> bool:
        """Load a persisted BM25 index.

        :return: True if loaded successfully, False otherwise.
        """
        import bm25s

        save_dir = index_dir / "bm25_data"
        ids_file = save_dir / "corpus_ids.json"
        if not save_dir.exists() or not ids_file.exists():
            return False

        try:
            self._index = bm25s.BM25.load(str(save_dir))
            self._corpus_ids = json.loads(ids_file.read_text())
            self._index_dir = index_dir
            return True
        except Exception:
            return False

    def search(
        self,
        query: str,
        chunks: list[Chunk],
        chunk_ids: list[str],
        top_k: int = 10,
        selector: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Search the BM25 index.

        :param query: Search query string.
        :param chunks: List of all chunks (for fallback).
        :param chunk_ids: List of all chunk IDs.
        :param top_k: Maximum results to return.
        :param selector: Optional set of chunk indices to restrict search to.
        :return: List of (chunk_index, score) tuples sorted by score descending.
        """
        import bm25s

        if self._index is None:
            return []

        # Tokenize query
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Search — bm25s.retrieve() accepts List[List[str]] directly
        results, scores = self._index.retrieve([query_tokens], k=min(top_k * 3, len(chunk_ids)))

        # Map results back to chunk indices
        output = []
        for doc_id, score in zip(results[0], scores[0]):
            if score <= 0:
                continue

            # bm25s returns document indices (0-based)
            idx = int(doc_id)

            if selector is not None and idx not in selector:
                continue

            output.append((idx, float(score)))

            if len(output) >= top_k:
                break

        return output

    def clear(self) -> None:
        """Remove the persisted index from disk."""
        if self._index_dir:
            save_dir = self._index_dir / "bm25_data"
            if save_dir.exists():
                shutil.rmtree(save_dir)
        self._index = None
