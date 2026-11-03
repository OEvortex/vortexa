"""BM25 sparse index using bm25s for lexical code search.

Uses bm25s (numpy-based) instead of whoosh for ~10x faster index builds.
Provides persistent BM25 indexing with identifier-aware tokenization.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path

from vortexa.core.types import Chunk
from vortexa.search.tokens import enrich_for_bm25, tokenize

logger = logging.getLogger(__name__)


class BM25Index:
    """BM25 sparse index for code chunk search using bm25s."""

    def __init__(self, index_dir: Path | None = None) -> None:
        self._index_dir = index_dir
        self._index = None
        self._corpus_ids: list[str] = []
        self._tokenized_corpus = None

    def build(
        self,
        chunks: list[Chunk],
        chunk_ids: list[str],
        persist_dir: Path | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Build a BM25 index from chunks.

        :param chunks: List of Chunk objects to index.
        :param chunk_ids: Corresponding chunk IDs.
        :param persist_dir: Optional directory to persist the index.
        :param progress: Optional callback receiving (completed, total) chunks.
        """
        index_dir = persist_dir or self._index_dir
        if index_dir is None:
            raise ValueError("Either index_dir or persist_dir must be provided")

        self._corpus_ids = list(chunk_ids)
        total = len(chunks)
        if total == 0:
            self._index = None
            if index_dir.exists():
                save_dir = index_dir / "bm25_data"
                if save_dir.exists():
                    shutil.rmtree(save_dir)
            self._index_dir = index_dir
            if progress is not None:
                progress(0, 0)
            return

        import bm25s

        tokenized_corpus: list[list[str]] = []
        report_every = max(1, total // 100)
        for index, chunk in enumerate(chunks, 1):
            enriched = enrich_for_bm25(chunk.content, chunk.file_path)
            tokenized_corpus.append(tokenize(enriched))
            if progress is not None and (index == total or index % report_every == 0):
                progress(index, total)

        retriever = bm25s.BM25()
        retriever.index(
            tokenized_corpus,
            show_progress=False,
            leave_progress=False,
        )
        self._index = retriever
        self._index_dir = index_dir

        if index_dir:
            self._save(index_dir, chunk_ids)
        if progress is not None:
            progress(total, total)

    def _save(self, index_dir: Path, chunk_ids: list[str]) -> None:
        """Persist the BM25 index to disk."""
        index_dir.mkdir(parents=True, exist_ok=True)
        save_dir = index_dir / "bm25_data"
        if save_dir.exists():
            shutil.rmtree(save_dir)
        assert self._index is not None
        self._index.save(str(save_dir), show_progress=False)
        # Save chunk ID mapping. This is the source of truth for translating
        # BM25 document positions back to chunks after a restart.
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
            logger.debug("Failed to load BM25 index from %s", save_dir, exc_info=True)
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
        :param chunks: List of all chunks (matched by ID).
        :param chunk_ids: List of all chunk IDs in current chunk order.
        :param top_k: Maximum results to return.
        :param selector: Optional set of chunk indices to restrict search to.
        :return: List of (chunk_index, score) tuples sorted by score descending.
        """
        if self._index is None or not chunk_ids or top_k <= 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        limit = min(top_k * 3, len(self._corpus_ids))
        if limit <= 0:
            return []
        results, scores = self._index.retrieve(
            [query_tokens],
            k=limit,
            show_progress=False,
            leave_progress=False,
        )

        # Translate persisted BM25 document positions through chunk IDs. This
        # remains correct even when LMDB returns chunks in a different order
        # than the corpus used to build the BM25 index.
        id_to_position = {cid: index for index, cid in enumerate(chunk_ids)}
        output: list[tuple[int, float]] = []
        for document_id, score in zip(results[0], scores[0], strict=False):
            if score <= 0:
                continue
            document_id = int(document_id)
            if document_id < 0 or document_id >= len(self._corpus_ids):
                continue
            chunk_id = self._corpus_ids[document_id]
            position = id_to_position.get(chunk_id)
            if position is None:
                continue
            if selector is not None and position not in selector:
                continue
            output.append((position, float(score)))
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
        self._corpus_ids = []
