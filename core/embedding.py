"""Embedding model abstraction for the codebase indexer.

Provides lazy-loading, thread-safe embedders with memoization support.
Inspired by cocoindex's SentenceTransformerEmbedder pattern.
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding models used by the indexer."""

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        ...

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        """Embed a single text string."""
        ...

    def embed_batch(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed a batch of text strings."""
        ...

    @property
    def memo_key(self) -> tuple:
        """Identity key for memoization cache invalidation."""
        ...


class Model2VecEmbedder:
    """Thread-safe, lazy-loading embedder wrapping model2vec.StaticModel.

    The model is loaded on first use and cached. Thread-safe via a lock.
    Memo key includes the model ID for cache invalidation.
    """

    def __init__(self, model_id: str = "AI4free/JARVIS-tool-search-v1") -> None:
        self._model_id = model_id
        self._model = None
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        assert self._model is not None
        return self._model.dim

    def _ensure_loaded(self) -> None:
        if self._model is None:
            with self._lock:
                if self._model is None:  # Double-checked locking
                    from model2vec import StaticModel
                    logger.info("Loading embedding model: %s", self._model_id)
                    self._model = StaticModel.from_pretrained(self._model_id)

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        """Embed a single text string."""
        self._ensure_loaded()
        assert self._model is not None
        return self._model.encode([text])[0]

    def embed_batch(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed a batch of text strings."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        self._ensure_loaded()
        assert self._model is not None
        result = self._model.encode(texts)
        return np.array(result, dtype=np.float32)

    @property
    def memo_key(self) -> tuple:
        """Identity key: (class, model_id)."""
        return ("Model2VecEmbedder", self._model_id)


class SentenceTransformerEmbedder:
    """Thread-safe embedder wrapping sentence-transformers.

    Supports any sentence-transformers model with lazy loading.
    Memo key includes model name and device for cache invalidation.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        assert self._model is not None
        dim = self._model.get_embedding_dimension()
        assert dim is not None
        return dim

    def _ensure_loaded(self) -> None:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    logger.info("Loading sentence-transformers model: %s", self._model_name)
                    self._model = SentenceTransformer(self._model_name, device=self._device)

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        """Embed a single text string."""
        self._ensure_loaded()
        assert self._model is not None
        return self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

    def embed_batch(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed a batch of text strings."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        self._ensure_loaded()
        assert self._model is not None
        return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    @property
    def memo_key(self) -> tuple:
        """Identity key: (class, model_name, device)."""
        return ("SentenceTransformerEmbedder", self._model_name, self._device)


class LF4Embedder:
    """Thread-safe, lazy-loading embedder wrapping LF4StaticEmbedding (4-bit quantized).

    Uses the VTXAI/Vortex-Embed-4.7M model by default — a 4-bit static embedding
    model with ~3.5 MB footprint. Loads on first use, cached thereafter.
    """

    def __init__(self, model_id: str = "VTXAI/Vortex-Embed-4.7M") -> None:
        self._model_id = model_id
        self._model = None
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        assert self._model is not None
        return self._model.dim

    def _ensure_loaded(self) -> None:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info("Loading LF4 embedding model: %s", self._model_id)
                    from vortexa.core.lf4_model import LF4StaticEmbedding
                    self._model = LF4StaticEmbedding.from_pretrained(self._model_id)

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        """Embed a single text string."""
        self._ensure_loaded()
        assert self._model is not None
        return self._model.encode([text])[0]

    def embed_batch(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed a batch of text strings."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        self._ensure_loaded()
        assert self._model is not None
        return self._model.encode(texts)

    @property
    def memo_key(self) -> tuple:
        return ("LF4Embedder", self._model_id)
