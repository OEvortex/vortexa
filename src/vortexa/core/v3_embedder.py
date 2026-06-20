"""VortexEmbedderV3 — v3 embedder with SIF+PC and graph-friendly features.

Differences from LF4Embedder (v1):
  - v3 weights from VTXAI/Vortex-Embed-v3-sentence (improved Spearman 0.7560)
  - SIF IDF weighting
  - Top-1 principal component removal (PC)
  - Optional corpus-level SIF/PC fitting
  - Better chunk-level pooling
  - Search-method that does cosine + returns top-k paths

This is the v3-era embedder. Use it as the default.
"""
from __future__ import annotations

import json
import logging
import math
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class VortexEmbedderV3:
    """Thread-safe v3 embedder with SIF+PC.

    Default model: VTXAI/Vortex-Embed-v3-sentence.
    Lazy-loads on first use, cached thereafter.
    """

    DEFAULT_MODEL_ID = "VTXAI/Vortex-Embed-v3-sentence"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        sif_a: float = 0.01,
        sif_pc: float = 1.0,
        pc_k: int = 1,
    ) -> None:
        self._model_id = model_id
        self.sif_a = sif_a
        self.sif_pc = sif_pc
        self.pc_k = pc_k
        self._model = None
        self._lock = threading.Lock()
        # Corpus-level state (fitted via fit_corpus)
        self._sif_weights: Optional[np.ndarray] = None
        self._pc_directions: Optional[np.ndarray] = None  # shape (pc_k, dim)

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        assert self._model is not None
        return self._model.dim

    def _ensure_loaded(self) -> None:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info("Loading V3 embedding model: %s", self._model_id)
                    from vortexa.core.lf4_v3_model import VortexEmbedV3
                    self._model = VortexEmbedV3.from_pretrained(self._model_id)
                    # Apply our SIF/PC settings
                    self._model.sif_a = self.sif_a
                    self._model.sif_pc = self.sif_pc
                    self._model.pc_k = self.pc_k

    def fit_corpus(self, texts: List[str]) -> None:
        """Fit SIF IDF and PC removal on a corpus (do this before embedding)."""
        self._ensure_loaded()
        assert self._model is not None
        # Cap per-text length to avoid OOM
        cap = self._model.DEFAULT_MAX_CHARS_PER_TEXT
        capped = [t if len(t) <= cap else t[: cap // 2] + t[-(cap - cap // 2):] for t in texts]
        # Tokenize and fit
        tl = self._model._tokenize_batch(capped)
        self._model.fit_idf(tl)
        embs = self._model.encode_batch(capped, normalize=True)
        self._model.fit_pc(embs, k=self.pc_k)
        # Cache the state
        self._sif_weights = self._model._sif_weights
        self._pc_directions = self._model._pc_directions

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        self._ensure_loaded()
        assert self._model is not None
        return self._model.encode(text, normalize=True)

    def embed_batch(self, texts: List[str]) -> npt.NDArray[np.float32]:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        self._ensure_loaded()
        assert self._model is not None
        return self._model.encode_batch(texts, normalize=True)

    def search(
        self,
        query: str,
        index: np.ndarray,
        paths: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> List[dict]:
        """Search index for query. Returns list of {path, score, rank}."""
        q_emb = self.embed(query)
        scores = index @ q_emb
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_idx, 1):
            results.append({
                "rank": rank,
                "idx": int(idx),
                "score": float(scores[idx]),
                "path": paths[int(idx)] if paths else None,
            })
        return results

    @property
    def memo_key(self) -> tuple:
        return ("VortexEmbedderV3", self._model_id, self.sif_a, self.pc_k)
