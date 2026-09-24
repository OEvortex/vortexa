"""VortexEmbedderV4 — v4 embedder with native LF4/LF2 dequantization and SIF+PC.

Differences from VortexEmbedderV3:
  - v4 weights from VTXAI/vtx-embed-7M (improved Spearman, on-the-fly dequant)
  - No precomputed embedding table — dequantizes per batch (lower RAM)
  - Matryoshka support (truncation to variable dims)
  - SIF IDF weighting
  - Top-K principal component removal (PC)
  - Optional corpus-level SIF/PC fitting

Default model: VTXAI/vtx-embed-7M-lf2.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

_MODEL_ALIASES = {
    "mini": "VTXAI/vtx-embed-7M-lf2",
    "nano": "VTXAI/vtx-embed-1M-lf2",
    "mini-lf2": "VTXAI/vtx-embed-7M-lf2",
    "nano-lf2": "VTXAI/vtx-embed-1M-lf2",
    "mini-full": "VTXAI/vtx-embed-7M",
    "nano-full": "VTXAI/vtx-embed-1M",
}


class VortexEmbedderV4:
    """Thread-safe v4 embedder with SIF+PC and native LF4/LF2 dequantization.

    Default model: VTXAI/vtx-embed-7M-lf2.
    Lazy-loads on first use, cached thereafter.
    """

    DEFAULT_MODEL_ID = "VTXAI/vtx-embed-7M-lf2"
    MINI_MODEL_ID = "VTXAI/vtx-embed-7M-lf2"
    NANO_MODEL_ID = "VTXAI/vtx-embed-1M-lf2"
    MINI_FULL_MODEL_ID = "VTXAI/vtx-embed-7M"
    NANO_FULL_MODEL_ID = "VTXAI/vtx-embed-1M"
    MINI_LF2_MODEL_ID = "VTXAI/vtx-embed-7M-lf2"
    NANO_LF2_MODEL_ID = "VTXAI/vtx-embed-1M-lf2"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        sif_a: float = 0.05,
        sif_pc: float = 1.0,
        pc_k: int = 1,
        matryoshka_dim: Optional[int] = None,
        model_kwargs: Optional[dict] = None,
        tokenizer_kwargs: Optional[dict] = None,
    ) -> None:
        self._model_id = _MODEL_ALIASES.get(model_id, model_id)
        self.sif_a = sif_a
        self.sif_pc = sif_pc
        self.pc_k = pc_k
        self.matryoshka_dim = matryoshka_dim
        self._model_kwargs = model_kwargs or {}
        self._tokenizer_kwargs = tokenizer_kwargs or {}
        self._model = None
        self._lock = threading.Lock()
        self._sif_weights: Optional[np.ndarray] = None
        self._pc_directions: Optional[np.ndarray] = None

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        assert self._model is not None
        return self._model.dim

    def _load_model(self):
        """Load LF4 or LF2 after resolving a Hub ID to a local checkpoint."""
        model_path = Path(self._model_id)
        if not model_path.is_dir():
            from huggingface_hub import snapshot_download

            model_path = Path(snapshot_download(self._model_id))

        config_path = model_path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"model config not found: {config_path}")
        config = json.loads(config_path.read_text())
        quantization = str(config.get("quantization", "")).lower()
        if quantization == "lf2":
            from vortexa.core.lf2_model import VortexEmbedLF2

            return VortexEmbedLF2.from_pretrained(model_path, matryoshka_dim=self.matryoshka_dim)

        from vortexa.core.lf4_v4_model import VortexEmbedV4_5

        return VortexEmbedV4_5.from_pretrained(model_path, matryoshka_dim=self.matryoshka_dim)

    def _ensure_loaded(self) -> None:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info("Loading V4 embedding model: %s", self._model_id)
                    self._model = self._load_model()
                    self._model.sif_a = self.sif_a
                    self._model.sif_pc = self.sif_pc
                    self._model.pc_k = self.pc_k
                    if self.matryoshka_dim is not None:
                        self._model.matryoshka_dim = self.matryoshka_dim

    def fit_corpus(self, texts: List[str]) -> None:
        """Fit SIF IDF and PC removal on a corpus (do this before embedding)."""
        self._ensure_loaded()
        assert self._model is not None
        cap = self._model.DEFAULT_MAX_CHARS_PER_TEXT if hasattr(self._model, "DEFAULT_MAX_CHARS_PER_TEXT") else 50_000
        capped = [t if len(t) <= cap else t[: cap // 2] + t[-(cap - cap // 2):] for t in texts]
        tl = self._model._tokenize_batch(capped)
        self._model.fit_idf(tl)
        embs = self._model.encode_batch(capped, normalize=True)
        self._model.fit_pc(embs, k=self.pc_k)
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

    def encode(
        self,
        text: str,
        *,
        normalize: bool = True,
        truncate_dim: Optional[int] = None,
    ) -> npt.NDArray[np.float32]:
        self._ensure_loaded()
        assert self._model is not None
        return self._model.encode(text, normalize=normalize, truncate_dim=truncate_dim)

    def encode_batch(
        self,
        texts: List[str],
        *,
        normalize: bool = True,
        truncate_dim: Optional[int] = None,
    ) -> npt.NDArray[np.float32]:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        self._ensure_loaded()
        assert self._model is not None
        return self._model.encode_batch(texts, normalize=normalize, truncate_dim=truncate_dim)

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
        return ("VortexEmbedderV4", self._model_id, self.sif_a, self.pc_k, self.matryoshka_dim)
