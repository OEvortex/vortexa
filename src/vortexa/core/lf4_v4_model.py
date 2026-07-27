"""Vortex-Embed v4.5 — Native 4-bit Sentence Embedding Model.

Adapted from VTXAI/vtx-embed-7M (HuggingFace).
On-the-fly dequantization per batch — no precomputed embedding table.
Matryoshka representation learning support.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
from safetensors.numpy import load_file, save_file

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None  # ty:ignore[invalid-assignment]


@dataclass
class VortexEmbedConfigV4:
    vocab_size: int = 29528
    embedding_dim: int = 256
    block_size: int = 32
    num_blocks: int = 8
    model_type: str = "vortex-embed"
    architectures: Optional[List[str]] = None
    quantization: str = "lf4"
    bits: int = 4
    sif_a: float = 0.05
    sif_pc: float = 1.0
    pc_k: int = 1
    matryoshka_dim: Optional[int] = None

    def __post_init__(self):
        if self.architectures is None:
            self.architectures = ["VortexEmbedV4_5"]


class VortexEmbedV4_5:
    """Vortex-Embed v4.5 — Native 4-Bit Sentence Embedding Model.

    Features:
      - 4.72 MB RAM Footprint (Zero FP32 matrix in RAM).
      - On-the-fly dequantization per batch.
      - Matryoshka Representation Learning (truncation to 256, 128, 64 dims).
    """

    def __init__(
        self,
        packed: np.ndarray,
        scales: np.ndarray,
        zeros: np.ndarray,
        tokenizer_data: Union[str, Path],
        config: Union[dict, VortexEmbedConfigV4],
        *,
        matryoshka_dim: Optional[int] = None,
    ) -> None:
        self.packed = np.asarray(packed, dtype=np.uint8)
        self.scales = np.asarray(scales, dtype=np.float16)
        self.zeros = np.asarray(zeros, dtype=np.float16)
        self.tokenizer_data = str(tokenizer_data)
        self.config = config if isinstance(config, VortexEmbedConfigV4) else VortexEmbedConfigV4(
            **{k: v for k, v in config.items() if k in VortexEmbedConfigV4.__dataclass_fields__}
        )
        self.vocab_size = int(self.config.vocab_size)
        self.dim = int(self.config.embedding_dim)
        self.block_size = int(self.config.block_size)
        self.num_blocks = int(self.config.num_blocks)
        self.sif_a = float(self.config.sif_a)
        self.sif_pc = float(self.config.sif_pc)
        self.pc_k = int(self.config.pc_k)
        self.matryoshka_dim = matryoshka_dim or self.config.matryoshka_dim

        self._tokenizer: Optional[Tokenizer] = None
        self._sif_weights: Optional[np.ndarray] = None
        self._pc_directions: Optional[np.ndarray] = None
        self.DEFAULT_MAX_CHARS_PER_TEXT = 50_000

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            if Tokenizer is None:
                raise RuntimeError("tokenizers required: pip install tokenizers")
            self._tokenizer = Tokenizer.from_file(self.tokenizer_data)
        return self._tokenizer

    @property
    def model_size_mb(self) -> float:
        return (self.packed.nbytes + self.scales.nbytes + self.zeros.nbytes) / 1e6

    @classmethod
    def from_pretrained(
        cls,
        path_or_id: Union[str, Path],
        matryoshka_dim: Optional[int] = None,
        **overrides,
    ) -> "VortexEmbedV4_5":
        path = Path(path_or_id)
        if not path.is_dir():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(str(path_or_id)))
        tensors = load_file(str(path / "model.safetensors"))
        config = json.loads((path / "config.json").read_text())
        for k, v in overrides.items():
            if hasattr(cls, "__dataclass_fields__") and k in VortexEmbedConfigV4.__dataclass_fields__:
                config[k] = v
        return cls(
            packed=tensors["embedding_packed"],
            scales=tensors["embedding_scales"],
            zeros=tensors["embedding_zeros"],
            tokenizer_data=str(path / "tokenizer.json"),
            config=config,
            matryoshka_dim=matryoshka_dim,
        )

    def save_pretrained(self, path: Union[str, Path]) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "embedding_packed": self.packed,
                "embedding_scales": self.scales,
                "embedding_zeros": self.zeros,
            },
            str(out / "model.safetensors"),
        )
        (out / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2))  # ty:ignore[unresolved-attribute]
        if not (out / "tokenizer.json").exists():
            (out / "tokenizer.json").write_text(Path(self.tokenizer_data).read_text())

    def fit_idf(self, corpus_token_lists: Sequence[Sequence[int]]) -> "VortexEmbedV4_5":
        flat = (np.concatenate(corpus_token_lists) if corpus_token_lists else np.empty(0, dtype=np.int64))
        total = max(int(flat.size), 1)
        counts = np.bincount(flat, minlength=self.vocab_size).astype(np.float64)
        p = counts / total
        denom = self.sif_a + p
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(p > 0, self.sif_a / denom, 1.0)
        self._sif_weights = weights.astype(np.float32)
        return self

    def fit_pc(self, corpus_embeddings: np.ndarray, k: Optional[int] = None) -> "VortexEmbedV4_5":
        if k is None:
            k = self.pc_k
        if corpus_embeddings.size == 0 or k <= 0:
            return self
        x = corpus_embeddings.astype(np.float32)
        x = x - x.mean(axis=0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(x, full_matrices=False)
            pcs = vt[:k].astype(np.float32)
            pcs = pcs / (np.linalg.norm(pcs, axis=1, keepdims=True) + 1e-12)
            self._pc_directions = pcs
        except np.linalg.LinAlgError:
            self._pc_directions = None
        return self

    def _apply_pc(self, x: np.ndarray) -> np.ndarray:
        if self.sif_pc <= 0 or self._pc_directions is None:
            return x
        out = x
        for pc in self._pc_directions:
            proj = (out @ pc)[:, None] * pc[None, :]
            out = out - self.sif_pc * proj
        return out

    def _tokenize_batch(self, texts: Sequence[str]) -> List[List[int]]:
        encoded = self.tokenizer.encode_batch(list(texts))
        return [[tid for tid in item.ids if 0 <= int(tid) < self.vocab_size] for item in encoded]

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        normalize: bool = True,
        truncate_dim: Optional[int] = None,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        token_lists = self._tokenize_batch(list(texts))
        n = len(token_lists)
        flat = (np.concatenate(token_lists) if token_lists else np.empty(0, dtype=np.int64))

        if flat.size == 0:
            return np.zeros((n, self.dim), dtype=np.float32)

        unique_ids, inverse_indices = np.unique(flat, return_inverse=True)
        unique_embs = self._dequantize_ids_on_the_fly(unique_ids)
        token_embs = unique_embs[inverse_indices]

        if self._sif_weights is not None:
            w = self._sif_weights[flat].astype(np.float32)[:, None]
            token_embs = token_embs * w

        chunk_lens = np.array([len(ids) for ids in token_lists], dtype=np.int64)
        chunk_ends = np.cumsum(chunk_lens)
        boundaries = np.empty(n + 1, dtype=np.int64)
        boundaries[0] = 0
        boundaries[1:] = chunk_ends

        sums = np.add.reduceat(token_embs, boundaries[:-1], axis=0)

        if self._sif_weights is not None:
            w_full = self._sif_weights[flat].astype(np.float32)
            w_per_row = np.add.reduceat(w_full, boundaries[:-1])
            w_per_row = np.maximum(w_per_row, 1e-12)
        else:
            w_per_row = np.maximum(chunk_lens.astype(np.float32), 1.0)

        embeddings = sums / w_per_row[:, None]
        embeddings = self._apply_pc(embeddings)

        dim = truncate_dim if truncate_dim is not None else self.matryoshka_dim
        if dim is not None and 0 < dim < self.dim:
            embeddings = embeddings[:, :dim]

        if normalize and embeddings.shape[0] > 0:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            np.divide(embeddings, np.maximum(norms, 1e-12), out=embeddings)

        return embeddings

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        *,
        normalize: bool = True,
        truncate_dim: Optional[int] = None,
    ) -> np.ndarray:
        if isinstance(texts, str):
            return self.encode_batch([texts], normalize=normalize, truncate_dim=truncate_dim)[0]
        return self.encode_batch(list(texts), normalize=normalize, truncate_dim=truncate_dim)

    def _dequantize_ids_on_the_fly(self, token_ids: np.ndarray) -> np.ndarray:
        if token_ids.size == 0:
            return np.empty((0, self.dim), dtype=np.float32)

        p = self.packed[token_ids]
        s = self.scales[token_ids].astype(np.float32)[:, :, None]
        z = self.zeros[token_ids].astype(np.float32)[:, :, None]

        low = (p & 0x0F).astype(np.float32)
        high = ((p >> 4) & 0x0F).astype(np.float32)

        n = len(token_ids)
        padded = p.shape[1] * 2
        unpacked = np.empty((n, padded), dtype=np.float32)
        unpacked[:, 0::2] = low
        unpacked[:, 1::2] = high

        blocked = unpacked.reshape(n, self.num_blocks, self.block_size)
        out = (blocked * s + z).reshape(n, padded)
        return out[:, : self.dim]
