"""Native 2-bit (LF2) Vortex embedding models.

The LF2 checkpoints published by VTXAI use the format documented in their
``lf2_native.py`` model card:

* four 2-bit values are packed into each ``uint8`` code byte;
* each block has a ``uint8`` scale and minimum, double-quantized using the
  three global values in ``config.json``;
* dequantization is performed only for token rows used by a batch.

This module intentionally keeps the checkpoint representation integer-native;
it does not materialize a full float embedding table at load time.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

try:
    from tokenizers import Tokenizer
except ImportError:  # pragma: no cover - dependency is declared by the project
    Tokenizer = None  # type: ignore[assignment,misc]

_VALUES_PER_BYTE = 4


@dataclass
class LF2Config:
    """Configuration for a Vortex LF2 checkpoint."""

    vocab_size: int = 29528
    embedding_dim: int = 256
    block_size: int = 32
    num_blocks: int = 8
    global_min: float = 0.0
    global_max: float = 0.0
    global_scale_max: float = 1.0
    matryoshka_dim: int | None = None
    quantization: str = "lf2"
    bits: int = 2

    @classmethod
    def from_dict(cls, values: dict) -> LF2Config:
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in fields})

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "global_min": self.global_min,
            "global_max": self.global_max,
            "global_scale_max": self.global_scale_max,
            "matryoshka_dim": self.matryoshka_dim,
            "quantization": self.quantization,
            "bits": self.bits,
        }


class VortexEmbedLF2:
    """A native 2-bit static embedding model with the LF4 wrapper API."""

    def __init__(
        self,
        packed: np.ndarray,
        scale_u8: np.ndarray,
        min_u8: np.ndarray,
        tokenizer_data: str | Path,
        config: dict | LF2Config,
        *,
        matryoshka_dim: int | None = None,
    ) -> None:
        self.packed = np.asarray(packed, dtype=np.uint8)
        self.scale_u8 = np.asarray(scale_u8, dtype=np.uint8)
        self.min_u8 = np.asarray(min_u8, dtype=np.uint8)
        self.tokenizer_data = str(tokenizer_data)
        self.config = (
            config if isinstance(config, LF2Config) else LF2Config.from_dict(config)
        )
        self.vocab_size = int(self.config.vocab_size)
        self.dim = int(self.config.embedding_dim)
        self.block_size = int(self.config.block_size)
        self.num_blocks = int(self.config.num_blocks)
        self._validate_tensors()
        self.matryoshka_dim = matryoshka_dim or self.config.matryoshka_dim
        self.sif_a = 0.05
        self.sif_pc = 1.0
        self.pc_k = 1
        self._tokenizer: Tokenizer | None = None
        self._sif_weights: np.ndarray | None = None
        self._pc_directions: np.ndarray | None = None
        self.DEFAULT_MAX_CHARS_PER_TEXT = 50_000

    def _validate_tensors(self) -> None:
        expected_packed = (
            self.vocab_size,
            self.config.num_blocks * self.config.block_size // _VALUES_PER_BYTE,
        )
        expected_meta = (self.vocab_size, self.config.num_blocks)
        if self.config.embedding_dim != self.config.block_size * self.config.num_blocks:
            raise ValueError(
                "LF2 embedding_dim must equal block_size * num_blocks; "
                f"got {self.config.embedding_dim} and "
                f"{self.config.block_size} * {self.config.num_blocks}"
            )
        if self.config.block_size % _VALUES_PER_BYTE:
            raise ValueError("LF2 block_size must be divisible by 4")
        if self.packed.shape != expected_packed:
            raise ValueError(
                f"LF2 packed shape {self.packed.shape} does not match {expected_packed}"
            )
        if self.scale_u8.shape != expected_meta or self.min_u8.shape != expected_meta:
            raise ValueError(
                "LF2 scale/min tensors must both have shape "
                f"{expected_meta}; got {self.scale_u8.shape} and {self.min_u8.shape}"
            )

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            if Tokenizer is None:
                raise RuntimeError("tokenizers required: pip install tokenizers")
            self._tokenizer = Tokenizer.from_file(self.tokenizer_data)
        return self._tokenizer

    @property
    def model_size_mb(self) -> float:
        """Integer checkpoint bytes plus the three float config scalars."""
        return (self.packed.nbytes + self.scale_u8.nbytes + self.min_u8.nbytes + 12) / 1e6

    @property
    def on_disk_size_mb(self) -> float:
        return self.model_size_mb

    @property
    def int_bytes(self) -> int:
        return int(self.packed.nbytes + self.scale_u8.nbytes + self.min_u8.nbytes)

    @classmethod
    def from_pretrained(
        cls,
        path_or_id: str | Path,
        matryoshka_dim: int | None = None,
    ) -> VortexEmbedLF2:
        """Load a local LF2 directory or a Hugging Face model ID."""
        path = Path(path_or_id)
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            path = Path(
                snapshot_download(
                    str(path_or_id),
                    allow_patterns=["config.json", "model.safetensors", "tokenizer.json"],
                )
            )
        tensors = load_file(str(path / "model.safetensors"))
        config = json.loads((path / "config.json").read_text())
        return cls(
            packed=tensors["embedding_packed"],
            scale_u8=tensors["embedding_scale_u8"],
            min_u8=tensors["embedding_min_u8"],
            tokenizer_data=str(path / "tokenizer.json"),
            config=config,
            matryoshka_dim=matryoshka_dim,
        )

    def save_pretrained(self, path: str | Path) -> None:
        """Save the native integer checkpoint and its tokenizer/config."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "embedding_packed": self.packed,
                "embedding_scale_u8": self.scale_u8,
                "embedding_min_u8": self.min_u8,
            },
            str(out / "model.safetensors"),
        )
        (out / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2))
        if not (out / "tokenizer.json").exists():
            (out / "tokenizer.json").write_text(Path(self.tokenizer_data).read_text())

    def _dequantize(self, token_ids: np.ndarray, out_dim: int | None = None) -> np.ndarray:
        if token_ids.size == 0:
            return np.empty((0, out_dim or self.dim), dtype=np.float32)
        token_ids = np.asarray(token_ids, dtype=np.int64)
        dim = out_dim or self.dim
        blocks_needed = min(self.num_blocks, (dim + self.block_size - 1) // self.block_size)
        bytes_per_block = self.block_size // _VALUES_PER_BYTE
        packed = self.packed[token_ids, : blocks_needed * bytes_per_block]
        packed = packed.reshape(len(token_ids), blocks_needed, bytes_per_block)
        codes = np.empty((len(token_ids), blocks_needed, self.block_size), dtype=np.uint8)
        codes[:, :, 0::4] = packed & 0x03
        codes[:, :, 1::4] = (packed >> 2) & 0x03
        codes[:, :, 2::4] = (packed >> 4) & 0x03
        codes[:, :, 3::4] = (packed >> 6) & 0x03

        step = (
            self.config.global_scale_max
            * self.scale_u8[token_ids, :blocks_needed].astype(np.float32)
            / 255.0
        )
        span = self.config.global_max - self.config.global_min
        if span <= 0:
            span = 1.0
        minimum = self.config.global_min + (
            self.min_u8[token_ids, :blocks_needed].astype(np.float32) / 255.0 * span
        )
        values = codes.astype(np.float32)
        values *= step[:, :, None]
        values += minimum[:, :, None]
        return values.reshape(len(token_ids), -1)[:, :dim]

    def _tokenize_batch(self, texts: Sequence[str]) -> list[list[int]]:
        encoded = self.tokenizer.encode_batch(list(texts))
        return [
            [int(token_id) for token_id in item.ids if 0 <= int(token_id) < self.vocab_size]
            for item in encoded
        ]

    def fit_idf(self, corpus_token_lists: Sequence[Sequence[int]]) -> VortexEmbedLF2:
        flat = (
            np.concatenate(corpus_token_lists)
            if corpus_token_lists
            else np.empty(0, dtype=np.int64)
        )
        total = max(int(flat.size), 1)
        counts = np.bincount(flat, minlength=self.vocab_size).astype(np.float64)
        probabilities = counts / total
        weights = np.where(
            probabilities > 0,
            self.sif_a / (self.sif_a + probabilities),
            1.0,
        )
        self._sif_weights = weights.astype(np.float32)
        return self

    def fit_pc(self, corpus_embeddings: np.ndarray, k: int | None = None) -> VortexEmbedLF2:
        if k is None:
            k = self.pc_k
        if corpus_embeddings.size == 0 or k <= 0:
            return self
        values = corpus_embeddings.astype(np.float32)
        values = values - values.mean(axis=0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(values, full_matrices=False)
            directions = vt[:k].astype(np.float32)
            self._pc_directions = directions / (
                np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
            )
        except np.linalg.LinAlgError:
            self._pc_directions = None
        return self

    def _apply_pc(self, values: np.ndarray) -> np.ndarray:
        if self.sif_pc <= 0 or self._pc_directions is None:
            return values
        for direction in self._pc_directions:
            values = values - self.sif_pc * (values @ direction)[:, None] * direction[None, :]
        return values

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        normalize: bool = True,
        truncate_dim: int | None = None,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        token_lists = self._tokenize_batch(list(texts))
        n = len(token_lists)
        flat = (
            np.concatenate(token_lists)
            if token_lists
            else np.empty(0, dtype=np.int64)
        )
        if flat.size == 0:
            return np.zeros((n, self.dim), dtype=np.float32)

        unique_ids, inverse = np.unique(flat, return_inverse=True)
        token_embeddings = self._dequantize(unique_ids)[inverse]
        if self._sif_weights is not None:
            token_embeddings *= self._sif_weights[flat, None]
        lengths = np.asarray([len(ids) for ids in token_lists], dtype=np.int64)
        boundaries = np.concatenate(([0], np.cumsum(lengths)))
        sums = np.add.reduceat(token_embeddings, boundaries[:-1], axis=0)
        if self._sif_weights is not None:
            weights = self._sif_weights[flat].astype(np.float32)
            denominators = np.maximum(np.add.reduceat(weights, boundaries[:-1]), 1e-12)
        else:
            denominators = np.maximum(lengths.astype(np.float32), 1.0)
        embeddings = sums / denominators[:, None]
        embeddings[lengths == 0] = 0.0
        embeddings = self._apply_pc(embeddings)
        dim = truncate_dim if truncate_dim is not None else self.matryoshka_dim
        if dim is not None and 0 < dim < self.dim:
            embeddings = embeddings[:, :dim]
        if normalize and embeddings.shape[0]:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            np.divide(embeddings, np.maximum(norms, 1e-12), out=embeddings)
        return embeddings.astype(np.float32, copy=False)

    def encode(
        self,
        texts: str | Sequence[str],
        *,
        normalize: bool = True,
        truncate_dim: int | None = None,
    ) -> np.ndarray:
        if isinstance(texts, str):
            return self.encode_batch([texts], normalize=normalize, truncate_dim=truncate_dim)[0]
        return self.encode_batch(list(texts), normalize=normalize, truncate_dim=truncate_dim)
