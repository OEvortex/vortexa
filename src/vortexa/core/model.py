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
class GateParams:
    """Position-aware attention pooling parameters (VTX-JEV checkpoints).

    The pooled vector is a softmax attention weighted sum of token rows::

        hidden = tanh(rows @ w_u + w_p[position] + b)
        score  = hidden @ v
        pooled = softmax(score) @ rows
    """

    w_u: np.ndarray
    w_p: np.ndarray
    b: np.ndarray
    v: np.ndarray
    max_position: int

    @classmethod
    def load(cls, path: str | Path) -> GateParams | None:
        """Load gate parameters, returning None when the file is absent."""
        file = Path(path)
        if not file.is_file():
            return None
        try:
            data = np.load(file)
        except (OSError, ValueError):
            return None
        required = {"W_u", "W_p", "b", "v", "max_position"}
        if not required.issubset(data.files):
            return None
        return cls(
            w_u=np.asarray(data["W_u"], dtype=np.float32),
            w_p=np.asarray(data["W_p"], dtype=np.float32),
            b=np.asarray(data["b"], dtype=np.float32),
            v=np.asarray(data["v"], dtype=np.float32),
            max_position=int(data["max_position"]),
        )


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
        gate: GateParams | None = None,
        pooling: str | None = None,
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
        self.gate = gate
        if pooling is None:
            pooling = "gate" if gate is not None else "sif"
        if pooling not in ("sif", "gate", "mean"):
            raise ValueError(f"unsupported LF2 pooling mode: {pooling}")
        if pooling == "gate" and gate is None:
            raise ValueError("gate pooling requires gate parameters")
        self.pooling = pooling
        self.pooling_max_position = int(self.gate.max_position) if self.gate is not None else 0

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
        *,
        pooling: str | None = None,
    ) -> VortexEmbedLF2:
        """Load a local LF2 directory or a Hugging Face model ID.

        Supports both published layouts:

        * ``config.json`` + ``model.safetensors`` (VTX embed checkpoints)
        * ``config_lf2.json`` + ``model_lf2.safetensors`` (VTX-JEV checkpoints,
          which also ship ``gate_params.npz`` for position-aware pooling)
        """
        path = Path(path_or_id)
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            try:
                path = Path(
                    snapshot_download(
                        str(path_or_id),
                        allow_patterns=[
                            "config.json",
                            "config_lf2.json",
                            "model.safetensors",
                            "model_lf2.safetensors",
                            "tokenizer.json",
                            "gate_params.npz",
                        ],
                        local_files_only=True,
                    )
                )
            except Exception:
                path = Path(
                    snapshot_download(
                        str(path_or_id),
                        allow_patterns=[
                            "config.json",
                            "config_lf2.json",
                            "model.safetensors",
                            "model_lf2.safetensors",
                            "tokenizer.json",
                            "gate_params.npz",
                        ],
                    )
                )

        jev_config = path / "config_lf2.json"
        jev_weights = path / "model_lf2.safetensors"
        if jev_config.is_file() and jev_weights.is_file():
            config_path, weights_path = jev_config, jev_weights
            gate = GateParams.load(path / "gate_params.npz")
        else:
            config_path, weights_path = path / "config.json", path / "model.safetensors"
            gate = None
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                f"LF2 checkpoint not found in {path}: expected {config_path.name} "
                f"and {weights_path.name}"
            )

        tensors = load_file(str(weights_path))
        config = json.loads(config_path.read_text())
        return cls(
            packed=tensors["embedding_packed"],
            scale_u8=tensors["embedding_scale_u8"],
            min_u8=tensors["embedding_min_u8"],
            tokenizer_data=str(path / "tokenizer.json"),
            config=config,
            matryoshka_dim=matryoshka_dim,
            gate=gate,
            pooling=pooling,
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

    def _pool_gate(self, token_embeddings: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
        """Position-aware attention pooling over token rows.

        Token rows are L2 normalized first, matching the VTX-JEV reference
        implementation, then a softmax gate over positions produces the pooled
        vector.
        """
        assert self.gate is not None
        rows = token_embeddings.astype(np.float32, copy=True)
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        rows /= np.maximum(norms, 1e-12)

        starts = boundaries[:-1]
        sizes = np.diff(boundaries)
        nonempty = sizes > 0

        # Positions restart per text and clamp at the gate's trained window.
        positions = np.arange(rows.shape[0], dtype=np.int64) - np.repeat(starts, sizes)
        positions = np.minimum(positions, self.gate.max_position - 1)
        hidden = np.tanh(rows @ self.gate.w_u + self.gate.w_p[positions] + self.gate.b)
        scores = (hidden @ self.gate.v).astype(np.float64)

        live = starts[nonempty]
        pooled = np.zeros((len(starts), self.dim), dtype=np.float32)
        if live.size == 0:
            return pooled

        seg_max = np.zeros(len(starts), dtype=np.float64)
        seg_max[nonempty] = np.maximum.reduceat(scores, live)
        weights = np.exp(scores - np.repeat(seg_max, sizes))
        totals = np.zeros(len(starts), dtype=np.float64)
        totals[nonempty] = np.add.reduceat(weights, live)
        weights /= np.maximum(np.repeat(totals, sizes), 1e-12)
        pooled[nonempty] = np.add.reduceat(weights[:, None] * rows, live, axis=0)
        return pooled.astype(np.float32, copy=False)

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
        lengths = np.asarray([len(ids) for ids in token_lists], dtype=np.int64)
        boundaries = np.concatenate(([0], np.cumsum(lengths)))

        if self.pooling == "gate":
            embeddings = self._pool_gate(token_embeddings, boundaries)
        else:
            if self.pooling == "mean":
                token_embeddings = token_embeddings.copy()
            elif self._sif_weights is not None:
                token_embeddings = token_embeddings * self._sif_weights[flat, None]
            sums = np.add.reduceat(token_embeddings, boundaries[:-1], axis=0)
            if self.pooling == "sif" and self._sif_weights is not None:
                weights = self._sif_weights[flat].astype(np.float32)
                denominators = np.maximum(np.add.reduceat(weights, boundaries[:-1]), 1e-12)
            else:
                denominators = np.maximum(lengths.astype(np.float32), 1.0)
            embeddings = sums / denominators[:, None]

        embeddings[lengths == 0] = 0.0
        if self.pooling != "gate":
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


# ===========================================================================
# 4-bit model (merged)
# ===========================================================================

"""Vortex-Embed v4.5 — Native 4-bit Sentence Embedding Model.

Adapted from VTXAI/vtx-embed-7M (HuggingFace).
On-the-fly dequantization per batch — no precomputed embedding table.
Matryoshka representation learning support.
"""

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

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "model_type": self.model_type,
            "architectures": self.architectures,
            "quantization": self.quantization,
            "bits": self.bits,
            "sif_a": self.sif_a,
            "sif_pc": self.sif_pc,
            "pc_k": self.pc_k,
            "matryoshka_dim": self.matryoshka_dim,
        }


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
            try:
                path = Path(snapshot_download(str(path_or_id), local_files_only=True))
            except Exception:
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


# ===========================================================================
# Embedder wrapper (merged)
# ===========================================================================

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
    "jev3": "VTXAI/VTX-JEV-3",
    "jev2": "VTXAI/VTX-JEV-2",
}

_HUB_PATTERNS = [
    "config.json",
    "config_lf2.json",
    "model.safetensors",
    "model_lf2.safetensors",
    "tokenizer.json",
    "gate_params.npz",
]


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
        pooling: Optional[str] = None,
    ) -> None:
        self._model_id = _MODEL_ALIASES.get(model_id, model_id)
        self.sif_a = sif_a
        self.sif_pc = sif_pc
        self.pc_k = pc_k
        self.matryoshka_dim = matryoshka_dim
        self.pooling = pooling
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

            try:
                model_path = Path(snapshot_download(self._model_id, allow_patterns=_HUB_PATTERNS, local_files_only=True))
            except Exception:
                model_path = Path(snapshot_download(self._model_id, allow_patterns=_HUB_PATTERNS))

        # VTX-JEV checkpoints ship the 2-bit weights under a dedicated LF2
        # filename with their own config, alongside gate params for pooling.
        jev_lf2 = model_path / "model_lf2.safetensors"
        if jev_lf2.is_file() and (model_path / "config_lf2.json").is_file():

            return VortexEmbedLF2.from_pretrained(
                model_path, matryoshka_dim=self.matryoshka_dim, pooling=self.pooling
            )

        config_path = model_path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"model config not found: {config_path}")
        config = json.loads(config_path.read_text())
        quantization = str(config.get("quantization", "")).lower()
        if quantization == "lf2":

            return VortexEmbedLF2.from_pretrained(
                model_path, matryoshka_dim=self.matryoshka_dim, pooling=self.pooling
            )


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
