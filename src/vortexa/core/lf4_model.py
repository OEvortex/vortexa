"""
Vortex-Embed v3 — Pure sentence-similarity (RAG) model.

Built on VTXAI/Vortex-Embed-4.7M (4-bit LF4 weights, 256-dim, 29528 vocab).
No code-search tricks. Focus is on:
  1. Lossless 4-bit LF4 quantization (vs FP32 reference)
  2. Fast inference (CPU-friendly)
  3. General text similarity for RAG retrieval

Default pipeline (per text):
  1. Tokenize (HuggingFace fast tokenizer, same as v1)
  2. SIF IDF weighting on every token
  3. Sum tokens per text via torch.scatter_add_ (CPU)
  4. Divide by SIF-weighted count
  5. Remove top-`pc_k` principal components (fitted on corpus)
  6. L2-normalize

Search: cosine similarity, no extension bias, no path headers.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from safetensors.numpy import load_file, save_file

if TYPE_CHECKING:
    from tokenizers import Tokenizer

try:
    from tokenizers import Tokenizer as _Tokenizer
except Exception:  # pragma: no cover
    _Tokenizer: Any = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class VortexEmbedConfig:
    """Configuration for VortexEmbedV3 + retrieval hyperparameters."""
    # Architecture
    vocab_size: int = 29528
    embedding_dim: int = 256
    block_size: int = 32
    num_blocks: int = 8
    model_type: str = "vortex-embed"
    architectures: List[str] = field(default_factory=lambda: ["VortexEmbedV3"])
    quantization: str = "lf4"
    bits: int = 4
    # v3 retrieval knobs
    sif_a: float = 0.01
    sif_pc: float = 1.0
    pc_k: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> "VortexEmbedConfig":
        kw = {k: d[k] for k in d if k in cls.__dataclass_fields__}
        return cls(**kw)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class VortexEmbedV3:
    """Vortex-Embed v3 — pure sentence-similarity RAG model.

    Quantization format: 4-bit LF4 (per-block FP16 scale + zero).
    For lossless 4-bit experiments, subclass and override _dequantize_all.
    """

    def __init__(
        self,
        packed: np.ndarray,
        scales: np.ndarray,
        zeros: np.ndarray,
        tokenizer_data: Union[str, Path],
        config: Union[dict, VortexEmbedConfig],
        *,
        precompute: bool = True,
    ) -> None:
        self.packed = np.asarray(packed, dtype=np.uint8)
        self.scales = np.asarray(scales, dtype=np.float16)
        self.zeros = np.asarray(zeros, dtype=np.float16)
        self.tokenizer_data = str(tokenizer_data)
        self.config = config if isinstance(config, VortexEmbedConfig) else VortexEmbedConfig.from_dict(config)
        self.vocab_size = int(self.config.vocab_size)
        self.dim = int(self.config.embedding_dim)
        self.block_size = int(self.config.block_size)
        self.num_blocks = int(self.config.num_blocks)
        # v3 retrieval knobs
        self.sif_a = float(self.config.sif_a)
        self.sif_pc = float(self.config.sif_pc)
        self.pc_k = int(self.config.pc_k)
        # State
        self._tokenizer: Optional[Tokenizer] = None
        self._embedding_table: Optional[np.ndarray] = None
        self._sif_weights: Optional[np.ndarray] = None
        self._pc_directions: Optional[np.ndarray] = None
        self.cache_path: Optional[Path] = None
        if precompute:
            self._embedding_table = self._dequantize_all()
        # FP16 cached table was tested (experiment_0080) — saves 50% RAM
        # but adds an FP16→FP32 cast in the encode hot path that costs
        # ~20% throughput. We keep FP32 here; the on-disk LF4 is still
        # only 4.7 MB. To save RAM, users can downcast after load.

    def _dequantize_ids(self, token_ids: np.ndarray) -> np.ndarray:
        if self._embedding_table is not None:
            return self._embedding_table[token_ids]
        # Cold path
        return self._dequantize_all()[token_ids]

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            if _Tokenizer is None:
                raise RuntimeError("tokenizers required: pip install tokenizers")
            self._tokenizer = _Tokenizer.from_file(self.tokenizer_data)
        return self._tokenizer

    @property
    def embedding_table(self) -> np.ndarray:
        if self._embedding_table is None:
            self._embedding_table = self._dequantize_all()
        return self._embedding_table

    @property
    def model_size_mb(self) -> float:
        if self._embedding_table is not None:
            return self._embedding_table.nbytes / 1e6
        return (self.packed.nbytes + self.scales.nbytes + self.zeros.nbytes) / 1e6

    @property
    def on_disk_size_mb(self) -> float:
        return (self.packed.nbytes + self.scales.nbytes + self.zeros.nbytes) / 1e6

    # ---- (de)serialization ---------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        path_or_id: Union[str, Path],
        *,
        precompute: bool = True,
        cache_path: Optional[Union[str, Path]] = None,
        **overrides,
    ) -> "VortexEmbedV3":
        path = Path(path_or_id)
        if not path.is_dir():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(str(path_or_id)))
        tensors = load_file(str(path / "model.safetensors"))
        config = json.loads((path / "config.json").read_text())
        for k, v in overrides.items():
            if k in VortexEmbedConfig.__dataclass_fields__:
                config[k] = v
        obj = cls(
            packed=tensors["embedding_packed"],
            scales=tensors["embedding_scales"],
            zeros=tensors["embedding_zeros"],
            tokenizer_data=str(path / "tokenizer.json"),
            config=config,
            precompute=precompute,
        )
        if cache_path is not None:
            obj.cache_path = Path(cache_path)
        return obj

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
        (out / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2))
        if not (out / "tokenizer.json").exists():
            (out / "tokenizer.json").write_text(Path(self.tokenizer_data).read_text())

    # ---- LF4 dequantization (override point for quantization experiments) ---

    def _dequantize_all(self) -> np.ndarray:
        """LF4 dequantization: 4-bit packed + per-block FP16 scale + zero.

        This is the v1/v2 implementation. Override in subclasses to
        experiment with better quantization schemes (per-dim scales,
        residual storage, GPTQ-style optimal 4-bit, etc).
        """
        low = (self.packed & 0x0F).astype(np.float32)
        high = ((self.packed >> 4) & 0x0F).astype(np.float32)
        padded = self.packed.shape[1] * 2
        unpacked = np.empty((self.packed.shape[0], padded), dtype=np.float32)
        unpacked[:, 0::2] = low
        unpacked[:, 1::2] = high
        blocked = unpacked.reshape(self.packed.shape[0], self.num_blocks, self.block_size)
        scales = self.scales.astype(np.float32)[:, :, None]
        zeros = self.zeros.astype(np.float32)[:, :, None]
        out = (blocked * scales + zeros).reshape(self.packed.shape[0], padded)
        return out[:, : self.dim]

    # ---- SIF + PC fitting ----------------------------------------------

    def fit_idf(self, corpus_token_lists: Sequence[Sequence[int]]) -> "VortexEmbedV3":
        flat = (np.concatenate(corpus_token_lists)
                if corpus_token_lists else np.empty(0, dtype=np.int64))
        total = max(int(flat.size), 1)
        counts = np.bincount(flat, minlength=self.vocab_size).astype(np.float64)
        p = counts / total
        denom = self.sif_a + p
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(p > 0, self.sif_a / denom, 1.0)
        self._sif_weights = weights.astype(np.float32)
        return self

    def fit_pc(self, corpus_embeddings: np.ndarray, k: Optional[int] = None) -> "VortexEmbedV3":
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

    # ---- tokenization ----------------------------------------------------

    DEFAULT_MAX_CHARS_PER_TEXT = 50_000
    DEFAULT_MAX_TOKENS_PER_TEXT = 4096
    DEFAULT_MAX_TOKENS_PER_BATCH = 262_144

    def _tokenize_batch(self, texts: Sequence[str]) -> List[List[int]]:
        encoded = self.tokenizer.encode_batch(list(texts))
        return [
            [tid for tid in item.ids if 0 <= int(tid) < self.vocab_size]
            for item in encoded
        ]

    def _cap_inputs(self, texts: Sequence[str]) -> List[str]:
        cap = self.DEFAULT_MAX_CHARS_PER_TEXT
        if cap <= 0:
            return list(texts)
        out = []
        for t in texts:
            if len(t) <= cap:
                out.append(t)
            else:
                half = cap // 2
                out.append(t[:half] + t[-(cap - half):])
        return out

    def _cap_token_lists(self, token_lists: List[List[int]]) -> List[List[int]]:
        cap = self.DEFAULT_MAX_TOKENS_PER_TEXT
        if cap <= 0:
            return token_lists
        out = []
        for ids in token_lists:
            if len(ids) <= cap:
                out.append(ids)
            else:
                half = cap // 2
                out.append(ids[:half] + ids[-(cap - half):])
        return out

    @staticmethod
    def _normalize_inplace(x: np.ndarray) -> None:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        np.divide(x, np.maximum(norms, 1e-12), out=x)

    # ---- core encode -----------------------------------------------------

    def _encode_subbatch(
        self, token_lists: Sequence[Sequence[int]], *, normalize: bool
    ) -> np.ndarray:
        n = len(token_lists)
        flat = (np.concatenate(token_lists)
                if token_lists else np.empty(0, dtype=np.int64))
        if flat.size == 0:
            return np.zeros((n, self.dim), dtype=np.float32)

        token_embs = self._dequantize_ids(flat)
        if token_embs.dtype != np.float32:
            token_embs = token_embs.astype(np.float32)
        if self._sif_weights is not None:
            w = self._sif_weights[flat].astype(np.float32)[:, None]
            token_embs = token_embs * w

        # Segment-sum via torch.index_add_ on CPU. ~30× faster than
        # np.add.reduceat for this size, because ATen's scatter-add is
        # highly tuned and reduceat has per-call overhead. (experiment_0101)
        import torch
        ro = torch.from_numpy(
            np.repeat(np.arange(n, dtype=np.int64),
                      [len(ids) for ids in token_lists])
        )
        em = torch.from_numpy(np.ascontiguousarray(token_embs))
        sums = torch.zeros((n, self.dim), dtype=torch.float32)
        sums.index_add_(0, ro, em)
        sums = sums.numpy()

        if self._sif_weights is not None:
            w_full = self._sif_weights[flat].astype(np.float32)
            chunk_lens = np.array([len(ids) for ids in token_lists], dtype=np.int64)
            chunk_ends = np.cumsum(chunk_lens)
            boundaries = np.empty(n + 1, dtype=np.int64)
            boundaries[0] = 0
            boundaries[1:] = chunk_ends
            w_per_row = np.add.reduceat(w_full, boundaries[:-1])
            w_per_row = np.maximum(w_per_row, 1e-12)
        else:
            chunk_lens = np.array([len(ids) for ids in token_lists], dtype=np.int64)
            w_per_row = np.maximum(chunk_lens.astype(np.float32), 1.0)

        embeddings = sums / w_per_row[:, None]
        embeddings = self._apply_pc(embeddings)
        if normalize:
            self._normalize_inplace(embeddings)
        return embeddings

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        normalize: bool = True,
        max_tokens_per_text: Optional[int] = None,
        max_tokens_per_batch: Optional[int] = None,
        max_chars_per_text: Optional[int] = None,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        capped = self._cap_inputs(list(texts))
        token_lists = self._tokenize_batch(capped)
        token_lists = self._cap_token_lists(token_lists)
        cap_t = (self.DEFAULT_MAX_TOKENS_PER_TEXT
                 if max_tokens_per_text is None else int(max_tokens_per_text))
        cap_b = (self.DEFAULT_MAX_TOKENS_PER_BATCH
                 if max_tokens_per_batch is None else int(max_tokens_per_batch))
        _ = cap_t

        total_tokens = sum(len(ids) for ids in token_lists)
        if total_tokens == 0:
            return np.zeros((len(texts), self.dim), dtype=np.float32)

        if total_tokens <= cap_b or len(texts) <= 1:
            return self._encode_subbatch(token_lists, normalize=normalize)

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        sub: List[List[int]] = []
        sub_tokens = 0
        sub_start = 0
        for i, ids in enumerate(token_lists):
            if sub and (sub_tokens + len(ids) > cap_b):
                out[sub_start:i] = self._encode_subbatch(
                    token_lists[sub_start:i], normalize=False
                )
                sub_start = i
                sub = [ids]
                sub_tokens = len(ids)
            else:
                sub.append(ids)
                sub_tokens += len(ids)
        if sub:
            out[sub_start:] = self._encode_subbatch(
                token_lists[sub_start:], normalize=False
            )
        if normalize:
            self._normalize_inplace(out)
        return out

    def encode(self, texts: Union[str, Sequence[str]], *, normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            return self.encode_batch([texts], normalize=normalize)[0]
        return self.encode_batch(list(texts), normalize=normalize)

    def encode_batch_cached(
        self,
        texts: Sequence[str],
        *,
        normalize: bool = True,
        cache_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        if cache_path is None and self.cache_path is not None:
            cache_path = self.cache_path
        if cache_path is None:
            return self.encode_batch(texts, normalize=normalize)
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        emb_path = cache_path.with_suffix(".npy")
        meta_path = cache_path.with_suffix(".json")
        import hashlib
        h = hashlib.sha1()
        h.update(f"{self.dim}|v3|{len(texts)}|".encode())
        for t in texts:
            h.update(t.encode("utf-8", errors="replace"))
            h.update(b"\x00")
        fp = h.hexdigest()
        if meta_path.exists() and emb_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                if meta.get("fingerprint") == fp and meta.get("dim") == self.dim:
                    cached = np.load(emb_path, mmap_mode=None)
                    if cached.shape == (len(texts), self.dim):
                        return cached.copy() if normalize else cached
            except Exception:
                pass
        emb = self.encode_batch(texts, normalize=normalize)
        np.save(emb_path, emb.astype(np.float32))
        meta_path.write_text(json.dumps({"fingerprint": fp, "dim": self.dim, "n": len(texts)}))
        return emb

    # ---- search ---------------------------------------------------------

    def search(
        self,
        queries: np.ndarray,
        index: np.ndarray,
        top_k: int = 10,
        *,
        index_normalized: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cosine search. Returns ``(scores, indices)`` of shape ``(Q, top_k)``."""
        queries = np.asarray(queries, dtype=np.float32)
        index = np.asarray(index, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]
        if not index_normalized:
            index = index.copy()
            self._normalize_inplace(index)
        qn = queries.copy()
        self._normalize_inplace(qn)

        scores = qn @ index.T
        n_docs = scores.shape[1]
        k = min(int(top_k), n_docs)
        if k <= 0:
            return (np.empty((queries.shape[0], 0), dtype=np.float32),
                    np.empty((queries.shape[0], 0), dtype=np.int64))
        if k == n_docs:
            idx = np.argsort(-scores, axis=1)[:, :k]
        else:
            part = np.argpartition(-scores, kth=k, axis=1)[:, :k]
            ps = np.take_along_axis(scores, part, axis=1)
            order = np.argsort(-ps, axis=1)
            idx = np.take_along_axis(part, order, axis=1)
        ordered_scores = np.take_along_axis(scores, idx, axis=1)
        return (ordered_scores.astype(np.float32, copy=False),
                idx.astype(np.int64, copy=False))