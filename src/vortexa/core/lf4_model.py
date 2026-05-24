"""LF4 Static Embedding Model - Native 4-bit quantized sentence embeddings.

Usage:
    from lf4_model import LF4StaticEmbedding
    model = LF4StaticEmbedding.from_pretrained("VTXAI/Vortex-Embed-4.7M")
    embeddings = model.encode(["find python json parser", "weather API tool"])

    # Search
    scores, indices = model.search(query_emb, index_emb, top_k=10)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class LF4StaticEmbedding:
    """Native LF4 4-bit static embedding model.

    Weights are stored as packed 4-bit integers with per-block FP16 scales/zeros.
    Total model size: ~3.5 MB (vs 29 MB FP32).
    """

    def __init__(self, packed, scales, zeros, tokenizer_data, config):
        self.packed = packed  # uint8 (vocab, dim/2)
        self.scales = scales  # float16 (vocab, num_blocks)
        self.zeros = zeros  # float16 (vocab, num_blocks)
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.dim = config["embedding_dim"]
        self.block_size = config["block_size"]
        self._tokenizer_data = tokenizer_data
        self._tokenizer = None

        self._embedding_table = self._dequantize_all()

    def _dequantize_all(self) -> np.ndarray:
        """Dequantize full embedding table to FP32 for fast token lookup."""
        N = self.packed.shape[0]
        D = self.dim
        B = self.block_size

        low = (self.packed & 0x0F).astype(np.float32)
        high = ((self.packed >> 4) & 0x0F).astype(np.float32)
        D_padded = self.packed.shape[1] * 2

        unpacked = np.empty((N, D_padded), dtype=np.float32)
        unpacked[:, 0::2] = low
        unpacked[:, 1::2] = high

        num_blocks = D_padded // B
        blocked = unpacked.reshape(N, num_blocks, B)
        s = self.scales.astype(np.float32)[:, :, None]
        z = self.zeros.astype(np.float32)[:, :, None]

        return (blocked * s + z).reshape(N, D_padded)[:, :D]

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from tokenizers import Tokenizer
            self._tokenizer = Tokenizer.from_str(self._tokenizer_data)
        return self._tokenizer

    @classmethod
    def from_pretrained(cls, path_or_id: str) -> LF4StaticEmbedding:
        """Load model from local path or HuggingFace Hub."""
        p = Path(path_or_id)
        if p.is_dir():
            model_path = str(p / "model.safetensors")
            config_path = p / "config.json"
            tok_path = str(p / "tokenizer.json")
        else:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(path_or_id, "model.safetensors")
            config_path = Path(hf_hub_download(path_or_id, "config.json"))
            tok_path = hf_hub_download(path_or_id, "tokenizer.json")

        from safetensors.numpy import load_file
        tensors = load_file(model_path)
        config = json.loads(config_path.read_text())

        return cls(
            packed=tensors["embedding_packed"],
            scales=tensors["embedding_scales"],
            zeros=tensors["embedding_zeros"],
            tokenizer_data=Path(tok_path).read_text(encoding="utf-8"),
            config=config,
        )

    def encode(self, texts: str | list[str], normalize: bool = True) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: single string or list of strings
            normalize: L2-normalize output embeddings (default True for cosine sim)

        Returns:
            np.ndarray of shape (N, dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = np.zeros((len(texts), self.dim), dtype=np.float32)

        for i, text in enumerate(texts):
            encoded = self.tokenizer.encode(text)
            token_ids = encoded.ids

            valid_ids = [tid for tid in token_ids if 0 <= tid < self.vocab_size]
            if valid_ids:
                token_embs = self._embedding_table[valid_ids]
                embeddings[i] = token_embs.mean(axis=0)

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            embeddings = embeddings / norms

        return embeddings

    def search(
        self,
        queries: np.ndarray,
        index: np.ndarray,
        top_k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cosine similarity search.

        Args:
            queries: (Q, D) query embeddings
            index: (N, D) document embeddings
            top_k: number of results

        Returns:
            (scores, indices) arrays
        """
        queries = np.asarray(queries, dtype=np.float32)
        index = np.asarray(index, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]

        qn = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
        dn = index / (np.linalg.norm(index, axis=1, keepdims=True) + 1e-8)

        scores = qn @ dn.T

        if top_k >= scores.shape[1]:
            idx = np.argsort(-scores, axis=1)
            return np.take_along_axis(scores, idx, 1), idx

        idx = np.argpartition(-scores, top_k, axis=1)[:, :top_k]
        s = np.take_along_axis(scores, idx, 1)
        order = np.argsort(-s, axis=1)
        return np.take_along_axis(s, order, 1), np.take_along_axis(idx, order, 1)

    @property
    def model_size_mb(self) -> float:
        return (self.packed.nbytes + self.scales.nbytes + self.zeros.nbytes) / 1e6

    def __repr__(self):
        return (
            f"LF4StaticEmbedding(vocab={self.vocab_size}, dim={self.dim}, "
            f"bits=4, size={self.model_size_mb:.1f}MB, "
            f"block_size={self.block_size})"
        )
