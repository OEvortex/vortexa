"""Standalone inference engine for VTXAI Vortex-Embed models.

Provides a sentence-transformers-style API for encoding arbitrary text
into dense vector embeddings using Vortex-Embed v4.x models.

Quick start:
    from vortexa.core.inference import VortexEmbedInference

    # Load a model (default: mini)
    model = VortexEmbedInference("mini")

    # Encode text
    vec = model.encode("hello world")              # shape: (1, 256)
    vecs = model.encode(["hello", "world"])        # shape: (N, 256)

    # Control output dimension (Matryoshka truncation)
    vec = model.encode("hello world", dim=128)     # shape: (1, 128)

    # Get the effective dimension
    print(model.dim)                               # 256
    print(model.get_embedding_dimension())         # 256

    # Use the nano model
    nano = VortexEmbedInference("nano")
    print(nano.dim)                                # nano's native dim

Convenience function (stateless):
    from vortexa.core.inference import embed

    vec = embed("hello world", model="nano", dim=64)
"""
from __future__ import annotations

from typing import List, Union

import numpy as np
import numpy.typing as npt

_MODEL_ALIASES = {
    "mini": "VTXAI/vtx-embed-7M",
    "nano": "VTXAI/vtx-embed-1M",
}


def _resolve_model_id(model: str) -> str:
    return _MODEL_ALIASES.get(model, model)


class VortexEmbedInference:
    """A sentence-transformers-style inference engine for Vortex-Embed models.

    Supports on-the-fly LF4 4-bit dequantization, SIF+PC weighting, and
    Matryoshka representation learning for dimension truncation.

    Args:
        model: Model ID or alias (``mini``, ``nano``, or any
            HuggingFace model name). Default: ``mini``.
        dim: If set, truncate all output embeddings to this number of
            dimensions via Matryoshka truncation. Can be overridden per-call
            in ``encode()``.

    Example:
        >>> from vortexa.core.inference import VortexEmbedInference
        >>> model = VortexEmbedInference("mini")
        >>> vec = model.encode("hello world")
        >>> vec.shape
        (1, 256)
        >>> vec = model.encode("hello world", dim=128)
        >>> vec.shape
        (1, 128)
        >>> model.dim
        256
    """

    def __init__(self, model: str = "mini", *, dim: int | None = None) -> None:
        self._model_id = _resolve_model_id(model)
        from vortexa.core.v4_embedder import VortexEmbedderV4

        self._embedder = VortexEmbedderV4(self._model_id)
        self._dim = dim

    @property
    def model_id(self) -> str:
        """The HuggingFace model ID currently loaded."""
        return self._model_id

    @property
    def dim(self) -> int:
        """The full (untruncated) embedding dimensionality of the model."""
        return self._embedder.dim

    def get_embedding_dimension(self) -> int:
        """Return the full (untruncated) embedding dimensionality."""
        return self.dim

    def encode(
        self,
        texts: Union[str, List[str]],
        *,
        normalize: bool = True,
        dim: int | None = None,
    ) -> npt.NDArray[np.float32]:
        """Encode text strings into dense vector embeddings.

        Args:
            texts: A single string or a list of strings to encode.
            normalize: Whether to L2-normalize the output vectors.
                Default: ``True``.
            dim: If set, truncate embeddings to this dimension via
                Matryoshka truncation. Overrides the instance ``dim`` if
                provided. Default: ``None`` (use full dimension).

        Returns:
            A numpy array of shape ``(N, D)`` where ``N`` is the number
            of input texts and ``D`` is the (possibly truncated) embedding
            dimensionality.
        """
        effective_dim = dim if dim is not None else self._dim
        if isinstance(texts, str):
            result = self._embedder.encode(
                texts,
                normalize=normalize,
                truncate_dim=effective_dim,
            )
            return result[np.newaxis, :]
        return self._embedder.encode_batch(
            texts,
            normalize=normalize,
            truncate_dim=effective_dim,
        )


def embed(
    texts: Union[str, List[str]],
    *,
    model: str = "mini",
    dim: int | None = None,
    normalize: bool = True,
) -> npt.NDArray[np.float32]:
    """Encode text strings into dense vector embeddings (stateless convenience).

    This is a thin wrapper that creates a temporary
    :class:`VortexEmbedInference` instance for each call.

    Args:
        texts: A single string or a list of strings to encode.
        model: Model ID or alias (``mini``, ``nano``, or any
            HuggingFace model name). Default: ``mini``.
        dim: If set, truncate embeddings to this dimension via
            Matryoshka truncation. Default: ``None`` (full dimension).
        normalize: Whether to L2-normalize the output vectors.
            Default: ``True``.

    Returns:
        A numpy array of shape ``(N, D)`` where ``N`` is the number
        of input texts and ``D`` is the embedding dimensionality.

    Example:
        >>> from vortexa.core.inference import embed
        >>> vec = embed("hello world")
        >>> vec.shape
        (1, 256)
        >>> vec = embed(["hello", "world"], model="nano", dim=64)
        >>> vec.shape
        (2, 64)
    """
    model_id = _resolve_model_id(model)
    from vortexa.core.v4_embedder import VortexEmbedderV4

    embedder = VortexEmbedderV4(model_id)
    if isinstance(texts, str):
        result = embedder.encode(
            texts,
            normalize=normalize,
            truncate_dim=dim,
        )
        return result[np.newaxis, :]
    return embedder.encode_batch(
        texts,
        normalize=normalize,
        truncate_dim=dim,
    )