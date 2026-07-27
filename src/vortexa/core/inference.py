"""Standalone inference engine for VTXAI Vortex-Embed models.

Provides a simple, embeddable API for encoding arbitrary text strings
into dense vector embeddings using Vortex-Embed v4.x models.

Usage:
    from vortexa.core.inference import embed

    # Encode a single string (default: mini model)
    vec = embed("hello world")

    # Encode multiple strings with the nano model
    vecs = embed(["hello", "world"], model="nano")

    # Use any HuggingFace model ID
    vecs = embed(["query"], model="VTXAI/vtx-embed-7M")
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


def embed(
    texts: Union[str, List[str]],
    *,
    model: str = "mini",
) -> npt.NDArray[np.float32]:
    """Encode text strings into dense vector embeddings.

    Args:
        texts: A single string or a list of strings to encode.
        model: Model ID or alias (``mini``, ``nano``, or any
            HuggingFace model name). Default: ``mini``.

    Returns:
        A numpy array of shape ``(N, D)`` where ``N`` is the number
        of input texts and ``D`` is the embedding dimensionality.
    """
    model_id = _resolve_model_id(model)
    from vortexa.core.v4_embedder import VortexEmbedderV4

    embedder = VortexEmbedderV4(model_id)
    if isinstance(texts, str):
        return embedder.embed(texts)[np.newaxis, :]
    return embedder.embed_batch(texts)