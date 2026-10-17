from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file


class _Encoding:
    def __init__(self, ids):
        self.ids = ids


class _FakeTokenizer:
    @classmethod
    def from_file(cls, path):
        return cls()

    def encode_batch(self, texts):
        return [_Encoding([0, 1]) for _ in texts]


def _write_lf2_checkpoint(path: Path) -> None:
    config = {
        "vocab_size": 2,
        "embedding_dim": 4,
        "block_size": 4,
        "num_blocks": 1,
        "global_min": 0.0,
        "global_max": 3.0,
        "global_scale_max": 1.0,
        "matryoshka_dim": None,
        "quantization": "lf2",
        "bits": 2,
    }
    packed = np.asarray([[228], [228]], dtype=np.uint8)
    scale_u8 = np.asarray([[255], [255]], dtype=np.uint8)
    min_u8 = np.asarray([[0], [255]], dtype=np.uint8)
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    save_file(
        {
            "embedding_packed": packed,
            "embedding_scale_u8": scale_u8,
            "embedding_min_u8": min_u8,
        },
        str(path / "model.safetensors"),
    )
    (path / "tokenizer.json").write_text("{}")


def test_v4_embedder_selects_lf2_from_config(tmp_path, monkeypatch):
    from vortexa.core import model as lf2_model
    from vortexa.core.model import VortexEmbedderV4

    checkpoint = tmp_path / "lf2"
    _write_lf2_checkpoint(checkpoint)
    monkeypatch.setattr(lf2_model, "Tokenizer", _FakeTokenizer)

    embedder = VortexEmbedderV4(str(checkpoint))
    values = embedder.encode("native 2-bit text")

    assert values.shape == (4,)
    assert values.dtype == np.float32
    assert np.isclose(np.linalg.norm(values), 1.0)
    assert embedder._model.__class__ is lf2_model.VortexEmbedLF2


def test_lf2_dequantizes_and_pools_native_codes(tmp_path, monkeypatch):
    from vortexa.core import model as lf2_model
    from vortexa.core.model import VortexEmbedderV4

    checkpoint = tmp_path / "lf2"
    _write_lf2_checkpoint(checkpoint)
    monkeypatch.setattr(lf2_model, "Tokenizer", _FakeTokenizer)
    embedder = VortexEmbedderV4(str(checkpoint))

    values = embedder.encode_batch(["a", "b"], normalize=False)

    expected = np.asarray([[1.5, 2.5, 3.5, 4.5], [1.5, 2.5, 3.5, 4.5]])
    assert np.allclose(values, expected)


def test_lf2_rejects_non_divisible_embedding_dimension(tmp_path):
    from vortexa.core.model import LF2Config, VortexEmbedLF2

    config = LF2Config(vocab_size=2, embedding_dim=3, block_size=4, num_blocks=1)
    with pytest.raises(ValueError, match="embedding_dim"):
        VortexEmbedLF2(
            np.zeros((2, 1), dtype=np.uint8),
            np.zeros((2, 1), dtype=np.uint8),
            np.zeros((2, 1), dtype=np.uint8),
            str(tmp_path / "tokenizer.json"),
            config,
        )


def test_fast_resolves_mini_and_nano_to_lf2():
    from vortexa.core.inference import _resolve_model_id

    assert _resolve_model_id("mini", fast=True) == "VTXAI/vtx-embed-7M-lf2"
    assert _resolve_model_id("nano", fast=True) == "VTXAI/vtx-embed-1M-lf2"
    assert _resolve_model_id("mini") == "VTXAI/vtx-embed-7M-lf2"
    assert _resolve_model_id("nano") == "VTXAI/vtx-embed-1M-lf2"
    assert _resolve_model_id("mini-full") == "VTXAI/vtx-embed-7M"
    assert _resolve_model_id("nano-full") == "VTXAI/vtx-embed-1M"

    from vortexa.core.model import VortexEmbedderV4

    assert VortexEmbedderV4("mini")._model_id == "VTXAI/vtx-embed-7M-lf2"
    assert VortexEmbedderV4("nano-full")._model_id == "VTXAI/vtx-embed-1M"


def test_fast_embed_parser_exposes_short_flag():
    from vortexa.interfaces.cli import _build_embed_parser

    args = _build_embed_parser().parse_args(["text", "-f"])
    assert args.model == "mini"
    assert args.fast is True
