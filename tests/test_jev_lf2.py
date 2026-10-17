from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file


class _SequenceEncoding:
    def __init__(self, ids):
        self.ids = ids


class _VariableTokenizer:
    """Maps each text to a distinct id sequence so batching effects are visible."""

    TABLE = {"a": [0], "b": [1], "long": [0, 1, 0, 1]}

    @classmethod
    def from_file(cls, path):
        return cls()

    def encode_batch(self, texts):
        return [_SequenceEncoding(self.TABLE[t]) for t in texts]


def _write_jev_checkpoint(path: Path, *, max_position: int = 4) -> None:
    """Minimal VTX-JEV layout: LF2 config/weights plus gate params."""
    config = {
        "model_type": "vortex_embed_lf2",
        "quantization": "lf2_2bit",
        "bits": 2,
        "vocab_size": 2,
        "embedding_dim": 4,
        "block_size": 4,
        "num_blocks": 1,
        "global_min": 0.0,
        "global_max": 3.0,
        "global_scale_max": 1.0,
    }
    path.mkdir()
    (path / "config_lf2.json").write_text(json.dumps(config))
    save_file(
        {
            "embedding_packed": np.asarray([[228], [228]], dtype=np.uint8),
            "embedding_scale_u8": np.asarray([[255], [255]], dtype=np.uint8),
            "embedding_min_u8": np.asarray([[0], [255]], dtype=np.uint8),
        },
        str(path / "model_lf2.safetensors"),
    )
    (path / "tokenizer.json").write_text("{}")

    rng = np.random.default_rng(0)
    np.savez(
        path / "gate_params.npz",
        W_u=rng.normal(size=(4, 3)).astype(np.float32),
        W_p=np.cumsum(rng.normal(size=(max_position, 3)), axis=0).astype(np.float32),
        b=rng.normal(size=(3,)).astype(np.float32),
        v=rng.normal(size=(3,)).astype(np.float32),
        max_position=np.asarray(max_position, dtype=np.int64),
    )


def test_jev_layout_loads_lf2_with_gate_pooling(tmp_path, monkeypatch):
    from vortexa.core import model as lf2_model

    checkpoint = tmp_path / "jev"
    _write_jev_checkpoint(checkpoint)
    monkeypatch.setattr(lf2_model, "Tokenizer", _VariableTokenizer)

    model = lf2_model.VortexEmbedLF2.from_pretrained(checkpoint)

    assert model.pooling == "gate"
    assert model.gate is not None
    assert model.pooling_max_position == 4
    assert model.dim == 4


def test_gate_pooling_output_is_unit_norm(tmp_path, monkeypatch):
    from vortexa.core import model as lf2_model

    checkpoint = tmp_path / "jev"
    _write_jev_checkpoint(checkpoint)
    monkeypatch.setattr(lf2_model, "Tokenizer", _VariableTokenizer)
    model = lf2_model.VortexEmbedLF2.from_pretrained(checkpoint)

    values = model.encode_batch(["a", "long"])

    assert values.shape == (2, 4)
    assert values.dtype == np.float32
    assert np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-6)


def test_gate_positions_restart_per_text(tmp_path, monkeypatch):
    """Regression: positions must restart per text, not run across the batch.

    Encoding the same text alone and as the second batch item must agree. A
    batch-global position counter shifts later texts and silently corrupts
    every pooled vector.
    """
    from vortexa.core import model as lf2_model

    checkpoint = tmp_path / "jev"
    _write_jev_checkpoint(checkpoint)
    monkeypatch.setattr(lf2_model, "Tokenizer", _VariableTokenizer)
    model = lf2_model.VortexEmbedLF2.from_pretrained(checkpoint)

    alone = model.encode_batch(["b"])[0]
    batched = model.encode_batch(["a", "b"])[1]
    batched_last = model.encode_batch(["a", "long", "b"])[2]

    assert np.allclose(alone, batched, atol=1e-6)
    assert np.allclose(alone, batched_last, atol=1e-6)


def test_pooling_falls_back_to_sif_without_gate_params(tmp_path, monkeypatch):
    from vortexa.core import model as lf2_model

    checkpoint = tmp_path / "plain"
    _write_jev_checkpoint(checkpoint)
    (checkpoint / "gate_params.npz").unlink()
    monkeypatch.setattr(lf2_model, "Tokenizer", _VariableTokenizer)

    model = lf2_model.VortexEmbedLF2.from_pretrained(checkpoint)

    assert model.pooling == "sif"
    assert model.gate is None


def test_gate_pooling_requires_gate_params(tmp_path):
    from vortexa.core.model import LF2Config, VortexEmbedLF2

    config = LF2Config(vocab_size=2, embedding_dim=4, block_size=4, num_blocks=1)
    with pytest.raises(ValueError, match="gate pooling requires"):
        VortexEmbedLF2(
            np.zeros((2, 1), dtype=np.uint8),
            np.zeros((2, 1), dtype=np.uint8),
            np.zeros((2, 1), dtype=np.uint8),
            str(tmp_path / "tokenizer.json"),
            config,
            pooling="gate",
        )


def test_index_directory_isolates_models(tmp_path):
    from vortexa.core.indexer import _index_dir_for_root

    root = tmp_path / "repo"
    mini = _index_dir_for_root(root, "VTXAI/vtx-embed-7M-lf2")
    jev = _index_dir_for_root(root, "VTXAI/VTX-JEV-3")

    assert mini != jev
    assert _index_dir_for_root(root, None) != mini
    assert _index_dir_for_root(root, "VTXAI/vtx-embed-7M-lf2") == mini
