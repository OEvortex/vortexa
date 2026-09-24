# Model Configuration

Vortexa supports configurable embedding models for both indexing
and inference. All models use the `VortexEmbedderV4` wrapper. The
`mini` and `nano` aliases use native LF2 2-bit checkpoints by default;
use `mini-full` or `nano-full` when the original LF4 checkpoints are
needed.

## Available Models

| Alias | Model ID | Type | Dimensions | Description |
|-------|----------|------|------------|-------------|
| `mini` | `VTXAI/vtx-embed-7M-lf2` | Vortex-Embed LF2 | 256 | Default. Native 2-bit 7M model; ~2.36 MB integer parameters. |
| `nano` | `VTXAI/vtx-embed-1M-lf2` | Vortex-Embed LF2 | 64 | Default lightweight model. Native 2-bit 1M model; ~0.39 MB integer parameters. |
| `mini-full` | `VTXAI/vtx-embed-7M` | Vortex-Embed v4.5 LF4 | 256 | Original 4-bit checkpoint for maximum quality. |
| `nano-full` | `VTXAI/vtx-embed-1M` | Vortex-Embed v4.x LF4 | Lower | Original 4-bit lightweight checkpoint. |
| `mini-lf2` | `VTXAI/vtx-embed-7M-lf2` | Vortex-Embed LF2 | 256 | Explicit alias for the default mini model. |
| `nano-lf2` | `VTXAI/vtx-embed-1M-lf2` | Vortex-Embed LF2 | 64 | Explicit alias for the default nano model. |

### Model Characteristics

| Property | `mini` | `nano` | `mini-full` | `nano-full` |
|----------|--------|-------|------------|------------|
| Parameters | 7.56M | 1.05M | 7M | 1M |
| Embedding dim | 256 | 64 | 256 | Lower (varies by config) |
| Quantization | LF2 native 2-bit | LF2 native 2-bit | LF4 4-bit | LF4 4-bit |
| Integer parameters | ~2.36 MB | ~0.39 MB | — | — |
| SIF+PC | Yes | Yes | Yes | Yes |
| Matryoshka | Yes | Yes | Yes | Yes |
| Best for | Default quality/speed balance | Maximum speed | Maximum quality | Original lightweight quality |

---

## Python API

### Choosing a Model at Construction Time

```python
from vortexa.core.indexer import CodebaseIndexer
from vortexa.core.inference import VortexEmbedInference

# Index with nano model (smaller, faster, LF2 by default)
indexer = CodebaseIndexer(root="/path/to/project", model_id="nano")

# Index with mini model (default, LF2 by default)
indexer = CodebaseIndexer(root="/path/to/project", model_id="mini")

# Opt into the original LF4 checkpoint explicitly
indexer = CodebaseIndexer(root="/path/to/project", model_id="mini-full")

# Inference with nano
embedder = VortexEmbedInference("nano")
vec = embedder.encode("India is a diverse country")
```

### Using Aliases

```python
# `mini` and `nano` use LF2 by default
indexer = CodebaseIndexer(root=".", model_id="mini")
indexer = CodebaseIndexer(root=".", model_id="nano")

# Explicit full LF4 checkpoints
indexer = CodebaseIndexer(root=".", model_id="mini-full")
indexer = CodebaseIndexer(root=".", model_id="nano-full")
```

---

## CLI

### Search with a Specific Model

```bash
# Use nano (smaller, faster)
vortexa -q "authentication" --model nano /path/to/project

# Use mini explicitly (same as default)
vortexa -q "authentication" --model mini /path/to/project

# Use a custom HuggingFace model ID
vortexa -q "authentication" --model VTXAI/vtx-embed-1M /path/to/project

# `mini` and `nano` already use LF2 by default
vortexa -q "authentication" --model mini /path/to/project
vortexa -q "authentication" --model nano /path/to/project

# Use the original full LF4 checkpoints explicitly
vortexa -q "authentication" --model mini-full /path/to/project
vortexa -q "authentication" --model nano-full /path/to/project
```

### Embed with a Specific Model

```bash
# Encode with mini (default)
vortexa embed "India is a diverse country"

# Encode with nano
vortexa embed "Indian cricket team is strong" --model nano

# Encode with the default native 2-bit models
vortexa embed "What is the capital of India?" --model mini
vortexa embed "Explain gravity and general relativity" --model nano

# -f/--fast is still accepted as an explicit compatibility flag
vortexa embed "What is the capital of India?" --model mini --fast

# Encode with a custom model ID
vortexa embed "Indian agriculture output" --model VTXAI/vtx-embed-7M

# Encode and truncate dimensions
vortexa embed "India has 28 states" --model nano --dim 32
```

---

---

1. `VortexEmbedderV4` is created with a `model_id`
2. On first `embed()` or `embed_batch()` call, it lazy-loads the model
3. The loader reads `config.json` and selects the LF2 or LF4 decoder automatically
4. The model is cached for subsequent calls
5. All operations are thread-safe via a lock

```python
# This doesn't load the model yet
embedder = VortexEmbedderV4("mini")

# This triggers model download (if not cached) and loading
vec = embedder.embed("India is a diverse country")

# Subsequent calls reuse the loaded model (no re-download)
vec2 = embedder.embed("Chennai is in Tamil Nadu")
```

---

## Dimension Control

All Vortex-Embed models support Matryoshka representation learning,
allowing you to truncate embeddings to fewer dimensions:

```python
from vortexa.core.inference import VortexEmbedInference

model = VortexEmbedInference("mini")

# Full dimension (256)
vec_full = model.encode("India is a diverse country")

# Truncate to 128
vec_128 = model.encode("India has 28 states", dim=128)

# Truncate to 64
vec_64 = model.encode("Chennai is in Tamil Nadu", dim=64)
```

---

## Adding a Custom Model

To add support for a new Vortex-Embed model:

1. Add the model ID to the `_MODEL_ALIASES` dict in `inference.py`
2. Add a constant in `v4_embedder.py` (`NANO_MODEL_ID`, etc.)
3. Update this documentation

The model will automatically work with all Vortex-Embed v4.x
features (LF4 dequantization, SIF+PC, Matryoshka).

---

## See Also

- [Inference API](inference.md) — Detailed inference API documentation
- [README](../README.md) — Project overview and full API reference