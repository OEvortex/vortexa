# Model Configuration

Vortexa supports configurable embedding models for both indexing
and inference. All models use the `VortexEmbedderV4` wrapper with
on-the-fly LF4 4-bit dequantization.

## Available Models

| Alias | Model ID | Type | Dimensions | Description |
|-------|----------|------|------------|-------------|
| `mini` | `VTXAI/vtx-embed-7M` | Vortex-Embed v4.5 | 256 | Default. 7M-param model with LF4 4-bit dequant, SIF+PC, Matryoshka. Best accuracy. |
| `nano` | `VTXAI/vtx-embed-1M` | Vortex-Embed v4.x | Lower | Lightweight 1M-param variant. Lower RAM, faster inference, lower dimensional embeddings. |

### Model Characteristics

| Property | `mini` (vtx-embed-7M) | `nano` (vtx-embed-1M) |
|----------|------------------------|------------------------|
| Parameters | 7M | 1M |
| Embedding dim | 256 | Lower (varies by config) |
| Quantization | LF4 4-bit | LF4 4-bit |
| RAM footprint | ~4.7 MB (on disk) | ~1.2 MB (on disk) |
| SIF+PC | Yes | Yes |
| Matryoshka | Yes | Yes |
| Best for | Production search, RAG | Edge devices, prototypes |

---

## Python API

### Choosing a Model at Construction Time

```python
from vortexa.core.indexer import CodebaseIndexer
from vortexa.core.inference import VortexEmbedInference

# Index with nano model (smaller, faster)
indexer = CodebaseIndexer(root="/path/to/project", model_id="VTXAI/vtx-embed-1M")

# Index with mini model (default, better quality)
indexer = CodebaseIndexer(root="/path/to/project", model_id="VTXAI/vtx-embed-7M")

# Inference with nano
embedder = VortexEmbedInference("nano")
vec = embedder.encode("India is a diverse country")
```

### Using Aliases

```python
# These resolve to the model IDs above
indexer = CodebaseIndexer(root=".", model_id="mini")
indexer = CodebaseIndexer(root=".", model_id="nano")

embedder = VortexEmbedInference("mini")
embedder = VortexEmbedInference("nano")
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
```

### Embed with a Specific Model

```bash
# Encode with mini (default)
vortexa embed "India is a diverse country"

# Encode with nano
vortexa embed "Indian cricket team is strong" --model nano

# Encode with custom model ID
vortexa embed "Indian agriculture output" --model VTXAI/vtx-embed-7M

# Encode and truncate dimensions
vortexa embed "India has 28 states" --model nano --dim 32
```

---

---

1. `VortexEmbedderV4` is created with a `model_id`
2. On first `embed()` or `embed_batch()` call, it lazy-loads the model
3. The model is cached for subsequent calls
4. All operations are thread-safe via a lock

```python
# This doesn't load the model yet
embedder = VortexEmbedderV4("VTXAI/vtx-embed-7M")

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