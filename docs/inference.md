# Inference API

The inference module provides a sentence-transformers-style API for
encoding arbitrary text strings into dense vector embeddings using
Vortex-Embed models.

## Quick Reference

| Function / Class | Description |
|------------------|-------------|
| `VortexEmbedInference` | Class-based inference engine with lazy model loading |
| `embed()` | Stateless convenience function |
| `_resolve_model_id()` | Resolve model aliases to HuggingFace IDs |

---

## `VortexEmbedInference`

### Constructor

```python
from vortexa.core.inference import VortexEmbedInference

model = VortexEmbedInference(model="mini", *, dim=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"mini"` | Model ID or alias (`mini`, `nano`, or any HuggingFace model name) |
| `dim` | `int \| None` | `None` | If set, truncate all output embeddings to this dimension via Matryoshka truncation |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `model_id` | `str` | The resolved HuggingFace model ID currently loaded |
| `dim` | `int` | The full (untruncated) embedding dimensionality of the model |

### Methods

#### `encode(texts, *, normalize=True, dim=None)`

Encode text strings into dense vector embeddings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `texts` | `str \| List[str]` | — | A single string or a list of strings to encode |
| `normalize` | `bool` | `True` | Whether to L2-normalize the output vectors |
| `dim` | `int \| None` | `None` | If set, truncate embeddings to this dimension. Overrides the instance `dim` if provided |

**Returns:** `numpy.ndarray` of shape `(N, D)` where:
- `N` is the number of input texts (`1` for a single string)
- `D` is the embedding dimensionality (possibly truncated)

**Example:**

```python
from vortexa.core.inference import VortexEmbedInference

model = VortexEmbedInference("mini")

# Single string → shape (1, 256)
vec = model.encode("hello world")

# Batch → shape (N, 256)
vecs = model.encode(["hello", "world", "foo bar"])

# Truncate to 64 dimensions → shape (N, 64)
vecs_small = model.encode(["hello", "world"], dim=64)

# Override instance dim → shape (N, 32)
vec_32 = model.encode("test", dim=32)
```

#### `get_embedding_dimension()`

Return the full (untruncated) embedding dimensionality.

```python
model = VortexEmbedInference("mini")
print(model.get_embedding_dimension())  # 256
```

---

## `embed()` (Stateless Convenience)

```python
from vortexa.core.inference import embed

vec = embed("hello world")
# shape: (1, 256)

vecs = embed(["hello", "world"], model="nano", dim=64)
# shape: (2, 64)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `texts` | `str \| List[str]` | — | A single string or a list of strings to encode |
| `model` | `str` | `"mini"` | Model ID or alias |
| `dim` | `int \| None` | `None` | Truncate embeddings to this dimension |
| `normalize` | `bool` | `True` | L2-normalize the output vectors |

**Returns:** `numpy.ndarray` of shape `(N, D)`.

> **Note:** The stateless `embed()` function creates a temporary
> `VortexEmbedInference` instance on every call. For repeated calls
> with the same model, prefer using the class-based API to avoid
> reloading the model.

---

## Model Aliases

| Alias | Actual Model ID | Description |
|-------|-----------------|-------------|
| `mini` | `VTXAI/vtx-embed-7M` | Default. 7M-parameter Vortex-Embed v4.5 with LF4 4-bit dequant, SIF+PC, Matryoshka. 256-dimensional embeddings. |
| `nano` | `VTXAI/vtx-embed-1M` | Lightweight 1M-parameter variant. Lower RAM, lower dimension. |

Any HuggingFace model name or path is also accepted directly:

```python
model = VortexEmbedInference("VTXAI/vtx-embed-7M")
model = VortexEmbedInference("/path/to/local/model")
```

---

## Dimension Control (Matryoshka)

Vortex-Embed models support Matryoshka representation learning,
which allows truncating embeddings to fewer dimensions without
retraining. This is useful for:

- **Lower memory usage**: Store smaller vectors in your vector store
- **Faster similarity search**: Fewer dimensions = faster dot products
- **Multi-scale retrieval**: Use different dimensions for different queries

```python
model = VortexEmbedInference("mini")

# Full dimension (256 for mini)
full = model.encode("query")
print(full.shape)  # (1, 256)

# Truncate to 128
half = model.encode("query", dim=128)
print(half.shape)  # (1, 128)

# Truncate to 64
quarter = model.encode("query", dim=64)
print(quarter.shape)  # (1, 64)
```

You can also set the default truncation at the instance level:

```python
# All encodes from this model will use dim=128 by default
model = VortexEmbedInference("mini", dim=128)
vec = model.encode("query")  # shape: (1, 128)

# Override per-call
vec_full = model.encode("query", dim=None)  # uses full 256
vec_64 = model.encode("query", dim=64)      # uses 64
```

---

## Comparison: Class API vs Stateless Function

| Aspect | `VortexEmbedInference` | `embed()` |
|--------|------------------------|-----------|
| Model loading | Once at construction | Once per call |
| Best for | Repeated calls, batch processing | One-off, quick scripts |
| Default dim | Set at construction | Not set (full dim) |
| Stateful | Yes (model is cached) | No (stateless) |
| Performance | Faster for repeated use | Slightly slower |

---

## Tips & Best Practices

### Batch Processing

```python
from vortexa.core.inference import VortexEmbedInference

model = VortexEmbedInference("mini")

# Process a large corpus in batches
corpus = ["document " + str(i) for i in range(10000)]
batch_size = 512

all_vectors = []
for i in range(0, len(corpus), batch_size):
    batch = corpus[i : i + batch_size]
    vecs = model.encode(batch)
    all_vectors.append(vecs)

import numpy as np
all_vectors = np.concatenate(all_vectors, axis=0)
print(all_vectors.shape)  # (10000, 256)
```

### Using with a Vector Store

```python
from vortexa.core.inference import VortexEmbedInference
import numpy as np

model = VortexEmbedInference("mini", dim=128)

# Encode documents
documents = ["doc 1", "doc 2", "doc 3"]
doc_vectors = model.encode(documents)

# Encode query
query = "search term"
query_vector = model.encode(query, dim=128)

# Compute similarity (cosine with normalized vectors)
scores = doc_vectors @ query_vector.T.flatten()
top_idx = np.argsort(scores)[::-1]
```

### Choosing Between `mini` and `nano`

| Criteria | `mini` (`vtx-embed-7M`) | `nano` (`vtx-embed-1M`) |
|----------|--------------------------|--------------------------|
| Parameters | 7M | 1M |
| Embedding dim | 256 | Lower (check model config) |
| Quality | Higher accuracy | Good enough for simple use cases |
| RAM usage | Higher | Lower |
| Speed | Slightly slower | Faster |
| Use case | Search, RAG, production | Edge devices, prototypes |

### Custom Model IDs

You can use any HuggingFace model ID that provides a compatible
Vortex-Embed checkpoint:

```python
# Private or gated models
model = VortexEmbedInference("your-org/your-model", token="hf_...")

# Local directory
model = VortexEmbedInference("/path/to/model/directory")

# Revision / commit
model = VortexEmbedInference("VTXAI/vtx-embed-7M", revision="main")
```

> **Note:** The `VortexEmbedInference` constructor currently does not
> expose a `token` parameter. For gated models, set the
> `HF_TOKEN` environment variable or pass a `huggingface_hub` token
> via `huggingface-cli login` before loading the model.

---

## See Also

- [Model Configuration](models.md) — Detailed model information
- [README](../README.md) — Project overview and full API reference