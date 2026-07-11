# Embedding Models

Embeddings power the semantic half of vortexa's hybrid retrieval, plus the
function- and symbol-level indexes. This page covers the default model and how to
swap in an alternative.

## Default: Vortex-Embed (`VTXAI/Vortex-Embed-4.7M`)

The default embedder is `core.embedding.LF4Embedder`, which loads
`VTXAI/Vortex-Embed-4.7M` — a **4-bit LF4-quantized** static embedding model
implemented in `core.lf4_model.VortexEmbedV3`.

Key properties:

- **Dimension:** 256.
- **Footprint:** ~4.7 MB on disk (4-bit weights + FP16 scales/zeros); ~3.5 MB
  when dequantized to an FP32 table in memory.
- **CPU-friendly:** no GPU required; fast inference.
- **Pure sentence similarity:** designed for RAG retrieval — tokenize, SIF-IDF
  weighting, scatter-add pooling, principal-component removal, L2-normalize. No
  code-search tricks, no extension/path bias.
- **Tokenizer:** Hugging Face `tokenizers` (shipped with the model as
  `tokenizer.json`).
- **Cached:** the model is downloaded once via `huggingface-hub` and cached
  locally; a single instance is loaded and reused (thread-safe).

```python
from vortexa.core.indexer import CodebaseIndexer

indexer = CodebaseIndexer(root=".", model_id="VTXAI/Vortex-Embed-4.7M")
```

## Alternative embedders

Any object satisfying the `Embedder` protocol can be passed via the `model`
constructor argument:

```python
class Embedder(Protocol):
    @property
    def dim(self) -> int: ...
    def embed(self, text: str) -> np.ndarray: ...
    def embed_batch(self, texts: list[str]) -> np.ndarray: ...
    @property
    def memo_key(self) -> tuple: ...
```

### Model2Vec (`[full]`)

```python
from vortexa.core.embedding import Model2VecEmbedder

indexer = CodebaseIndexer(root=".", model=Model2VecEmbedder("AI4free/JARVIS-tool-search-v1"))
```

Static, fast, and tiny. Good for tool/name-style search.

### SentenceTransformers (`[full]`)

```python
from vortexa.core.embedding import SentenceTransformerEmbedder

indexer = CodebaseIndexer(root=".", model=SentenceTransformerEmbedder("all-MiniLM-L6-v2"))
```

Higher-quality dense embeddings for general semantic search, at the cost of a
larger model and slower CPU inference.

## Choosing an embedder

| Model | Size | Speed | Best for |
|-------|------|-------|----------|
| `VTXAI/Vortex-Embed-4.7M` (default) | ~4.7 MB | Fast (CPU) | General RAG-style code retrieval, zero extra deps. |
| Model2Vec | small | Very fast | Tool/name lookups, constrained environments. |
| SentenceTransformers | 20–400 MB | Slower (CPU) | Highest semantic fidelity when quality matters most. |

Because the Vortex Score and BM25 signals run alongside embeddings, even a small
static model yields strong retrieval when combined with filename, path, and
graph signals.

## Notes

- `dim` must be consistent for a given index; switching embedders means a full
  re-index (`indexer.index(force=True)`).
- The embedding model is loaded lazily on first use and cached, so importing
  vortexa is cheap.
- `memo_key` lets vortexa invalidate its embedding cache when the model identity
  changes.

## Next steps

- Understand the [Knowledge Graph & Scoring](knowledge-graph.md) that consumes
  these embeddings.
- Browse the full [Python API Reference](python-api.md).
