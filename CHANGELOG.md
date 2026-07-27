# Changelog

## [0.3.0] - Unreleased

### Changed
- Migrated default embedding model from `VTXAI/Vortex-Embed-v3-sentence` to `VTXAI/vtx-embed-7M` (Vortex-Embed v4.5)
- Replaced `VortexEmbedderV3` with `VortexEmbedderV4` in `indexer.py` and `context_engine.py`
- Added on-the-fly LF4 4-bit dequantization via `VortexEmbedV4_5` in `lf4_v4_model.py`
- Added Matryoshka embedding support and SIF+PC weighting to the default embedder
- Removed deprecated `VortexEmbedderV3` and `VortexEmbedV3` modules

### Added
- New `VortexEmbedderV4` thread-safe embedder wrapper (`src/vortexa/core/v4_embedder.py`)
- New `VortexEmbedV4_5` model class with LF4 dequantization (`src/vortexa/core/lf4_v4_model.py`)
- `VortexEmbedderV4` auto-creates `VortexEmbedV4_5` when `model_id` is `VTXAI/vtx-embed-7M`
- Alternative embedder models still available: `Model2VecEmbedder`, `SentenceTransformerEmbedder`, `LF4Embedder`

### Removed
- Deleted `src/vortexa/core/v3_embedder.py` (superseded by v4_embedder.py)
- Deleted `src/vortexa/core/lf4_v3_model.py` (superseded by lf4_v4_model.py)
- Deleted old V2-era test files (`v2_test.py`, `v2_line_test.py`, `v2_tune.py`)

### Fixed
- Cleaned up stale `__pycache__` directories
- Verified all imports work correctly with `uv run python`