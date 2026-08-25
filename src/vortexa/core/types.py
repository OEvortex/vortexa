"""Core types for the JARVIS codebase indexer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

EmbeddingMatrix: TypeAlias = npt.NDArray[np.float32]


class SearchMode(str, Enum):
    """Search strategy."""

    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    BM25 = "bm25"


class Encoder(Protocol):
    """Protocol for embedding models (legacy, use Embedder instead)."""

    @property
    def dim(self) -> int:
        """The dimensionality of the embedding."""
        ...

    def encode(self, texts: Sequence[str], /, **kwargs: Any) -> EmbeddingMatrix:
        """Encode texts into embeddings as a 2D float32 array."""
        ...


@dataclass(frozen=True, slots=True)
class Lineage:
    """Tracks the exact source location of a chunk in the original file."""

    source_path: str
    start_line: int
    end_line: int
    byte_start: int = 0
    byte_end: int = 0


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """Configuration for code chunking (cocoindex RecursiveSplitter-style)."""

    chunk_size: int = 1500
    min_chunk_size: int | None = None  # Defaults to chunk_size // 2
    chunk_overlap: int = 200
    language: str | None = None

    def __post_init__(self) -> None:
        if self.min_chunk_size is None:
            object.__setattr__(self, "min_chunk_size", self.chunk_size // 2)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A single indexable unit of code with lineage tracking."""

    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str | None = None
    lineage: Lineage | None = None
    chunk_hash: str | None = None  # SHA256 for memoization

    @property
    def location(self) -> str:
        """File path and line range as a string."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single search result with score and source."""

    chunk: Chunk
    score: float
    source: SearchMode


@dataclass(frozen=True, slots=True)
class IndexStats:
    """Statistics about the current index state."""

    indexed_files: int = 0
    total_chunks: int = 0
    languages: dict[str, int] = field(default_factory=dict)
    memo_hits: int = 0  # Chunks skipped due to memoization
    memo_misses: int = 0  # Chunks re-embedded
    index_time_ms: float = 0  # Elapsed wall-clock time for the index run


@dataclass(frozen=True, slots=True)
class GraphContext:
    """Compact structural context for one search-result file."""

    key_symbol: str = ""
    incoming: tuple[str, ...] = ()
    outgoing: tuple[str, ...] = ()
