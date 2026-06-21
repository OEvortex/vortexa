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
    PATH = "path"
    SYMBOL = "symbol"


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


# ── V2 Layer Types ─────────────────────────────────────────────────────────


class SymbolKind(str, Enum):
    """Kinds of code symbols."""

    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    INTERFACE = "interface"
    STRUCT = "struct"
    ENUM = "enum"
    TRAIT = "trait"
    TYPE = "type"
    MODULE = "module"
    NAMESPACE = "namespace"
    UNKNOWN = "unknown"


class EdgeType(str, Enum):
    """Types of edges in the knowledge graph."""

    IMPORTS = "imports"
    CALLS = "calls"
    USES = "uses"
    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    TESTS = "tests"
    REFERENCES = "references"
    CONTAINS = "contains"
    DEFINED_BY = "defined_by"
    SIBLING = "sibling"


# ── V2 Graph Traversal Types ───────────────────────────────────────────────


class GraphTraversalMode(str, Enum):
    """Traversal strategy for knowledge graph queries.

    BFS — broad context expansion (default): useful for "what does this connect to?"
    DFS — trace a specific path: useful for "how does A relate to B?"
    """

    BFS = "bfs"
    DFS = "dfs"


# Structural relations: edges that represent real code structure (not type
# annotations, prose references, or other noise). Used by the hybrid search
# graph enrichment to filter the incoming/outgoing context per result file.
# Mirrors JARVIS's STRUCT_RELATIONS set.
GRAPH_STRUCTURAL_RELATIONS: frozenset[str] = frozenset({
    "call", "calls", "called",
    "import", "imports", "imported", "imports_from", "re_exports",
    "method",
    "contains",
    "inherits", "extends",
    "implements",
    "references",
})

# Default relation filter for graph_query. Suppresses type-annotation noise
# (str/bool/Path edges created by AST extractors) and surfaces only structural
# relations by default. Callers can override by passing an explicit list.
GRAPH_DEFAULT_RELATION_FILTER: list[str] = [
    "call", "calls", "import", "imports", "imports_from", "method",
    "contains", "references", "inherits", "extends", "implements",
    "re_exports", "rationale_for",
]


@dataclass(frozen=True, slots=True)
class GraphContext:
    """Compact structural context for one search-result file.

    Used by hybrid search enrichment — for each result file we surface one
    query-relevant symbol plus one incoming and one outgoing structural edge.
    Keeps the per-result overhead small while still telling the agent what the
    file is connected to in the codebase.
    """

    key_symbol: str
    incoming: tuple[str, ...] = ()
    outgoing: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GraphNodeInfo:
    """Detailed information about a single knowledge-graph node."""

    id: str
    label: str
    kind: str
    file_path: str | None
    degree: int
    community: str | None = None


@dataclass(frozen=True, slots=True)
class GraphEdgeInfo:
    """One edge in the knowledge graph, formatted for agent consumption."""

    source: str
    target: str
    relation: str
    direction: str  # "in" or "out"


@dataclass(frozen=True, slots=True)
class GraphPath:
    """A shortest-path result between two nodes."""

    source: str
    target: str
    hops: int
    segments: tuple[tuple[str, str, str], ...] = ()
    # Each segment is (source_label, relation, target_label).


@dataclass(frozen=True, slots=True)
class SymbolInfo:
    """Structured information about a code symbol."""

    name: str
    kind: SymbolKind
    file_path: str
    start_line: int
    end_line: int
    docstring: str | None = None
    parent: str | None = None  # parent symbol name (e.g. class for method)
    signature: str | None = None  # full function/class signature line


@dataclass(frozen=True, slots=True)
class ImportInfo:
    """Information about an import statement."""

    source_file: str
    imported_module: str
    imported_names: list[str] = field(default_factory=list)
    is_relative: bool = False
    alias: str | None = None


@dataclass(frozen=True, slots=True)
class GraphNode:
    """A node in the repository knowledge graph."""

    id: str  # unique identifier (e.g. "file:src/main.py", "class:MyClass")
    kind: SymbolKind
    label: str  # human-readable name
    file_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """An edge in the repository knowledge graph."""

    source: str  # source node id
    target: str  # target node id
    type: EdgeType
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ContextPack:
    """Assembled context for an agent to solve a task.

    This is the primary output of the context engine.
    """

    query: str
    primary_chunks: list[SearchResult] = field(default_factory=list)
    related_files: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    imported_by: list[str] = field(default_factory=list)
    symbols: list[SymbolInfo] = field(default_factory=list)
    callers: list[SymbolInfo] = field(default_factory=list)
    callees: list[SymbolInfo] = field(default_factory=list)
    sibling_chunks: list[Chunk] = field(default_factory=list)
    dependency_chain: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_trace: list[str] = field(default_factory=list)
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class VortexScoreConfig:
    """Configuration for the Vortex Score weighting system."""

    embedding_weight: float = 1.0
    filename_weight: float = 0.8
    path_weight: float = 0.6
    symbol_weight: float = 1.2
    graph_weight: float = 0.5
    import_weight: float = 0.4
    bm25_idf_weight: float = 0.7
    structural_weight: float = 0.3

    @classmethod
    def default(cls) -> VortexScoreConfig:
        return cls()

    def as_dict(self) -> dict[str, float]:
        return {
            "embedding": self.embedding_weight,
            "filename": self.filename_weight,
            "path": self.path_weight,
            "symbol": self.symbol_weight,
            "graph": self.graph_weight,
            "import": self.import_weight,
            "bm25_idf": self.bm25_idf_weight,
            "structural": self.structural_weight,
        }
