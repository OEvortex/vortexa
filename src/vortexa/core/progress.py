"""Progress events emitted while a codebase index is being built.

The core indexer is intentionally UI-agnostic: it emits small immutable
``IndexProgress`` events through an optional callback. The CLI and MCP server
render those events as a terminal progress bar.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

IndexPhase = Literal["model", "scan", "chunk", "embed", "bm25", "save", "done"]


@dataclass(frozen=True, slots=True)
class IndexProgress:
    """One indexing progress update.

    Attributes:
        phase: Indexing stage (``model``, ``scan``, ``chunk``, ``embed``,
            ``bm25``, ``save`` or ``done``).
        current: Completed units within this phase.
        total: Total units within this phase. ``0`` means "unknown/indeterminate".
        message: Human-readable phase label.
        detail: Optional file path or other contextual detail.
        elapsed: Seconds elapsed since the whole index run started.
    """

    phase: IndexPhase
    current: int = 0
    total: int = 0
    message: str = ""
    detail: str = ""
    elapsed: float = 0.0

    @property
    def fraction(self) -> float:
        """Completion fraction in the range 0.0–1.0 (0.0 when total unknown)."""
        if self.total <= 0:
            return 0.0
        return min(1.0, max(0.0, self.current / self.total))

    @property
    def percent(self) -> int:
        """Completion percentage as an integer."""
        return int(self.fraction * 100)


ProgressCallback = Callable[[IndexProgress], None]


def emit_progress(
    callback: ProgressCallback | None,
    phase: IndexPhase,
    current: int = 0,
    total: int = 0,
    message: str = "",
    detail: str = "",
    elapsed: float = 0.0,
) -> None:
    """Send one progress event, never letting a UI failure break indexing."""
    if callback is None:
        return
    try:
        callback(
            IndexProgress(
                phase=phase,
                current=current,
                total=total,
                message=message,
                detail=detail,
                elapsed=elapsed,
            )
        )
    except Exception:  # pragma: no cover - defensive: progress must not break indexing
        logger.debug("Index progress callback raised", exc_info=True)
