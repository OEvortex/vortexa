"""Context compression — packs expanded context into efficient agent-ready format.

Design: point the agent at the right code + a relevant snippet to confirm.
The engine should NOT dump everything it knows — the agent reads files.
"""

from __future__ import annotations

import logging

from vortexa.core.types import ContextPack

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


def compress(
    pack: ContextPack,
    max_tokens: int = 8000,
    max_primary_chunks: int = 10,
) -> ContextPack:
    """Compress a ContextPack — strip expansion noise, keep primary results."""
    if not pack.primary_chunks:
        return pack

    primary_sorted = sorted(pack.primary_chunks, key=lambda r: -r.score)[:max_primary_chunks]

    return ContextPack(
        query=pack.query,
        primary_chunks=primary_sorted,
        related_files=pack.related_files[:5],
        test_files=pack.test_files[:3],
        imports=pack.imports[:2],
        imported_by=pack.imported_by[:1],
        symbols=pack.symbols[:5],
        callers=pack.callers[:2],
        callees=pack.callees[:2],
        sibling_chunks=[],
        dependency_chain=pack.dependency_chain[:3],
        confidence=pack.confidence,
        reasoning_trace=[],
        total_tokens=sum(estimate_tokens(r.chunk.content) for r in primary_sorted),
    )


def format_for_agent(pack: ContextPack, max_snippet: int = 300) -> str:
    """Format context for the agent: ranked files + meaningful code snippet + navigation hints.

    The snippet is the actual matched code (enough to confirm relevance).
    Navigation hints point to tests, imports, callers.
    """
    lines: list[str] = []
    header = f"[{pack.confidence:.2f}] {pack.query}"
    if pack.primary_chunks:
        header += f"  ({len(pack.primary_chunks)} files)"
    lines.append(header)
    lines.append("")

    for i, result in enumerate(pack.primary_chunks, 1):
        chunk = result.chunk
        span = f"{chunk.start_line}-{chunk.end_line}" if chunk.end_line != chunk.start_line else str(chunk.start_line)
        lines.append(f"  {i}. {chunk.file_path}:{span}  [{result.score:.2f}]")
        snippet = chunk.content.strip()[:max_snippet]
        for l in snippet.split("\n"):
            lines.append(f"     {l}")

    if pack.symbols:
        names = [s.name for s in pack.symbols[:6]]
        if len(pack.symbols) > 6:
            names.append("...")
        lines.append(f"  sym: {' '.join(names)}")

    hints = []
    if pack.test_files:
        hints.append(f"tests: {', '.join(f.split('/')[-1] for f in pack.test_files[:2])}")
    if pack.imports:
        hints.append(f"deps: {', '.join(f.split('/')[-1] for f in pack.imports[:2])}")
    if pack.callers:
        hints.append(f"callers: {', '.join(c.name for c in pack.callers[:2])}")
    if hints:
        lines.append(f"  {' | '.join(hints)}")

    return "\n".join(lines)
