"""Code-aware chunking using tree-sitter with line-based fallback.

Splits source code into chunks respecting AST boundaries (functions, classes, etc.)
when tree-sitter supports the language, otherwise falls back to line-based splitting.

Supports configurable chunk_size, min_chunk_size, and chunk_overlap
(inspired by cocoindex's RecursiveSplitter).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from functools import lru_cache

from vortexa.core.types import Chunk, ChunkConfig, Lineage

logger = logging.getLogger(__name__)


@dataclass
class ChunkBoundary:
    """The output of the internal chunking algorithm."""

    start: int
    end: int


@lru_cache(maxsize=64)
def _get_parser(language: str):
    """Get a tree-sitter parser for the given language. Returns None if unavailable."""
    try:
        from tree_sitter_language_pack import get_parser as _get_ts_parser

        return _get_ts_parser(language)
    except Exception:
        return None


def is_supported_language(language: str) -> bool:
    """Check if tree-sitter supports the given language."""
    return _get_parser(language) is not None


def _merge_adjacent_chunks(
    chunks: list[ChunkBoundary],
    desired_length: int,
    overlap: int = 0,
) -> list[ChunkBoundary]:
    """Merge adjacent chunks up to the desired length, with optional overlap.

    When overlap > 0, each chunk (after the first) starts `overlap` bytes
    before the end of the previous chunk, creating overlapping regions.
    """
    if not chunks:
        return []

    merged: list[ChunkBoundary] = []
    current_start = chunks[0].start
    current_end = chunks[0].end
    current_length = current_end - current_start

    for group in chunks[1:]:
        start, end = group.start, group.end
        length = end - start

        if current_length + length > desired_length:
            merged.append(ChunkBoundary(start=current_start, end=current_end))
            # Apply overlap: start the next chunk overlap bytes before current end
            if overlap > 0:
                current_start = max(current_end - overlap, start)
            else:
                current_start = start
            current_end = end
            current_length = current_end - current_start
            continue

        current_end = end
        current_length += length

    merged.append(ChunkBoundary(start=current_start, end=current_end))
    return merged


def _merge_node_inner(node, desired_length: int) -> list[ChunkBoundary]:
    """Recursively merge and split AST nodes into chunks."""
    if not node.children:
        return [ChunkBoundary(node.start_byte, node.end_byte)]

    groups: list[ChunkBoundary] = []
    children = node.children
    index = 0

    while index < len(children):
        child = children[index]
        start = child.start_byte
        end = child.end_byte
        length = child.end_byte - child.start_byte

        index += 1

        # If this single chunk is longer than desired, recurse into it
        if length > desired_length:
            groups.extend(_merge_node_inner(child, desired_length))
            continue

        while index < len(children):
            child = children[index]
            child_length = child.end_byte - child.start_byte

            if length + child_length > desired_length:
                break

            end = child.end_byte
            length += child_length
            index += 1

        groups.append(ChunkBoundary(start, end))

    return groups


def _merge_node(node, desired_length: int, overlap: int = 0) -> list[ChunkBoundary]:
    """Recursively turn AST nodes into chunks, then merge adjacent chunks."""
    raw_chunks = _merge_node_inner(node, desired_length)
    return _merge_adjacent_chunks(raw_chunks, desired_length, overlap)


def chunk_lines(text: str, desired_length: int, overlap: int = 0) -> list[ChunkBoundary]:
    """Chunk source code by line boundaries with optional overlap."""
    if not text.strip():
        return []
    lines_as_groups: list[ChunkBoundary] = []
    index = 0
    for line in text.splitlines(keepends=True):
        lines_as_groups.append(ChunkBoundary(start=index, end=index + len(line)))
        index += len(line)

    return _merge_adjacent_chunks(lines_as_groups, desired_length, overlap)


def chunk_source(
    source: str,
    file_path: str,
    language: str | None,
    config: ChunkConfig | None = None,
) -> list[Chunk]:
    """Chunk source code into indexable units with lineage tracking.

    Uses tree-sitter for AST-aware chunking when the language is supported,
    falls back to line-based chunking otherwise.

    :param source: Source code text.
    :param file_path: Relative file path for the chunk metadata.
    :param language: Detected programming language (or None).
    :param config: Chunking configuration (chunk_size, overlap, etc.).
    :return: List of Chunk objects with lineage and chunk_hash.
    """
    if not source.strip():
        return []

    if config is None:
        config = ChunkConfig()

    chunk_boundaries = None

    if language is not None and is_supported_language(language):
        parser = _get_parser(language)
        if parser is not None:
            try:
                as_bytes = source.encode("utf-8")
                root = parser.parse(as_bytes).root_node
                chunk_boundaries = _merge_node(root, config.chunk_size, config.chunk_overlap)
                # Convert byte offsets to char offsets
                char_boundaries = []
                for boundary in chunk_boundaries:
                    start_char = len(as_bytes[: boundary.start].decode("utf-8"))
                    end_char = len(as_bytes[: boundary.end].decode("utf-8"))
                    char_boundaries.append(ChunkBoundary(start=start_char, end=end_char))
                chunk_boundaries = char_boundaries
            except Exception:
                logger.debug("Tree-sitter chunking failed for %s, falling back", file_path)
                chunk_boundaries = None

    if chunk_boundaries is None:
        chunk_boundaries = chunk_lines(source, config.chunk_size, config.chunk_overlap)

    # Compute source hash for memoization
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

    chunks: list[Chunk] = []
    for boundary in chunk_boundaries:
        end_index = max(boundary.end - 1, boundary.start)
        text = source[boundary.start : end_index + 1]
        if not text.strip():
            continue

        start_line = source[: boundary.start].count("\n") + 1
        end_line = source[:end_index].count("\n") + 1

        # Compute chunk-specific hash for memoization
        chunk_hash = hashlib.sha256(
            f"{file_path}:{source_hash}:{boundary.start}:{boundary.end}".encode()
        ).hexdigest()[:16]

        # Compute byte offsets for lineage
        as_bytes = source.encode("utf-8")
        byte_start = len(source[: boundary.start].encode("utf-8"))
        byte_end = len(source[:end_index].encode("utf-8"))

        chunks.append(
            Chunk(
                content=text,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                language=language,
                lineage=Lineage(
                    source_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    byte_start=byte_start,
                    byte_end=byte_end,
                ),
                chunk_hash=chunk_hash,
            )
        )
    return chunks
