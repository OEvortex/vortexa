"""Live watch mode for the codebase indexer.

Watches a directory for file changes and automatically re-indexes.
Inspired by cocoindex's live mode with debounced batch updates.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vortexa.core.indexer import CodebaseIndexer

logger = logging.getLogger(__name__)

# Debounce window: batch changes within this period before re-indexing
_DEBOUNCE_SECONDS = 2.0
# Minimum time between index runs
_MIN_INDEX_INTERVAL = 5.0


class IndexWatcher:
    """Watches a directory for file changes and auto-reindexes.

    Uses a polling approach (check file hashes periodically) for cross-platform
    compatibility. Debounces changes to avoid excessive re-indexing.
    """

    def __init__(
        self,
        indexer: CodebaseIndexer,
        poll_interval: float = 3.0,
        include_text_files: bool | None = None,
    ) -> None:
        self._indexer = indexer
        self._poll_interval = poll_interval
        self._include_text_files = (
            indexer.include_text_files
            if include_text_files is None
            else include_text_files
        )
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_index_time = 0.0
        self._pending_changes: set[str] = set()
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        """Whether the watcher is currently running."""
        return self._running

    @property
    def include_text_files(self) -> bool:
        """File extensions the watcher and indexer should share."""
        return self._include_text_files

    def start(self) -> None:
        """Start watching for file changes in a background thread."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info("Index watcher started for %s", self._indexer.root)

    def stop(self) -> None:
        """Stop watching for file changes."""
        self._running = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval + 1)
            self._thread = None
        logger.info("Index watcher stopped")

    def _watch_loop(self) -> None:
        """Main watch loop: poll for changes and re-index."""
        while self._running:
            if self._stop_event.wait(self._poll_interval):
                break
            if not self._running:
                break

            try:
                changed = self._detect_changes()
                if not changed:
                    continue

                with self._lock:
                    self._pending_changes.update(changed)

                # Debounce: wait for more changes before touching the index.
                if self._stop_event.wait(_DEBOUNCE_SECONDS):
                    break

                now = time.time()
                if now - self._last_index_time < _MIN_INDEX_INTERVAL:
                    continue
                if self._indexer.is_indexing:
                    continue

                with self._lock:
                    pending = self._pending_changes.copy()
                    self._pending_changes.clear()
                if not pending:
                    continue

                logger.info("Auto-reindexing: %d files changed", len(pending))
                # Keep the watcher's extension set in sync with the indexer;
                # this is what prevents text files from being "changed"
                # forever when they are not part of the index.
                self._include_text_files = self._indexer.include_text_files
                self._indexer.index(include_text_files=self._include_text_files)
                self._last_index_time = time.time()
            except Exception:
                logger.exception("Error in watch loop")

    def _detect_changes(self) -> set[str]:
        """Detect files that have changed since last index.

        Uses the exact same extension set as the last index run, so a normal
        code-only index does not repeatedly see Markdown/JSON files as pending.
        """
        from vortexa.core.language import get_extensions
        from vortexa.storage.walker import walk_files

        self._include_text_files = self._indexer.include_text_files
        extensions = get_extensions(self._include_text_files)
        current_hashes: dict[str, str] = {}

        for file_path in walk_files(self._indexer.root, extensions):
            try:
                if file_path.stat().st_size > 1_000_000:
                    continue
                rel = str(file_path.relative_to(self._indexer.root))
                current_hashes[rel] = self._file_hash(file_path)
            except OSError:
                continue

        changed: set[str] = set()
        stored = self._indexer.file_hashes

        for rel, digest in current_hashes.items():
            if stored.get(rel) != digest:
                changed.add(rel)

        for rel in stored:
            if rel not in current_hashes:
                changed.add(rel)

        return changed

    @staticmethod
    def _file_hash(path: Path) -> str:
        """Compute SHA256 hash of file contents."""
        import hashlib

        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()
