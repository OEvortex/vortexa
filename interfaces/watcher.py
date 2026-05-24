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

    def __init__(self, indexer: CodebaseIndexer, poll_interval: float = 3.0) -> None:
        self._indexer = indexer
        self._poll_interval = poll_interval
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_index_time = 0.0
        self._pending_changes: set[str] = set()
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        """Whether the watcher is currently running."""
        return self._running

    def start(self) -> None:
        """Start watching for file changes in a background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info("Index watcher started for %s", self._indexer.root)

    def stop(self) -> None:
        """Stop watching for file changes."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval + 1)
            self._thread = None
        logger.info("Index watcher stopped")

    def _watch_loop(self) -> None:
        """Main watch loop: poll for changes and re-index."""
        while self._running:
            time.sleep(self._poll_interval)

            if not self._running:
                break

            # Check for changes
            try:
                changed = self._detect_changes()
                if changed:
                    with self._lock:
                        self._pending_changes.update(changed)

                    # Debounce: wait for more changes
                    time.sleep(_DEBOUNCE_SECONDS)

                    # Check if enough time has passed since last index
                    now = time.time()
                    if now - self._last_index_time >= _MIN_INDEX_INTERVAL:
                        with self._lock:
                            pending = self._pending_changes.copy()
                            self._pending_changes.clear()

                        if pending:
                            logger.info("Auto-reindexing: %d files changed", len(pending))
                            self._indexer.index()
                            self._last_index_time = time.time()
            except Exception:
                logger.exception("Error in watch loop")

    def _detect_changes(self) -> set[str]:
        """Detect files that have changed since last index.

        Compares current file hashes with stored hashes.
        """
        from vortexa.core.language import get_extensions
        from vortexa.storage.walker import walk_files

        extensions = get_extensions(include_text_files=True)
        current_hashes: dict[str, str] = {}

        for file_path in walk_files(self._indexer.root, extensions):
            if file_path.stat().st_size > 1_000_000:
                continue
            rel = str(file_path.relative_to(self._indexer.root))
            try:
                current_hashes[rel] = self._file_hash(file_path)
            except OSError:
                continue

        # Compare with stored hashes
        changed: set[str] = set()
        stored = self._indexer.file_hashes

        for rel, h in current_hashes.items():
            if stored.get(rel) != h:
                changed.add(rel)

        # Detect deleted files
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
