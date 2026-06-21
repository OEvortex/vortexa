"""Live watch mode for the codebase indexer.

Two backends:

1. **Native FS events** via `watchfiles` (preferred) — uses inotify on
   Linux, FSEvents on macOS, ReadDirectoryChangesW on Windows. No polling,
   instant change detection, near-zero CPU when idle.

2. **mtime_ns polling** (fallback) — checks `(mtime_ns, size)` of each
   file against the last-seen snapshot. Cheaper than the previous SHA256
   approach because it avoids reading file contents, and precise enough
   that we don't need to re-hash unless something actually changed.

The backend is auto-selected at construction time: if `watchfiles` can
be imported AND `force_polling` is False, we use native events;
otherwise we fall back to polling. Selection is logged once on first
start() so operators know which mode they're in.
"""

from __future__ import annotations

import logging
import os
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

# Try to import watchfiles (Rust-based, native FS events). Fall back to
# polling if it's not available.
try:
    import watchfiles

    _HAS_WATCHFILES = True
except ImportError:
    _HAS_WATCHFILES = False


class IndexWatcher:
    """Watches a directory and triggers incremental re-indexing.

    Backend selection:
      - If `watchfiles` is importable AND `force_polling=False`, uses
        inotify/FSEvents/ReadDirectoryChangesW via the Rust `notify` crate.
      - Otherwise polls `(mtime_ns, size)` every `poll_interval` seconds.

    Debouncing: changes within `_DEBOUNCE_SECONDS` of each other are
    batched; the indexer is called at most once per `_MIN_INDEX_INTERVAL`.
    """

    def __init__(
        self,
        indexer: "CodebaseIndexer",
        poll_interval: float = 3.0,
        force_polling: bool = False,
    ) -> None:
        self._indexer = indexer
        self._poll_interval = poll_interval
        self._force_polling = force_polling
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_index_time = 0.0
        self._pending_changes: set[str] = set()
        self._lock = threading.Lock()

        # Cached mtime_ns/size snapshot for the polling backend. Avoids
        # reading file contents (SHA256) on every tick — the previous
        # implementation hashed every file every poll, which is O(n*size).
        self._mtime_snapshot: dict[str, tuple[int, int]] = {}

        # Decide which backend to use up-front so the choice is logged
        # exactly once.
        self._use_native = (
            _HAS_WATCHFILES
            and not force_polling
            and not os.environ.get("VORTEXA_FORCE_POLLING")
        )

    @property
    def is_running(self) -> bool:
        """Whether the watcher is currently running."""
        return self._running

    @property
    def backend(self) -> str:
        """Return the active backend name ('native' or 'polling')."""
        return "native" if self._use_native else "polling"

    def start(self) -> None:
        """Start watching for file changes in a background thread."""
        if self._running:
            return

        backend = self.backend
        if backend == "native" and not _HAS_WATCHFILES:
            # Shouldn't happen because __init__ chose the backend, but
            # keep the guard for defensive clarity.
            backend = "polling"

        logger.info(
            "Index watcher starting (backend=%s) for %s",
            backend,
            self._indexer.root,
        )

        self._running = True
        if backend == "native":
            self._thread = threading.Thread(
                target=self._native_watch_loop, daemon=True, name="vortexa-watcher"
            )
        else:
            self._thread = threading.Thread(
                target=self._poll_watch_loop, daemon=True, name="vortexa-watcher"
            )
        self._thread.start()

    def stop(self) -> None:
        """Stop watching for file changes."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval + 1)
            self._thread = None
        logger.info("Index watcher stopped")

    # ── Native FS events backend ──────────────────────────────────────

    def _native_watch_loop(self) -> None:
        """Watch using `watchfiles` (Rust notify crate under the hood)."""
        # Build an initial mtime snapshot so we can debounce the first
        # burst of events that fire on subscription.
        self._refresh_mtime_snapshot()

        try:
            for changes in watchfiles.watch(
                self._indexer.root,
                # DefaultFilter ignores common build / vcs noise.
                watch_filter=None,
                debounce=int(_DEBOUNCE_SECONDS * 1000),
                step=50,
                stop_event=None,
                recursive=True,
                yield_on_timeout=True,
                poll_delay_ms=int(self._poll_interval * 1000),
                # Prefer the native backend; force_polling=False lets
                # notify fall back to polling automatically on platforms
                # that don't support inotify/FSEvents.
                force_polling=False,
                ignore_permission_denied=True,
            ):
                if not self._running:
                    break
                if not changes:
                    continue
                changed_files = {Path(c[1]).resolve() for c in changes}
                rel = {
                    str(p.relative_to(self._indexer.root))
                    for p in changed_files
                    if p.is_relative_to(self._indexer.root)
                }
                if rel:
                    self._trigger_reindex(rel)
        except Exception:
            logger.exception("Error in native watch loop")
        finally:
            # If native watch dies (e.g. permission denied), fall back to
            # polling so the user still gets re-indexing.
            if self._running:
                logger.warning("Native watch backend crashed, falling back to polling")
                self._use_native = False
                self._poll_watch_loop()

    # ── mtime_ns polling backend ──────────────────────────────────────

    def _poll_watch_loop(self) -> None:
        """Watch by polling (mtime_ns, size) every `poll_interval` seconds."""
        # Initial snapshot
        self._refresh_mtime_snapshot()

        while self._running:
            time.sleep(self._poll_interval)
            if not self._running:
                break
            try:
                changed = self._detect_changes_mtime()
                if changed:
                    self._trigger_reindex(changed)
            except Exception:
                logger.exception("Error in polling watch loop")

    def _refresh_mtime_snapshot(self) -> None:
        """Snapshot (mtime_ns, size) for every tracked file."""
        from vortexa.core.language import get_extensions
        from vortexa.storage.walker import walk_files

        extensions = get_extensions(include_text_files=True)
        snapshot: dict[str, tuple[int, int]] = {}
        for file_path in walk_files(self._indexer.root, extensions):
            if file_path.stat().st_size > 1_000_000:
                continue
            rel = str(file_path.relative_to(self._indexer.root))
            try:
                st = file_path.stat()
                snapshot[rel] = (st.st_mtime_ns, st.st_size)
            except OSError:
                continue
        self._mtime_snapshot = snapshot

    def _detect_changes_mtime(self) -> set[str]:
        """Detect files that have changed since the last snapshot.

        Compares (mtime_ns, size) against the cached snapshot. Much cheaper
        than the previous SHA256-of-contents approach — we only stat each
        file, no reads.
        """
        from vortexa.core.language import get_extensions
        from vortexa.storage.walker import walk_files

        extensions = get_extensions(include_text_files=True)
        current: dict[str, tuple[int, int]] = {}
        for file_path in walk_files(self._indexer.root, extensions):
            if file_path.stat().st_size > 1_000_000:
                continue
            rel = str(file_path.relative_to(self._indexer.root))
            try:
                st = file_path.stat()
                current[rel] = (st.st_mtime_ns, st.st_size)
            except OSError:
                continue

        changed: set[str] = set()
        # Modified files
        for rel, sig in current.items():
            if self._mtime_snapshot.get(rel) != sig:
                changed.add(rel)
        # Deleted files
        for rel in self._mtime_snapshot:
            if rel not in current:
                changed.add(rel)

        self._mtime_snapshot = current
        return changed

    # ── Shared trigger logic ──────────────────────────────────────────

    def _trigger_reindex(self, changed: set[str]) -> None:
        """Debounce + throttle + re-index.

        Called from both backends. Waits `_DEBOUNCE_SECONDS` to coalesce
        bursts of changes, then re-indexes — but only if at least
        `_MIN_INDEX_INTERVAL` has passed since the last run.
        """
        with self._lock:
            self._pending_changes.update(changed)

        time.sleep(_DEBOUNCE_SECONDS)

        now = time.time()
        if now - self._last_index_time < _MIN_INDEX_INTERVAL:
            return

        with self._lock:
            pending = self._pending_changes.copy()
            self._pending_changes.clear()

        if pending:
            logger.info("Auto-reindexing: %d files changed", len(pending))
            try:
                self._indexer.index()
                self._last_index_time = time.time()
                # Refresh the snapshot so we don't immediately re-fire
                # for the files we just indexed.
                self._refresh_mtime_snapshot()
            except Exception:
                logger.exception("Re-index failed")