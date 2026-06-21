"""Tests for the IndexWatcher's two backends (native FS events + mtime polling)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vortexa.interfaces.watcher import (
    IndexWatcher,
    _HAS_WATCHFILES,
)


def _make_indexer(tmp_path: Path):
    """Build a stub indexer with a real root directory."""
    idx = MagicMock()
    idx.root = tmp_path
    return idx


# ── Backend selection ──────────────────────────────────────────────────────


def test_backend_selection_uses_native_when_watchfiles_available():
    """If watchfiles is importable AND force_polling is False, use native."""
    watcher = IndexWatcher(indexer=MagicMock(), force_polling=False)
    if _HAS_WATCHFILES:
        assert watcher.backend == "native"
    else:
        assert watcher.backend == "polling"


def test_backend_selection_uses_polling_when_force_polling_true():
    watcher = IndexWatcher(indexer=MagicMock(), force_polling=True)
    assert watcher.backend == "polling"


def test_backend_selection_uses_polling_when_env_var_set(tmp_path):
    """VORTEXA_FORCE_POLLING env var overrides the auto-selection."""
    os.environ["VORTEXA_FORCE_POLLING"] = "1"
    try:
        watcher = IndexWatcher(indexer=_make_indexer(tmp_path))
        assert watcher.backend == "polling"
    finally:
        del os.environ["VORTEXA_FORCE_POLLING"]


# ── mtime_ns polling backend ───────────────────────────────────────────────


def test_polling_detects_file_modification(tmp_path):
    """Writing to an existing file should be detected as a change."""
    idx = _make_indexer(tmp_path)
    (tmp_path / "a.py").write_text("original")

    watcher = IndexWatcher(
        indexer=idx, poll_interval=0.1, force_polling=True
    )
    watcher.start()
    try:
        # Wait briefly for the initial snapshot
        time.sleep(0.2)
        # Modify the file — ensure mtime_ns changes by writing twice
        time.sleep(0.05)
        (tmp_path / "a.py").write_text("modified content")

        # Poll once explicitly to bypass the wait
        changed = watcher._detect_changes_mtime()
        assert "a.py" in changed, f"expected a.py in changed, got {changed}"
    finally:
        watcher.stop()


def test_polling_detects_new_file(tmp_path):
    """Adding a new file should be detected."""
    idx = _make_indexer(tmp_path)
    (tmp_path / "existing.py").write_text("x")

    watcher = IndexWatcher(
        indexer=idx, poll_interval=0.1, force_polling=True
    )
    watcher.start()
    try:
        time.sleep(0.3)
        # Add a new file. Sleep briefly so the mtime_ns of the new file
        # is strictly later than the snapshot's last refresh — otherwise
        # the snapshot might already include it on fast filesystems.
        time.sleep(0.1)
        (tmp_path / "new.py").write_text("hello")

        changed = watcher._detect_changes_mtime()
        assert "new.py" in changed, f"expected new.py in {changed}"
    finally:
        watcher.stop()


def test_polling_detects_deleted_file(tmp_path):
    """Removing a file should be detected."""
    idx = _make_indexer(tmp_path)
    (tmp_path / "doomed.py").write_text("x")

    watcher = IndexWatcher(
        indexer=idx, poll_interval=0.1, force_polling=True
    )
    watcher.start()
    try:
        time.sleep(0.2)
        (tmp_path / "doomed.py").unlink()

        changed = watcher._detect_changes_mtime()
        assert "doomed.py" in changed
    finally:
        watcher.stop()


def test_polling_no_change_detected_when_unchanged(tmp_path):
    """Polling twice with no file changes should report zero modifications."""
    idx = _make_indexer(tmp_path)
    (tmp_path / "stable.py").write_text("x")

    watcher = IndexWatcher(
        indexer=idx, poll_interval=0.1, force_polling=True
    )
    watcher.start()
    try:
        time.sleep(0.2)
        # First poll populates snapshot
        first = watcher._detect_changes_mtime()
        # Second poll — nothing changed
        second = watcher._detect_changes_mtime()
        assert second == set(), f"expected no changes, got {second}"
    finally:
        watcher.stop()


# ── Throttling / debounce ─────────────────────────────────────────────────


def test_trigger_reindex_respects_min_interval(tmp_path):
    """trigger_reindex should throttle calls within _MIN_INDEX_INTERVAL."""
    idx = MagicMock()
    idx.root = tmp_path
    idx.index = MagicMock()
    watcher = IndexWatcher(indexer=idx, poll_interval=0.1, force_polling=True)

    # First call should invoke indexer
    watcher._last_index_time = 0.0
    watcher._trigger_reindex({"a.py"})
    assert idx.index.call_count == 1

    # Immediate second call within the throttle window should NOT invoke
    watcher._last_index_time = time.time()
    watcher._trigger_reindex({"b.py"})
    assert idx.index.call_count == 1


def test_refresh_mtime_snapshot_populates_snapshot(tmp_path):
    """refresh_mtime_snapshot should populate _mtime_snapshot for all files."""
    idx = _make_indexer(tmp_path)
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.py").write_text("y")

    watcher = IndexWatcher(indexer=idx, force_polling=True)
    watcher._refresh_mtime_snapshot()
    assert "a.py" in watcher._mtime_snapshot
    assert "b.py" in watcher._mtime_snapshot
    # Each entry is a (mtime_ns, size) tuple
    for rel, sig in watcher._mtime_snapshot.items():
        assert len(sig) == 2
        assert isinstance(sig[0], int)  # mtime_ns
        assert isinstance(sig[1], int)  # size


# ── Lifecycle ─────────────────────────────────────────────────────────────


def test_watcher_start_stop(tmp_path):
    idx = _make_indexer(tmp_path)
    watcher = IndexWatcher(indexer=idx, force_polling=True)
    assert not watcher.is_running

    watcher.start()
    assert watcher.is_running

    watcher.stop()
    # Allow the thread to join
    time.sleep(0.2)
    assert not watcher.is_running


def test_watcher_double_start_is_noop(tmp_path):
    idx = _make_indexer(tmp_path)
    watcher = IndexWatcher(indexer=idx, force_polling=True)
    watcher.start()
    first_thread = watcher._thread
    watcher.start()  # should be no-op
    assert watcher._thread is first_thread
    watcher.stop()


@pytest.mark.skipif(not _HAS_WATCHFILES, reason="watchfiles not installed")
def test_native_backend_smoke(tmp_path):
    """Smoke test: the native backend actually starts and stops cleanly."""
    (tmp_path / "a.py").write_text("x")
    idx = _make_indexer(tmp_path)
    watcher = IndexWatcher(indexer=idx, poll_interval=0.5)
    assert watcher.backend == "native"
    watcher.start()
    time.sleep(0.3)
    watcher.stop()