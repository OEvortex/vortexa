"""Dependency-free realtime rendering of indexing progress.

The renderer writes to stderr so JSON output on stdout and the MCP stdio
protocol are never contaminated. On a TTY it draws a single updating bar; when
stderr is redirected (agents, CI, MCP clients) it emits sparse, plain-text
progress lines instead of ANSI control codes.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from typing import TextIO

from vortexa.core.progress import IndexProgress

_PHASE_LABELS: dict[str, str] = {
    "model": "Loading model",
    "scan": "Scanning files",
    "chunk": "Chunking code",
    "embed": "Embedding chunks",
    "bm25": "Keyword index",
    "save": "Saving index",
    "done": "Index ready",
}

_BAR_CHARS = ("█", "░")
_MIN_WIDTH = 50
_MAX_WIDTH = 120
_RENDER_INTERVAL = 0.08


def _truncate(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def _format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}m {rest:.0f}s"


class IndexProgressBar:
    """Render :class:`IndexProgress` events as a live terminal bar.

    Args:
        stream: Output stream. Defaults to ``sys.stderr``.
        enabled: When false, all progress output is suppressed.
        title: Optional one-time heading, e.g. ``"Indexing /path/to/repo"``.
        width: Override terminal width (useful in tests).
    """

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        enabled: bool = True,
        title: str = "",
        width: int | None = None,
    ) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._enabled = enabled
        self._title = title
        self._isatty = self._stream_isatty()
        self._color = self._isatty and not os.environ.get("NO_COLOR")
        if width is None:
            width = shutil.get_terminal_size((100, 20)).columns
        self._width = max(_MIN_WIDTH, min(width, _MAX_WIDTH))
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._phase: str | None = None
        self._phase_started = self._started
        self._last_total = -1
        self._last_bucket = -1
        self._last_key: tuple | None = None
        self._last_render = 0.0
        self._header_written = False
        self._finished = False

    def _stream_isatty(self) -> bool:
        try:
            return bool(self._stream.isatty())
        except Exception:
            return False

    def _paint(self, text: str, code: str) -> str:
        if not self._color:
            return text
        return f"\033[{code}m{text}\033[0m"

    def __call__(self, event: IndexProgress) -> None:
        if not self._enabled or self._finished:
            return
        with self._lock:
            now = time.monotonic()
            if event.phase == "done":
                self._finish(event, now)
                return

            key = (event.phase, event.current, event.total, event.message, event.detail)
            if key == self._last_key:
                return
            self._last_key = key

            phase_changed = event.phase != self._phase
            total_changed = event.total != self._last_total
            if phase_changed:
                self._phase = event.phase
                self._phase_started = now
                self._last_bucket = -1
            self._last_total = event.total

            self._write_header()

            if self._isatty:
                completed = event.total > 0 and event.current >= event.total
                if (
                    not (phase_changed or total_changed or completed)
                    and now - self._last_render < _RENDER_INTERVAL
                ):
                    return
                self._render_tty(event, now)
                self._last_render = now
                return

            # Plain mode: print on stage change, at each 10% bucket, and when
            # the known total changes. This stays readable in agent logs.
            bucket = event.percent // 10
            if not (phase_changed or total_changed or bucket > self._last_bucket):
                return
            self._last_bucket = bucket
            print(self._render_plain(event, now), file=self._stream, flush=True)

    def _write_header(self) -> None:
        if self._header_written or not self._title:
            return
        self._header_written = True
        title = _truncate(self._title, self._width)
        print(f"[vortexa] {title}", file=self._stream, flush=True)

    def _phase_label(self, event: IndexProgress) -> str:
        return event.message or _PHASE_LABELS.get(event.phase, event.phase.title())

    def _phase_rate(self, event: IndexProgress, now: float) -> tuple[float, float | None]:
        elapsed = max(now - self._phase_started, 1e-6)
        if event.current <= 0:
            return 0.0, None
        rate = event.current / elapsed
        if event.total <= 0 or rate <= 0:
            return rate, None
        return rate, max(0.0, (event.total - event.current) / rate)

    def _render_tty(self, event: IndexProgress, now: float) -> None:
        label = self._phase_label(event)
        current = max(event.current, 0)
        total = max(event.total, 0)
        percent = event.percent if total else 0

        rate, eta = self._phase_rate(event, now)
        status = f"{current}/{total}" if total else "…"
        right = f"{status:>12}  {percent:3d}%"
        if rate > 0:
            right += f"  {rate:5.0f}/s"
        if eta is not None:
            right += f"  ETA {_format_duration(eta)}"

        left = f"[vortexa] {label}"
        detail = _truncate(event.detail, 34)
        if detail:
            left += f"  {detail}"

        bar_width = self._width - len(left) - len(right) - 6
        bar_width = max(10, min(30, bar_width))
        filled = int(bar_width * event.fraction)
        bar = _BAR_CHARS[0] * filled + _BAR_CHARS[1] * (bar_width - filled)

        line = f"{left}  {self._paint(bar, '32')}  {right}"
        self._stream.write("\r\033[2K" + line)
        self._stream.flush()

    def _render_plain(self, event: IndexProgress, now: float) -> str:
        label = self._phase_label(event)
        if event.total > 0:
            head = f"[vortexa] {label}: {event.current}/{event.total} ({event.percent}%)"
        else:
            head = f"[vortexa] {label}"
        _, eta = self._phase_rate(event, now)
        if eta is not None:
            head += f" ETA {_format_duration(eta)}"
        if event.detail:
            head += f" — {_truncate(event.detail, 60)}"
        return head

    def _finish(self, event: IndexProgress, now: float) -> None:
        self._finished = True
        elapsed = event.elapsed or (now - self._started)
        message = _truncate(event.message or "Index complete", max(10, self._width - 24))
        suffix = f" ({_format_duration(elapsed)})"
        line = f"[vortexa] {self._paint('✓', '32')} {message}{suffix}"
        if self._isatty:
            self._stream.write("\r\033[2K" + line + "\n")
        else:
            print(line, file=self._stream, flush=True)
        self._stream.flush()
