"""Agent session memory — tracks recent queries and results for context-aware retrieval.

Stores session state so subsequent queries can benefit from prior context:
- If agent previously looked at function X, boost X on subsequent queries
- Recent queries inform query expansion
- Session state is persisted as JSON for durability
"""

from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SessionMemory:
    """LRU session memory for agent context.

    Tracks:
    - Recent queries and their top results
    - Recently viewed files
    - Recently viewed symbols
    - Session-level metadata (start time, query count)
    """

    def __init__(
        self,
        max_queries: int = 50,
        max_files: int = 100,
        max_symbols: int = 100,
        ttl_seconds: int = 600,  # 10 minute TTL
    ) -> None:
        self._max_queries = max_queries
        self._max_files = max_files
        self._max_symbols = max_symbols
        self._ttl = ttl_seconds

        # OrderedDict for LRU behavior (most recent at end)
        self._queries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._files: OrderedDict[str, float] = OrderedDict()  # file_path -> timestamp
        self._symbols: OrderedDict[str, float] = OrderedDict()  # symbol_name -> timestamp
        self._session_start = time.time()
        self._query_count = 0

    # ── Record operations ─────────────────────────────────────────────────

    def record_query(self, query: str, results: list[dict[str, Any]] | None = None) -> None:
        """Record a query and its results."""
        self._query_count += 1
        self._queries[query] = {
            "timestamp": time.time(),
            "results": results or [],
            "count": self._query_count,
        }
        self._queries.move_to_end(query)
        # Evict oldest if over limit
        while len(self._queries) > self._max_queries:
            self._queries.popitem(last=False)

    def record_file_view(self, file_path: str) -> None:
        """Record that the agent viewed a file."""
        self._files[file_path] = time.time()
        self._files.move_to_end(file_path)
        while len(self._files) > self._max_files:
            self._files.popitem(last=False)

    def record_symbol_view(self, symbol_name: str) -> None:
        """Record that the agent viewed a symbol."""
        self._symbols[symbol_name] = time.time()
        self._symbols.move_to_end(symbol_name)
        while len(self._symbols) > self._max_symbols:
            self._symbols.popitem(last=False)

    # ── Query operations ──────────────────────────────────────────────────

    def recent_queries(self, n: int = 5) -> list[str]:
        """Get the N most recent queries."""
        return list(self._queries.keys())[-n:]

    def recent_files(self, n: int = 10) -> list[str]:
        """Get the N most recently viewed files (within TTL)."""
        now = time.time()
        files = [
            f for f, ts in reversed(self._files.items())
            if now - ts <= self._ttl
        ]
        return files[:n]

    def recent_symbols(self, n: int = 10) -> list[str]:
        """Get the N most recently viewed symbols (within TTL)."""
        now = time.time()
        symbols = [
            s for s, ts in reversed(self._symbols.items())
            if now - ts <= self._ttl
        ]
        return symbols[:n]

    def query_expansion(self, current_query: str) -> str:
        """Expand the current query with context from recent queries.

        If the current query is short/ambiguous, append terms from
        the most recent related query.
        """
        if len(current_query.split()) >= 4:
            return current_query

        recent = self.recent_queries(3)
        if not recent:
            return current_query

        # Find the most semantically related recent query
        # Simple approach: use the most recent non-identical query
        related_terms: list[str] = []
        for q in reversed(recent):
            if q != current_query:
                # Extract key terms (skip stopwords)
                terms = [w for w in q.split() if len(w) > 3]
                related_terms.extend(terms)
            if len(related_terms) >= 3:
                break

        if related_terms:
            expanded = current_query + " " + " ".join(related_terms[:3])
            return expanded.strip()
        return current_query

    def session_context(self) -> dict[str, Any]:
        """Get full session context for debugging or logging."""
        return {
            "duration_seconds": time.time() - self._session_start,
            "query_count": self._query_count,
            "recent_queries": self.recent_queries(5),
            "recent_files": self.recent_files(5),
            "recent_symbols": self.recent_symbols(5),
        }

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, directory: Path) -> None:
        """Persist session memory to disk."""
        directory.mkdir(parents=True, exist_ok=True)
        data = {
            "queries": {k: v for k, v in self._queries.items()},
            "files": {k: v for k, v in self._files.items()},
            "symbols": {k: v for k, v in self._symbols.items()},
            "session_start": self._session_start,
            "query_count": self._query_count,
        }
        (directory / "session.json").write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, directory: Path) -> SessionMemory:
        """Load session memory from disk, or create fresh."""
        path = directory / "session.json"
        if not path.exists():
            return cls()

        try:
            data = json.loads(path.read_text())
            mem = cls()
            mem._queries.update(OrderedDict(data.get("queries", {})))
            mem._files.update(OrderedDict(data.get("files", {})))
            mem._symbols.update(OrderedDict(data.get("symbols", {})))
            mem._session_start = data.get("session_start", time.time())
            mem._query_count = data.get("query_count", 0)
            return mem
        except Exception:
            logger.exception("Failed to load session memory")
            return cls()

    def clear(self) -> None:
        """Clear all session memory."""
        self._queries.clear()
        self._files.clear()
        self._symbols.clear()
        self._session_start = time.time()
        self._query_count = 0
