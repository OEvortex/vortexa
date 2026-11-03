"""Indexer regression tests: memoization, BM25 mapping, watcher, progress."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from vortexa.core.indexer import CodebaseIndexer
from vortexa.core.progress import IndexProgress
from vortexa.interfaces.progress import IndexProgressBar
from vortexa.interfaces.watcher import IndexWatcher


class FakeEmbedder:
    """Small deterministic embedder so tests never download a model."""

    dim = 32

    @property
    def memo_key(self) -> tuple:
        return ("FakeEmbedder",)

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower()):
                vectors[row, hash(token) % self.dim] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)


def _write_repo(root: Path) -> None:
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "auth.py").write_text(
        "def validate_jwt(token):\n"
        '    """Validate a JWT token."""\n'
        '    return token.startswith("eyJ")\n'
    )
    (root / "pkg" / "parser.py").write_text(
        "def parse_csv(path):\n"
        "    with open(path) as handle:\n"
        "        return [line.split(',') for line in handle]\n"
    )
    (root / "README.md").write_text("# Demo\nJWT authentication notes.\n")


def _indexer(root: Path, index_dir: Path) -> CodebaseIndexer:
    return CodebaseIndexer(
        root=root,
        model=FakeEmbedder(),
        index_dir=index_dir,
        embed_batch_size=2,
    )


def test_index_stats_search_and_zero_score_filtering(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_repo(root)
    indexer = _indexer(root, tmp_path / "index")

    stats = indexer.index()
    assert stats.indexed_files == 2  # README.md is text-only by default
    assert stats.total_chunks >= 2
    assert stats.languages == {"python": 2}
    assert stats.embedded_chunks == stats.total_chunks
    assert stats.up_to_date is False
    assert indexer.vector_store.size == len(indexer.chunks)

    results = indexer.search("parse_csv", top_k=10)
    assert results
    assert all(result.score > 0 for result in results)
    assert any(result.chunk.file_path == "pkg/parser.py" for result in results)


def test_second_run_is_up_to_date_and_resets_counters(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_repo(root)
    indexer = _indexer(root, tmp_path / "index")
    indexer.index()

    stats = indexer.index()
    assert stats.up_to_date is True
    assert stats.memo_hits == stats.total_chunks
    assert stats.memo_misses == 0
    assert stats.embedded_chunks == 0
    assert stats.changed_files == 0


def test_memoization_reuses_vectors_for_copied_content(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_repo(root)
    indexer = _indexer(root, tmp_path / "index")
    baseline = indexer.index()

    # Copying a file gives every chunk the same content hash as an existing
    # chunk. The copied file must reuse those vectors, not re-embed and not
    # lose the source chunks.
    (root / "pkg" / "auth_copy.py").write_text(
        (root / "pkg" / "auth.py").read_text()
    )
    stats = indexer.index()

    assert stats.memo_hits >= 1
    assert stats.embedded_chunks == 0
    assert stats.total_chunks == baseline.total_chunks + 1
    assert indexer.vector_store.size == len(indexer.chunks)
    assert {chunk.file_path for chunk in indexer.chunks} >= {
        "pkg/auth.py",
        "pkg/auth_copy.py",
    }


def test_edit_and_restore_never_loses_chunks(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_repo(root)
    indexer = _indexer(root, tmp_path / "index")
    baseline = indexer.index()
    auth = root / "pkg" / "auth.py"
    original = auth.read_text()

    auth.write_text("# temporary header\n" + original)
    indexer.index()
    auth.write_text(original)
    stats = indexer.index()

    assert stats.total_chunks == baseline.total_chunks
    assert "pkg/auth.py" in {chunk.file_path for chunk in indexer.chunks}
    assert indexer.vector_store.size == len(indexer.chunks)
    results = indexer.search("validate_jwt", top_k=5)
    assert any(result.chunk.file_path == "pkg/auth.py" for result in results)


def test_bm25_mapping_survives_reload(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_repo(root)
    index_dir = tmp_path / "index"
    _indexer(root, index_dir).index()

    reloaded = _indexer(root, index_dir)
    assert reloaded.load() is True
    assert len(reloaded.chunks) == 2

    results = reloaded.search("parse_csv", top_k=5, alpha=0.0)
    assert results
    assert results[0].chunk.file_path == "pkg/parser.py"


def test_explain_resolves_a_bare_file_path(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_repo(root)
    indexer = _indexer(root, tmp_path / "index")
    indexer.index()

    pack = indexer.explain("pkg/auth.py")
    assert [result.chunk.file_path for result in pack["primary_chunks"]] == [
        "pkg/auth.py"
    ]

    line_pack = indexer.explain("pkg/auth.py:2")
    assert line_pack["primary_chunks"]
    assert all(
        result.chunk.file_path == "pkg/auth.py"
        for result in line_pack["primary_chunks"]
    )


def test_watcher_uses_the_same_extension_set_as_the_index(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_repo(root)
    indexer = _indexer(root, tmp_path / "index")
    indexer.index()

    watcher = IndexWatcher(indexer)
    assert watcher._detect_changes() == set()

    indexer.index(include_text_files=True)
    text_watcher = IndexWatcher(indexer)
    assert text_watcher._detect_changes() == set()


def test_progress_events_cover_all_phases(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_repo(root)
    indexer = _indexer(root, tmp_path / "index")
    events: list[IndexProgress] = []

    stats = indexer.index(progress=events.append)

    phases = [event.phase for event in events]
    assert phases[0] == "scan"
    assert phases[-1] == "done"
    assert {"chunk", "model", "embed", "bm25", "save"} <= set(phases)
    assert events[-1].message
    assert events[-1].elapsed > 0
    assert stats.total_chunks == 2


def test_plain_progress_bar_is_ansi_free_and_reports_completion(tmp_path: Path) -> None:
    from io import StringIO

    stream = StringIO()
    bar = IndexProgressBar(stream=stream, enabled=True, width=80)
    bar(IndexProgress("scan", current=1, total=2, message="Scanning files", detail="a.py"))
    bar(IndexProgress("done", current=1, total=1, message="2 files · 4 chunks", elapsed=0.25))

    output = stream.getvalue()
    assert "Scanning files" in output
    assert "Index ready" in output or "2 files" in output
    assert "\033[" not in output
