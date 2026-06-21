"""Tests for session memory."""

from vortexa.storage.session_memory import SessionMemory


def test_record_and_retrieve():
    mem = SessionMemory()
    mem.record_query("find auth middleware")
    mem.record_file_view("src/auth/middleware.py")
    mem.record_symbol_view("AuthMiddleware")

    queries = mem.recent_queries(1)
    assert "find auth middleware" in queries

    files = mem.recent_files(1)
    assert "src/auth/middleware.py" in files

    symbols = mem.recent_symbols(1)
    assert "AuthMiddleware" in symbols


def test_lru_eviction():
    mem = SessionMemory(max_queries=3, max_files=3, max_symbols=3)
    for i in range(5):
        mem.record_query(f"query_{i}")
        mem.record_file_view(f"file_{i}.py")
        mem.record_symbol_view(f"Symbol_{i}")

    assert len(mem.recent_queries(10)) == 3
    assert len(mem.recent_files(10)) == 3
    assert len(mem.recent_symbols(10)) == 3


def test_query_expansion():
    mem = SessionMemory()
    mem.record_query("auth middleware JWT")
    mem.record_query("parse token")

    expanded = mem.query_expansion("token")
    assert "token" in expanded


def test_session_context():
    mem = SessionMemory()
    mem.record_query("test")
    ctx = mem.session_context()
    assert ctx["query_count"] == 1
    assert "duration_seconds" in ctx


def test_save_load(tmp_path):
    mem = SessionMemory()
    mem.record_query("save test")
    mem.save(tmp_path)

    loaded = SessionMemory.load(tmp_path)
    assert "save test" in loaded.recent_queries(5)
