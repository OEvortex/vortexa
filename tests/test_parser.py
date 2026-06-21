"""Tests for the code parser module."""

from vortexa.core.parser import parse_symbols, parse_docstring
from vortexa.core.types import SymbolKind


def test_parse_python_function():
    source = """
def hello(name: str) -> str:
    \"\"\"Say hello to someone.\"\"\"
    return f"Hello {name}"
"""
    symbols, imports = parse_symbols(source, "test.py", "python")
    assert len(symbols) >= 1
    func = symbols[0]
    assert func.name == "hello"
    assert func.kind == SymbolKind.FUNCTION


def test_parse_python_class():
    source = """
class MyService:
    \"\"\"A service class.\"\"\"

    def process(self):
        pass
"""
    symbols, imports = parse_symbols(source, "test.py", "python")
    classes = [s for s in symbols if s.kind == SymbolKind.CLASS]
    assert len(classes) >= 1
    assert classes[0].name == "MyService"


def test_parse_python_imports():
    source = """
import os
from pathlib import Path
import numpy as np
"""
    symbols, imports = parse_symbols(source, "test.py", "python")
    assert len(imports) >= 2
    modules = {i.imported_module for i in imports}
    assert "os" in modules
    assert "pathlib" in modules or "numpy" in modules


def test_parse_docstring():
    source = """
def foo():
    \"\"\"This is a docstring.\"\"\"
    pass
"""
    doc = parse_docstring(source, 2, 4)
    assert doc is not None
    assert "docstring" in doc


def test_parse_empty_source():
    symbols, imports = parse_symbols("", "empty.py", "python")
    assert len(symbols) == 0
    assert len(imports) == 0


def test_parse_no_language_fallback():
    source = """
def fallback_func():
    pass
"""
    symbols, imports = parse_symbols(source, "test.xyz", None)
    assert len(symbols) >= 1
    assert symbols[0].name == "fallback_func"
