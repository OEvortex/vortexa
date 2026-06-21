"""Tests for the multilingual tree-sitter parser.

Covers the per-language symbol + import extraction across the 13
languages that have first-class support: Python, JavaScript, TypeScript,
Go, Rust, Java, Ruby, Swift, C#, Kotlin, Haskell, Elixir, Scala.
"""

from __future__ import annotations

from vortexa.core.parser import parse_symbols, get_supported_languages


def test_supported_languages_nonempty():
    langs = get_supported_languages()
    assert len(langs) >= 13
    # Spot-check that every "expected" tier-1 language is there
    for lang in ("python", "javascript", "typescript", "go", "rust",
                 "java", "ruby", "swift", "csharp", "kotlin",
                 "haskell", "elixir", "scala"):
        assert lang in langs, f"missing supported language: {lang}"


# ── Per-language extraction tests ─────────────────────────────────────────


def test_python_function_and_class():
    syms, imps = parse_symbols(
        "def foo(): pass\nclass Bar: pass",
        "test.py", "python",
    )
    names = [(s.name, s.kind.value) for s in syms]
    assert ("foo", "function") in names
    assert ("Bar", "class") in names


def test_python_from_import_with_wildcard():
    syms, imps = parse_symbols(
        "from mod import a, b\nimport os.path\nfrom x import *",
        "test.py", "python",
    )
    modules = [i.imported_module for i in imps]
    assert "mod" in modules
    assert "os.path" in modules
    assert "x" in modules
    # Wildcard should be detected
    wildcard = next(i for i in imps if i.imported_module == "x")
    assert "*" in wildcard.imported_names


def test_python_module_name_not_in_names():
    """from x import y should NOT include 'x' in imported_names."""
    _, imps = parse_symbols("from foo import bar", "test.py", "python")
    foo = next(i for i in imps if i.imported_module == "foo")
    assert foo.imported_names == ["bar"], f"got {foo.imported_names}"


def test_javascript_class_and_import_default():
    syms, imps = parse_symbols(
        'function greet() {}\nclass Animal {}\nimport x from "y";',
        "test.js", "javascript",
    )
    names = [(s.name, s.kind.value) for s in syms]
    assert ("greet", "function") in names
    assert ("Animal", "class") in names
    assert any(i.imported_module == "y" and i.imported_names == ["x"] for i in imps)


def test_typescript_interface_and_named_imports():
    syms, imps = parse_symbols(
        'function foo(): void {}\ninterface IBar {}\nimport { a, b as c } from "mod";',
        "test.ts", "typescript",
    )
    names = [(s.name, s.kind.value) for s in syms]
    assert ("foo", "function") in names
    # Interface should be detected as INTERFACE kind, not CLASS
    kinds = {s.kind.value for s in syms}
    assert "interface" in kinds
    # Named imports
    assert any(i.imported_module == "mod" and set(i.imported_names) == {"a", "c"}
               for i in imps)


def test_go_function_and_type_declaration():
    syms, imps = parse_symbols(
        'package main\nfunc Hello() {}\ntype Foo struct {}\nimport "fmt"',
        "test.go", "go",
    )
    names = [(s.name, s.kind.value) for s in syms]
    assert ("Hello", "function") in names
    assert ("Foo", "class") in names  # type X struct surfaces as class
    assert any(i.imported_module == "fmt" for i in imps)


def test_rust_struct_trait_and_use():
    syms, imps = parse_symbols(
        'fn main() {}\nstruct S {}\ntrait T {}\nuse std::io::Write;',
        "test.rs", "rust",
    )
    names = [(s.name, s.kind.value) for s in syms]
    assert ("main", "function") in names
    assert ("S", "class") in names
    assert ("T", "class") in names
    assert any("std::io::Write" in i.imported_module for i in imps)


def test_java_class_method_and_import():
    syms, imps = parse_symbols(
        "class Foo {\n  void bar() {}\n  static void baz() {}\n}\nimport java.util.List;",
        "test.java", "java",
    )
    names = [(s.name, s.kind.value, s.parent) for s in syms]
    assert ("Foo", "class", None) in names
    assert ("bar", "method", "Foo") in names
    assert ("baz", "method", "Foo") in names
    assert any(i.imported_module == "java.util.List" for i in imps)


def test_ruby_method_class_require():
    syms, imps = parse_symbols(
        'def foo; end\nclass Bar; end\nrequire "json"',
        "test.rb", "ruby",
    )
    names = [(s.name, s.kind.value) for s in syms]
    assert ("foo", "function") in names
    assert ("Bar", "class") in names
    assert any(i.imported_module == "json" for i in imps)


def test_swift_function_class_import():
    syms, imps = parse_symbols(
        "func greet() {}\nclass Animal {}\nimport Foundation",
        "test.swift", "swift",
    )
    names = [(s.name, s.kind.value) for s in syms]
    assert ("greet", "function") in names
    assert ("Animal", "class") in names
    assert any(i.imported_module == "Foundation" for i in imps)


def test_csharp_class_method_using():
    syms, imps = parse_symbols(
        "class Foo { void Bar() {} }\nusing System;",
        "test.cs", "csharp",
    )
    names = [(s.name, s.kind.value, s.parent) for s in syms]
    assert ("Foo", "class", None) in names
    assert ("Bar", "method", "Foo") in names
    assert any(i.imported_module == "System" for i in imps)


def test_kotlin_function_class_and_import():
    syms, imps = parse_symbols(
        "fun greet() {}\nclass Animal\nimport kotlin.io.println",
        "test.kt", "kotlin",
    )
    names = [(s.name, s.kind.value) for s in syms]
    assert ("greet", "function") in names
    assert ("Animal", "class") in names
    assert any(i.imported_module == "kotlin.io.println" for i in imps)


def test_haskell_function_and_data():
    """Haskell: function definition + data type both extracted."""
    syms, imps = parse_symbols(
        "foo :: Int -> Int\nfoo x = x + 1\ndata Bar = Bar Int",
        "test.hs", "haskell",
    )
    names = [(s.name, s.kind.value) for s in syms]
    assert ("foo", "function") in names
    assert ("Bar", "class") in names
    # And the signature shouldn't produce a duplicate foo
    assert sum(1 for n in names if n[0] == "foo") == 1


def test_elixir_defmodule_and_def():
    """Elixir: defmodule name comes from the argument, not the keyword."""
    syms, imps = parse_symbols(
        "defmodule MyApp do\n  def hello, do: :world\n  defp secret, do: 42\nend",
        "test.ex", "elixir",
    )
    names = [(s.name, s.kind.value, s.parent) for s in syms]
    # The module name should be MyApp (the argument), not defmodule
    assert ("MyApp", "class", None) in names
    assert ("hello", "method", "MyApp") in names
    assert ("secret", "method", "MyApp") in names


def test_scala_function_class_import():
    syms, imps = parse_symbols(
        "def foo = 1\nclass Bar\nimport scala.collection.Map",
        "test.scala", "scala",
    )
    names = [(s.name, s.kind.value) for s in syms]
    assert ("foo", "function") in names
    assert ("Bar", "class") in names
    assert any(i.imported_module == "scala.collection.Map" for i in imps)


# ── Robustness / regression tests ────────────────────────────────────────


def test_empty_source_returns_empty():
    syms, imps = parse_symbols("", "test.py", "python")
    assert syms == []
    assert imps == []


def test_unknown_language_falls_back_to_regex():
    """Unknown language should still produce something via the regex path."""
    syms, imps = parse_symbols(
        "def my_func():\n    pass\nclass MyClass:\n    pass",
        "test.unknown_ext", None,
    )
    # At minimum, regex catches function defs
    assert any(s.name == "my_func" for s in syms)


def test_docstring_extracted_for_python_class():
    syms, _ = parse_symbols(
        'class Foo:\n    """Class docstring."""\n    pass',
        "test.py", "python",
    )
    foo = next(s for s in syms if s.name == "Foo")
    assert foo.docstring == "Class docstring."


def test_no_duplicates_for_same_definition():
    """Same symbol extracted twice (e.g. nested AST walks) → dedupe."""
    syms, _ = parse_symbols(
        "class Foo:\n    def bar(self): pass",
        "test.py", "python",
    )
    foo_names = [s for s in syms if s.name == "Foo"]
    assert len(foo_names) == 1


def test_signature_truncated_to_200_chars():
    """Long signatures should be truncated, not dropped."""
    long = "x" * 500
    syms, _ = parse_symbols(f"def {long}(): pass", "test.py", "python")
    assert syms
    assert syms[0].signature
    assert len(syms[0].signature) <= 200