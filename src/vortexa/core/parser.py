"""Multilingual code parser using tree-sitter-language-pack.

Extracts symbols (functions, methods, classes, interfaces, structs, enums,
traits, types) and imports from source code via tree-sitter ASTs. Falls
back to regex patterns for languages the tree-sitter grammar can't
handle (rare in 2025).

Replaces the earlier implementation which silently fell back to regex for
every language because it used the wrong tree-sitter API (passed `bytes`
to `parse()` which expects `str`, and used `.type`/`.children` instead
of `.kind`/`.child(i)` that the bundled tree-sitter binding exposes).

Supported languages (35+):
    Python, JavaScript, TypeScript, TSX, JSX, Go, Rust, Java, Kotlin,
    Ruby, Swift, C, C++, C#, Scala, PHP, Lua, Elixir, Erlang, Dart,
    Zig, Julia, Haskell, SQL, HTML, CSS, SCSS/LESS, Vue, Svelte, Astro,
    Markdown, YAML, JSON, TOML, Bash/Zsh/Fish, Dockerfile, GraphQL,
    Solidity.

Design:
    - AST-first: tree-sitter walks the parse tree for accurate bounds and
      parent context (method vs function, struct vs interface, etc.)
    - Regex fallback: only for grammars we can't acquire at import time.
    - Per-language node-type maps so each grammar's idioms are honoured
      (`impl_item` in Rust, `method_declaration` in Java, `arrow_function`
      in JS/TS, etc.).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from vortexa.core.types import ImportInfo, SymbolInfo, SymbolKind

logger = logging.getLogger(__name__)

# ── Per-language tree-sitter node-type maps ─────────────────────────────────
#
# Keys are the values returned by `node.kind` (a string). Values are
# categorised by what kind of code construct the node represents.
#
# A node type is treated as a definition when it appears in the appropriate
# set for the language being parsed. Multi-language grammars (TSX/JSX, etc.)
# inherit from their parent and add their own node types.


# Class-like nodes (produce SymbolKind.CLASS)
CLASS_NODE_TYPES: dict[str, frozenset[str]] = {
    "python":     frozenset({"class_definition"}),
    "javascript": frozenset({"class_declaration", "class"}),
    "jsx":        frozenset({"class_declaration", "class"}),
    "typescript": frozenset({"class_declaration", "abstract_class_declaration"}),
    "tsx":        frozenset({"class_declaration", "abstract_class_declaration"}),
    "go":         frozenset({"type_declaration", "type_spec"}),  # type X struct
    "rust":       frozenset({"struct_item", "union_item", "trait_item"}),
    "java":       frozenset({"class_declaration", "record_declaration", "enum_declaration"}),
    "kotlin":     frozenset({"class_declaration", "object_declaration"}),
    "ruby":       frozenset({"class", "module", "singleton_class"}),
    "swift":      frozenset({"class_declaration"}),
    "c":          frozenset({"struct_specifier", "union_specifier", "enum_specifier"}),
    "cpp":        frozenset({"class_specifier", "struct_specifier", "union_specifier"}),
    "csharp":     frozenset({"class_declaration", "struct_declaration", "record_declaration"}),
    "scala":      frozenset({"class_definition", "object_definition", "trait_definition"}),
    "php":        frozenset({"class_declaration", "interface_declaration", "trait_declaration"}),
    "lua":        frozenset(),
    "elixir":     frozenset({"defmodule", "defprotocol", "defimpl"}),
    "erlang":     frozenset(),
    "dart":       frozenset({"class_definition", "mixin_declaration", "extension_declaration"}),
    "zig":        frozenset(),
    "julia":      frozenset({"struct_definition"}),
    "haskell":    frozenset({"data_type", "newtype", "class_declaration", "type_family_declaration"}),
    "sql":        frozenset(),
    "html":       frozenset(),
    "css":        frozenset(),
    "scss":       frozenset(),
    "less":       frozenset(),
    "vue":        frozenset(),
    "svelte":     frozenset(),
    "astro":      frozenset(),
    "markdown":   frozenset(),
    "yaml":       frozenset(),
    "json":       frozenset(),
    "toml":       frozenset(),
    "bash":       frozenset(),
    "zsh":        frozenset(),
    "fish":       frozenset(),
    "dockerfile": frozenset(),
    "graphql":    frozenset({"object_type_definition", "interface_type_definition"}),
    "solidity":   frozenset({"contract_declaration", "library_declaration"}),
}

# Interface-like nodes (produce SymbolKind.INTERFACE)
INTERFACE_NODE_TYPES: dict[str, frozenset[str]] = {
    "java":       frozenset({"interface_declaration", "annotation_type_declaration"}),
    "typescript": frozenset({"interface_declaration"}),
    "tsx":        frozenset({"interface_declaration"}),
    "go":         frozenset({"type_declaration"}),  # type X interface is interface-like
    "rust":       frozenset({"trait_item"}),
    "kotlin":     frozenset({"interface_declaration"}),
    "csharp":     frozenset({"interface_declaration"}),
    "scala":      frozenset({"trait_definition"}),
    "php":        frozenset({"interface_declaration"}),
    "graphql":    frozenset({"interface_type_definition"}),
    "haskell":    frozenset({"class_declaration"}),  # typeclasses are interface-like
}

# Function-like nodes (produce SymbolKind.FUNCTION)
FUNCTION_NODE_TYPES: dict[str, frozenset[str]] = {
    "python":     frozenset({"function_definition"}),
    "javascript": frozenset({"function_declaration", "function", "generator_function_declaration"}),
    "jsx":        frozenset({"function_declaration", "function", "generator_function_declaration"}),
    "typescript": frozenset({"function_declaration", "generator_function_declaration"}),
    "tsx":        frozenset({"function_declaration", "generator_function_declaration"}),
    "go":         frozenset({"function_declaration", "method_declaration", "func_literal"}),
    "rust":       frozenset({"function_item"}),
    "java":       frozenset({"method_declaration", "constructor_declaration"}),
    "kotlin":     frozenset({"function_declaration", "getter", "setter", "secondary_constructor"}),
    "ruby":       frozenset({"method", "singleton_method"}),
    "swift":      frozenset({"function_declaration"}),
    "c":          frozenset({"function_definition"}),
    "cpp":        frozenset({"function_definition", "function_declarator"}),
    "csharp":     frozenset({"method_declaration", "constructor_declaration"}),
    "scala":      frozenset({"function_definition", "function_declaration"}),
    "php":        frozenset({"function_definition", "method_declaration"}),
    "lua":        frozenset({"function_declaration"}),
    "elixir":     frozenset(),  # call nodes handled by visit()'s elixir special-case
    "erlang":     frozenset({"function_clause"}),
    "dart":       frozenset({"function_signature", "method_signature", "method_declaration", "function_declaration"}),
    "zig":        frozenset({"function_declaration"}),
    "julia":      frozenset({"function_definition"}),
    "haskell":    frozenset({"function_declaration", "function_definition", "function", "signature"}),
    "sql":        frozenset(),
    "graphql":    frozenset(),
    "solidity":   frozenset({"function_definition"}),
}

# Method nodes — explicit method kinds (vs free functions)
METHOD_NODE_TYPES: dict[str, frozenset[str]] = {
    "python":     frozenset(),  # Python uses FUNCTION_NODE_TYPES + parent context
    "javascript": frozenset({"method_definition"}),
    "jsx":        frozenset({"method_definition"}),
    "typescript": frozenset({"method_definition"}),
    "tsx":        frozenset({"method_definition"}),
    "go":         frozenset({"method_declaration"}),
    "rust":       frozenset(),  # method_item is inside impl_item — handled via parent
    "java":       frozenset({"method_declaration"}),
    "kotlin":     frozenset(),
    "ruby":       frozenset(),
    "swift":      frozenset(),
    "csharp":     frozenset({"method_declaration"}),
    "scala":      frozenset(),
    "php":        frozenset({"method_declaration"}),
    "dart":       frozenset({"method_signature", "method_declaration"}),
}

# Import-like nodes
IMPORT_NODE_TYPES: dict[str, frozenset[str]] = {
    "python":     frozenset({"import_statement", "import_from_statement", "future_import_statement"}),
    "javascript": frozenset({"import_statement"}),
    "jsx":        frozenset({"import_statement"}),
    "typescript": frozenset({"import_statement", "export_statement"}),
    "tsx":        frozenset({"import_statement"}),
    "go":         frozenset({"import_declaration"}),
    "rust":       frozenset({"use_declaration", "extern_crate_declaration"}),
    "java":       frozenset({"import_declaration"}),
    "kotlin":     frozenset({"import_header", "infix_expression"}),  # infix fallback for grammar ambiguity
    "ruby":       frozenset({"call", "method_call"}),  # require 'x' is a call
    "swift":      frozenset({"import_declaration"}),
    "csharp":     frozenset({"using_directive"}),
    "scala":      frozenset({"import_declaration"}),
    "php":        frozenset({"namespace_use_declaration"}),
    "lua":        frozenset({"function_call"}),  # require 'x'
    "elixir":     frozenset({"call"}),  # import Foo
    "erlang":     frozenset({"attribute"}),  # -import(Mod, [fun/arity]).
    "dart":       frozenset({"import_specification", "library_import"}),
    "haskell":    frozenset({"import"}),
    "solidity":   frozenset({"import_directive"}),
}

# ── Node-iteration helpers (work with the bundled tree-sitter binding) ──────
#
# The bundled tree-sitter binding (shipped with tree-sitter-language-pack)
# exposes every node attribute as a bound method that must be called with
# `()` — accessing them as properties returns the method object instead of
# the value. We normalize that here so the rest of the parser can use a
# uniform, attribute-style interface.
#
# Methods that take arguments (child, named_child, child_by_field_name)
# have dedicated wrappers — `_call`'s no-arg assumption would otherwise
# call them with no args and crash.


def _kind(node: Any) -> str:
    """Get the node kind (e.g. 'function_declaration')."""
    return str(node.kind() or "")


def _child(node: Any, i: int) -> Any | None:
    """Return the i-th child of `node`, or None if out of range."""
    count = node.child_count() or 0
    if 0 <= i < count:
        return node.child(i)
    return None


def _named_children(node: Any) -> list[Any]:
    """Return the named children of `node` as a list."""
    count = node.named_child_count() or 0
    out: list[Any] = []
    for i in range(count):
        c = node.named_child(i)
        if c is not None:
            out.append(c)
    return out


def _children(node: Any) -> list[Any]:
    """Return all children of `node` as a list."""
    count = node.child_count() or 0
    out: list[Any] = []
    for i in range(count):
        c = node.child(i)
        if c is not None:
            out.append(c)
    return out


def _walk(node: Any):
    """Yield `node` and all its descendants (depth-first)."""
    yield node
    for c in _children(node):
        yield from _walk(c)


def _node_text(node: Any, as_bytes: bytes) -> str:
    """Get the source text for a node, decoded as UTF-8 (lossy on bad bytes)."""
    try:
        start = node.start_byte() or 0
        end = node.end_byte() or 0
        return as_bytes[start:end].decode("utf-8", errors="replace")
    except Exception:
        return ""


def _node_line_range(node: Any, as_bytes: bytes) -> tuple[int, int]:
    """Return (start_line, end_line) — both 1-indexed, inclusive."""
    start = node.start_byte() or 0
    end = node.end_byte() or 0
    return (
        as_bytes[:start].count(b"\n") + 1,
        as_bytes[:end].count(b"\n") + 1,
    )


def _node_name(node: Any, as_bytes: bytes) -> str | None:
    """Extract the identifier name from a definition node.

    Different grammars put the name in different child kinds:
      - Most C-family/JS: `identifier`, `property_identifier`,
        `field_identifier`, `type_identifier`
      - Kotlin: `simple_identifier`
      - Haskell: `variable`, `name`
      - Ruby: `constant` (PascalCase = class/module), `identifier` (snake_case = method)
      - Elixir: handled below — we need the `alias` child, not the
        `identifier` (which is the keyword)
    """
    # Elixir special-case: for `defmodule Foo` / `defprotocol Bar` / `defimpl Baz`,
    # the keyword is the first identifier and the actual name is the `alias` arg.
    if _kind(node) in ("defmodule", "defprotocol", "defimpl"):
        for child in _named_children(node):
            if _kind(child) == "arguments":
                for arg in _named_children(child):
                    if _kind(arg) in ("alias", "identifier"):
                        return _node_text(arg, as_bytes)
        return None

    # Common case: first named identifier-ish child
    for child in _named_children(node):
        kind = _kind(child)
        if kind in (
            "identifier", "property_identifier", "field_identifier",
            "type_identifier", "name", "constant", "pattern",
            "simple_identifier", "variable", "atom",
        ):
            return _node_text(child, as_bytes)
    # Some grammars (e.g. Go function_declaration) put the name in a
    # field called `name` — child_by_field_name is the safe access.
    try:
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            return _node_text(name_node, as_bytes)
    except Exception:
        pass
    return None


# ── Docstring extraction per language ───────────────────────────────────────


def _extract_docstring(source: str, start_line: int, end_line: int, language: str) -> str | None:
    """Extract a docstring/comment block above or inside a definition."""
    lines = source.splitlines()
    if start_line < 1 or start_line > len(lines):
        return None

    if language == "python":
        # Python: docstring is the first string expression INSIDE the body,
        # which lives one line after the `def`/`class` header.
        for i in range(start_line, min(end_line, len(lines))):
            stripped = lines[i].strip()
            if stripped.startswith(('"""', "'''")):
                if (stripped.startswith('"""') and '"""' in stripped[3:]) or \
                   (stripped.startswith("'''") and "'''" in stripped[3:]):
                    return stripped.strip('"').strip("'").strip() or None
                parts = [stripped]
                for j in range(i, end_line):
                    if j >= len(lines):
                        break
                    parts.append(lines[j])
                    s = lines[j].strip()
                    if s.endswith('"""') or s.endswith("'''"):
                        break
                doc = " ".join(p.strip().strip('"').strip("'") for p in parts)
                return doc.strip() or None
        # Fallback: look at lines before the definition for a triple-quoted block
        for i in range(start_line - 2, max(0, start_line - 5), -1):
            stripped = lines[i].strip()
            if stripped.startswith(('"""', "'''")):
                return stripped.strip('"').strip("'").strip() or None
        return None

    # C-family / Go / Rust / JS / TS: doc comments are ABOVE the definition
    for i in range(start_line - 2, max(-1, start_line - 12), -1):
        if i < 0:
            break
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        # JSDoc /** ... */ (JS/TS/Java/Kotlin/Scala/PHP/Swift/C#)
        if stripped.startswith("/**") and stripped.endswith("*/") and len(stripped) > 4:
            inner = stripped[3:-2].strip()
            return inner or None
        # Rust: /// or //!
        if language in ("rust",) and stripped.startswith("///"):
            return stripped[3:].strip() or None
        # Go: // comment immediately above
        if language == "go" and stripped.startswith("//"):
            return stripped[2:].strip() or None
        # Stop scanning on any non-comment line
        if not stripped.startswith(("//", "#", "///", "//!", "/*", "*")):
            return None
    return None


# ── Public API ──────────────────────────────────────────────────────────────


def get_supported_languages() -> list[str]:
    """Return languages with enhanced tree-sitter extraction support."""
    return list(CLASS_NODE_TYPES.keys())


def parse_symbols(
    source: str,
    file_path: str,
    language: str | None,
) -> tuple[list[SymbolInfo], list[ImportInfo]]:
    """Extract symbols and imports from source code.

    Tries tree-sitter first (works for 35+ languages via
    tree-sitter-language-pack), falls back to regex patterns.

    :param source: Source code text.
    :param file_path: Relative file path.
    :param language: Detected programming language (None → skip AST path).
    :return: Tuple of (symbols, imports).
    """
    symbols: list[SymbolInfo] = []
    imports: list[ImportInfo] = []

    if not source.strip():
        return symbols, imports

    extracted = False
    if language is not None:
        try:
            _extract_with_treesitter(source, file_path, language, symbols, imports)
            extracted = True
        except Exception:
            logger.debug("Tree-sitter extraction failed for %s (%s); using regex",
                         file_path, language, exc_info=True)

    if not extracted:
        _extract_with_regex(source, file_path, language, symbols, imports)

    return symbols, imports


def parse_docstring(source: str, start_line: int, end_line: int) -> str | None:
    """Extract the first docstring from a function/class body (1-indexed lines).

    Backwards-compat shim — the new per-language docstring extraction is in
    `_extract_docstring`, which takes an explicit `language` argument.
    """
    return _extract_docstring(source, start_line, end_line, "python")


# ── Tree-sitter AST walking ─────────────────────────────────────────────────


def _extract_with_treesitter(
    source: str,
    file_path: str,
    language: str,
    symbols: list[SymbolInfo],
    imports: list[ImportInfo],
) -> None:
    """Walk a tree-sitter AST to extract symbols and imports.

    Fixes the long-standing bug where the previous version called
    `parser.parse(bytes)` (which raises TypeError on the bundled binding)
    and then read `node.type` / `node.children` (which don't exist on
    this binding — it's `.kind` + `.child(i)` / `.child_count`).
    """
    from tree_sitter_language_pack import get_parser

    parser = get_parser(language)
    if parser is None:
        raise ValueError(f"No tree-sitter parser for {language}")

    # parse() expects str (lossy-decode UTF-8 so non-ASCII content parses)
    as_text = source if isinstance(source, str) else source.decode("utf-8", errors="replace")
    tree = parser.parse(as_text)
    if tree is None:
        raise ValueError(f"tree-sitter failed to parse {language}")
    root = tree.root_node()

    class_types = CLASS_NODE_TYPES.get(language, frozenset())
    iface_types = INTERFACE_NODE_TYPES.get(language, frozenset())
    func_types = FUNCTION_NODE_TYPES.get(language, frozenset())
    method_types = METHOD_NODE_TYPES.get(language, frozenset())
    import_types = IMPORT_NODE_TYPES.get(language, frozenset())

    # Convert back to bytes for byte-offset → line-number math.
    # (We could also use start_position/end_position, but those are
    # row/col tuples and require a separate conversion.)
    as_bytes = as_text.encode("utf-8", errors="replace")

    def _emit(node: Any, kind: SymbolKind, parent: str | None) -> None:
        name = _node_name(node, as_bytes)
        if not name:
            return
        start_line, end_line = _node_line_range(node, as_bytes)
        # Dedupe so the same definition isn't captured twice. By default
        # we key on (file, line, name, kind) — but for languages where a
        # single logical definition spans multiple AST nodes (e.g. Haskell's
        # `signature` + `function` pair), we also dedupe by name+kind within
        # the same file.
        for existing in symbols:
            if existing.name == name and existing.kind == kind:
                if existing.start_line == start_line:
                    return
                if language in ("haskell", "ruby", "elixir"):
                    # Same name + kind in same file → dedupe (keep first)
                    return
        sig = _node_text(node, as_bytes).splitlines()[0].strip() if _node_text(node, as_bytes) else ""
        doc = _extract_docstring(source, start_line, end_line, language)
        symbols.append(SymbolInfo(
            name=name,
            kind=kind,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            docstring=doc,
            parent=parent,
            signature=sig[:200],
        ))

    def visit(node: Any, parent_class: str | None) -> None:
        kind = _kind(node)

        # Class definitions (class, struct, trait, interface, contract, etc.)
        if kind in class_types:
            _emit(node, SymbolKind.CLASS, parent=None)
            name = _node_name(node, as_bytes)
            for child in _named_children(node):
                visit(child, parent_class=name)
            return

        # Interface definitions
        if kind in iface_types:
            _emit(node, SymbolKind.INTERFACE, parent=None)
            for child in _named_children(node):
                visit(child, parent_class=_node_name(node, as_bytes))
            return

        # Method nodes (within a class) — language-specific
        if kind in method_types and parent_class is not None:
            _emit(node, SymbolKind.METHOD, parent=parent_class)
            return

        # Rust: function_item inside impl_item → method
        if language == "rust" and kind == "function_item" and parent_class is not None:
            _emit(node, SymbolKind.METHOD, parent=parent_class)
            return

        # Free function definitions
        if kind in func_types and kind not in method_types:
            _emit(node, SymbolKind.METHOD if parent_class else SymbolKind.FUNCTION,
                  parent=parent_class)
            # Don't descend — function bodies aren't definitions
            return

        # Elixir special case: `defmodule Foo`, `defprotocol Bar`, `defimpl Baz`,
        # and `def hello` are all `call` nodes whose first identifier is the
        # keyword. Without this, none of them would match any of the per-language
        # node-type maps above because the call itself has no specific kind.
        if language == "elixir" and kind == "call":
            keyword = _elixir_call_keyword(node, as_bytes)
            if keyword in ("defmodule", "defprotocol", "defimpl"):
                _emit_elixir_class(node, as_bytes, file_path, source, language, symbols)
                # Recurse into the do_block to find def/defmodule nested children
                for child in _named_children(node):
                    visit(child, parent_class=_elixir_call_first_arg(node, as_bytes))
                return
            if keyword in ("def", "defp", "defmacro"):
                # Top-level `def foo` → FUNCTION. `def foo` inside a defmodule
                # → METHOD. defp/defmacro preserve their visibility/kind
                # through the signature prefix in `_emit_elixir_def`.
                _emit_elixir_def(node, as_bytes, file_path, source, language,
                                 parent_class, symbols)
                return

        # Imports
        if kind in import_types:
            _extract_import_node(node, as_bytes, source, file_path, language, imports)
            # Continue walking — grouped imports have multiple targets

        for child in _named_children(node):
            visit(child, parent_class)

    visit(root, parent_class=None)


def _elixir_call_keyword(node: Any, as_bytes: bytes) -> str | None:
    """Return the keyword (first identifier) of an Elixir `call` node."""
    for child in _named_children(node):
        if _kind(child) == "identifier":
            return _node_text(child, as_bytes)
    return None


def _elixir_call_first_arg(node: Any, as_bytes: bytes) -> str | None:
    """Return the first argument (alias/identifier) of an Elixir `call`."""
    for child in _named_children(node):
        if _kind(child) == "arguments":
            for arg in _named_children(child):
                if _kind(arg) in ("alias", "identifier"):
                    return _node_text(arg, as_bytes)
    return None


def _emit_elixir_class(
    node: Any, as_bytes: bytes, file_path: str, source: str,
    language: str, symbols: list[SymbolInfo],
) -> None:
    """Emit an Elixir defmodule/defprotocol/defimpl with the module name.

    The standard `_emit` helper would pick up the keyword (`defmodule`)
    as the name — we override to use the first argument (e.g. `MyApp`).
    """
    name = _elixir_call_first_arg(node, as_bytes)
    if not name:
        return
    start_line, end_line = _node_line_range(node, as_bytes)
    for existing in symbols:
        if existing.name == name and existing.kind == SymbolKind.CLASS:
            return
    sig = _node_text(node, as_bytes).splitlines()[0].strip()
    symbols.append(SymbolInfo(
        name=name,
        kind=SymbolKind.CLASS,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        parent=None,
        signature=sig[:200],
    ))


def _emit_elixir_def(
    node: Any, as_bytes: bytes, file_path: str, source: str,
    language: str, parent_class: str | None, symbols: list[SymbolInfo],
) -> None:
    """Emit an Elixir def with the function name (not the `def` keyword).

    `defp` (private def) and `defmacro` are also handled. Private/public
    distinction is preserved via signature.
    """
    keyword = _elixir_call_keyword(node, as_bytes)
    name = _elixir_def_name(node, as_bytes)
    if not name:
        return
    start_line, end_line = _node_line_range(node, as_bytes)
    kind = SymbolKind.METHOD if parent_class else SymbolKind.FUNCTION
    for existing in symbols:
        if existing.name == name and existing.kind == kind:
            return
    sig = _node_text(node, as_bytes).splitlines()[0].strip()
    if keyword == "defp":
        sig = "(private) " + sig
    elif keyword == "defmacro":
        sig = "(macro) " + sig
    symbols.append(SymbolInfo(
        name=name,
        kind=kind,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        parent=parent_class,
        signature=sig[:200],
    ))


def _elixir_def_name(node: Any, as_bytes: bytes) -> str | None:
    """Return the function name from a `def foo` / `defp foo` / `defmacro foo` node."""
    # In tree-sitter's elixir grammar, `def foo do ... end` is parsed as:
    #   call
    #     identifier ('def')
    #     arguments
    #       identifier ('foo')
    #       ...
    #     do_block
    #   OR for `def foo, do: :bar`:
    #   call
    #     identifier ('def')
    #     arguments
    #       identifier ('foo')
    #       ','
    #       keywords
    #         pair ('do: :bar')
    for child in _named_children(node):
        if _kind(child) == "arguments":
            for arg in _named_children(child):
                if _kind(arg) == "identifier":
                    return _node_text(arg, as_bytes)
    return None


def _extract_import_node(
    node: Any,
    as_bytes: bytes,
    source: str,
    file_path: str,
    language: str,
    imports: list[ImportInfo],
) -> None:
    """Extract ImportInfo from a tree-sitter import node, per language."""
    text = _node_text(node, as_bytes).strip()

    if language == "python":
        # `from x import y, z` or `import x` or `import x as y`
        if _kind(node) == "import_from_statement":
            module_name = node.child_by_field_name("module_name")
            module = _node_text(module_name, as_bytes) if module_name else ""
            # Collect imported bindings. The module name itself is a
            # `dotted_name` too — skip it so we only keep the imports.
            # `aliased_import` wraps a `name` + `as` + `alias`. Wildcards
            # come through as `wildcard_import`.
            names: list[str] = []
            module_start = module_name.start_byte() if module_name is not None else -1
            module_end = module_name.end_byte() if module_name is not None else -1
            for child in _named_children(node):
                ck = _kind(child)
                if ck == "aliased_import":
                    nm = child.child_by_field_name("name")
                    if nm is not None:
                        names.append(_node_text(nm, as_bytes))
                elif ck == "wildcard_import":
                    names.append("*")
                elif ck == "dotted_name":
                    # Skip the module itself
                    if module_start >= 0 and child.start_byte() == module_start:
                        continue
                    names.append(_node_text(child, as_bytes))
            imports.append(ImportInfo(
                source_file=file_path,
                imported_module=module,
                imported_names=names,
                is_relative=module.startswith("."),
            ))
            return
        if _kind(node) == "import_statement":
            # `import a.b.c as x` — module is dotted_name; the imported
            # name (for the knowledge graph) is the full dotted path,
            # so downstream code can resolve it to the actual file.
            for child in _named_children(node):
                if _kind(child) in ("dotted_name", "aliased_import"):
                    raw = _node_text(child, as_bytes).split(" as ")[0]
                    # `import os.path` → module='os.path', name='os.path'
                    imports.append(ImportInfo(
                        source_file=file_path,
                        imported_module=raw,
                        imported_names=[raw] if raw else [],
                    ))
            return

    if language == "go":
        # `import "path"` or `import ( "a"; "b" )` or `import alias "path"`
        for child in _named_children(node):
            ck = _kind(child)
            if ck == "import_spec":
                path_node = (
                    child.child_by_field_name("path")
                    or _find_named_child_by_kind(child, "interpreted_string_literal")
                )
                if path_node is not None:
                    path = _node_text(path_node, as_bytes).strip('"').strip("`")
                    imports.append(ImportInfo(
                        source_file=file_path,
                        imported_module=path,
                    ))
                else:
                    # bare import_spec like import "x" — first string literal
                    for sub in _named_children(child):
                        if _kind(sub) == "interpreted_string_literal":
                            path = _node_text(sub, as_bytes).strip('"')
                            imports.append(ImportInfo(
                                source_file=file_path,
                                imported_module=path,
                            ))
                            break

    elif language == "rust":
        # `use foo::bar::{baz, qux};` or `use foo::bar as fb;`
        # The argument/argument_list child holds the path; for grouped
        # imports each `use_as_clause` or `identifier` is a name.
        arg = node.child_by_field_name("argument")
        if arg is not None:
            module = _node_text(arg, as_bytes).strip().rstrip(";").strip("{}")
            imports.append(ImportInfo(
                source_file=file_path,
                imported_module=module,
            ))

    elif language == "java":
        # `import a.b.C;` or `import a.b.*;`
        m = re.match(r"import\s+([\w.*$]+)", text)
        if m:
            imports.append(ImportInfo(
                source_file=file_path,
                imported_module=m.group(1),
            ))

    elif language == "kotlin":
        # `import kotlin.io.println` — sometimes the tree-sitter kotlin
        # grammar parses this as `infix_expression` rather than the
        # expected `import_header` (especially in mixed sources), so we
        # fall through to a regex check on the raw line.
        if _kind(node) == "import_header":
            for child in _named_children(node):
                if _kind(child) == "identifier":
                    imports.append(ImportInfo(
                        source_file=file_path,
                        imported_module=_node_text(child, as_bytes),
                    ))
                    return
        m = re.match(r"import\s+([\w.]+)", text)
        if m:
            imports.append(ImportInfo(
                source_file=file_path,
                imported_module=m.group(1),
            ))

    elif language in ("typescript", "tsx", "javascript", "jsx"):
        # `import { a, b as c } from 'mod'` or `import x from 'mod'`
        # or `import * as x from 'mod'`
        source_node = None
        names: list[str] = []
        for child in _named_children(node):
            ck = _kind(child)
            if ck in ("string", "template_string"):
                source_node = child
            elif ck == "import_clause":
                for sub in _named_children(child):
                    sk = _kind(sub)
                    if sk in ("identifier", "default_identifier"):
                        names.append(_node_text(sub, as_bytes))
                    elif sk == "named_imports":
                        for spec in _named_children(sub):
                            if _kind(spec) == "import_specifier":
                                # `b as c` — prefer the alias if present,
                                # otherwise fall back to the original name.
                                alias_n = spec.child_by_field_name("alias")
                                name_n = spec.child_by_field_name("name")
                                if alias_n is not None:
                                    names.append(_node_text(alias_n, as_bytes))
                                elif name_n is not None:
                                    names.append(_node_text(name_n, as_bytes))
                    elif sk == "namespace_import":
                        id_n = sub.child_by_field_name("name")
                        if id_n is not None:
                            names.append(_node_text(id_n, as_bytes))
        if source_node is not None:
            module = _node_text(source_node, as_bytes).strip("'\"`")
            imports.append(ImportInfo(
                source_file=file_path,
                imported_module=module,
                imported_names=names,
            ))

    elif language == "csharp":
        # `using Foo.Bar;` or `using static Foo.Bar;`
        m = re.match(r"using(?:\s+static)?\s+([\w.]+)", text)
        if m:
            imports.append(ImportInfo(
                source_file=file_path,
                imported_module=m.group(1),
            ))

    elif language == "scala":
        # `import foo.bar.{A, B}` or `import foo.bar._`
        m = re.match(r"import\s+([\w.]+(?:\.\{[^}]+\}|\._)?)", text)
        if m:
            imports.append(ImportInfo(
                source_file=file_path,
                imported_module=m.group(1),
            ))

    elif language == "php":
        m = re.match(r"use\s+([\w\\]+)(?:\s+as\s+(\w+))?", text)
        if m:
            imports.append(ImportInfo(
                source_file=file_path,
                imported_module=m.group(1).replace("\\", "/"),
                alias=m.group(2),
            ))

    elif language == "ruby":
        # `require 'foo/bar'` or `require_relative '../baz'` or `require_relative "x"`
        m = re.match(r"require(?:_relative)?\s+['\"]([^'\"]+)['\"]", text)
        if m:
            imports.append(ImportInfo(
                source_file=file_path,
                imported_module=m.group(1),
                is_relative="require_relative" in text,
            ))

    elif language == "lua":
        m = re.match(r"require\s+['\"]([^'\"]+)['\"]", text)
        if m:
            imports.append(ImportInfo(
                source_file=file_path,
                imported_module=m.group(1),
            ))

    elif language == "elixir":
        m = re.match(r"(?:import|alias|require|use)\s+([\w.]+)", text)
        if m:
            imports.append(ImportInfo(
                source_file=file_path,
                imported_module=m.group(1),
            ))

    elif language == "erlang":
        m = re.match(r"-import\((\w+),", text)
        if m:
            imports.append(ImportInfo(
                source_file=file_path,
                imported_module=m.group(1),
            ))

    elif language == "haskell":
        m = re.match(r"import\s+(qualified\s+)?([\w.]+)", text)
        if m:
            imports.append(ImportInfo(
                source_file=file_path,
                imported_module=m.group(2),
            ))

    elif language == "solidity":
        m = re.match(r"import\s+(?:\"([^\"]+)\"|'([^']+)')", text)
        if m:
            path = m.group(1) or m.group(2)
            imports.append(ImportInfo(
                source_file=file_path,
                imported_module=path,
            ))

    elif language == "swift":
        m = re.match(r"import\s+(\w+)", text)
        if m:
            imports.append(ImportInfo(
                source_file=file_path,
                imported_module=m.group(1),
            ))

    elif language == "dart":
        m = re.match(r"import\s+['\"]([^'\"]+)['\"]", text)
        if m:
            imports.append(ImportInfo(
                source_file=file_path,
                imported_module=m.group(1),
            ))

    else:
        # Generic fallback: store the whole import line
        m = re.match(r"(?:import|use|from|require|include)\s+['\"]?([\w./\\-]+)", text)
        if m:
            imports.append(ImportInfo(
                source_file=file_path,
                imported_module=m.group(1),
            ))
        elif text:
            imports.append(ImportInfo(
                source_file=file_path,
                imported_module=text,
            ))


def _find_named_child_by_kind(node: Any, kind: str) -> Any | None:
    """Return the first named child of `node` whose kind equals `kind`."""
    for child in _named_children(node):
        if _kind(child) == kind:
            return child
    return None


# ── Regex extraction (universal fallback) ───────────────────────────────────


_RE_PATTERNS: dict[str, dict[str, Any]] = {
    "python": {
        "function": re.compile(r"^(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("),
        "class": re.compile(r"^class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[(:]"),
        "import": re.compile(r"^(?:from\s+([.\w]+)\s+)?import\s+(.+)$", re.MULTILINE),
    },
    "default": {
        "function": re.compile(
            r"(?:^(?:async|export|static|pub|fn|fun|func|def|sub|private|public|protected)\s+)?"
            r"(?:function|fn|fun|func|def|sub)\s+"
            r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
            r"|^(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
            r"|^(?:pub\s+)?fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
            r"|^(?:export\s+)?(?:function|fn|func)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
        ),
        "class": re.compile(
            r"^(?:public|private|protected|static|abstract|export|open|final|sealed)?\s*"
            r"(?:class|struct|interface|enum|trait|object|record|data\s+class)\s+"
            r"([a-zA-Z_][a-zA-Z0-9_]*)"
        ),
    },
}


def _extract_with_regex(
    source: str,
    file_path: str,
    language: str | None,
    symbols: list[SymbolInfo],
    imports: list[ImportInfo],
) -> None:
    """Extract symbols using regex patterns (universal fallback).

    Only used when tree-sitter can't parse the language — usually because
    tree-sitter-language-pack doesn't have a grammar for it. The Python
    branch handles its own import pattern; everything else uses the
    generic default patterns and a broad import scan.
    """
    patterns = _RE_PATTERNS.get(language or "", _RE_PATTERNS["default"])
    lines = source.splitlines()

    for i, line in enumerate(lines):
        lineno = i + 1
        stripped = line.strip()

        if not stripped or stripped.startswith(("#", "//", "/*", "*", "--")):
            continue

        func_pattern = patterns.get("function")
        if func_pattern:
            func_match = func_pattern.search(stripped)
            if func_match:
                # default regex has up to 4 capture groups; pick the first non-None
                name = next((g for g in func_match.groups() if g), None)
                if name:
                    end_line = _find_block_end(lines, i)
                    symbols.append(
                        SymbolInfo(
                            name=name,
                            kind=SymbolKind.FUNCTION,
                            file_path=file_path,
                            start_line=lineno,
                            end_line=end_line,
                            signature=stripped,
                        )
                    )
                    continue

        class_pattern = patterns.get("class")
        if class_pattern:
            class_match = class_pattern.search(stripped)
            if class_match:
                name = class_match.group(1)
                if name:
                    end_line = _find_block_end(lines, i)
                    symbols.append(
                        SymbolInfo(
                            name=name,
                            kind=SymbolKind.CLASS,
                            file_path=file_path,
                            start_line=lineno,
                            end_line=end_line,
                            signature=stripped,
                        )
                    )
                    continue

        if language == "python":
            import_pattern = patterns.get("import")
            if import_pattern:
                import_match = import_pattern.search(stripped)
                if import_match:
                    from_module = import_match.group(1)
                    names_str = import_match.group(2)
                    names = [n.strip().split(" as ")[0].strip()
                             for n in names_str.split(",") if n.strip()]
                    if from_module:
                        imports.append(
                            ImportInfo(
                                source_file=file_path,
                                imported_module=from_module,
                                imported_names=names,
                                is_relative=from_module.startswith("."),
                            )
                        )
                    elif names:
                        first_name = names[0]
                        module = first_name.split(".")[0]
                        imports.append(
                            ImportInfo(
                                source_file=file_path,
                                imported_module=module,
                                imported_names=names,
                            )
                        )

    # Broad import scan for non-Python languages
    if language != "python":
        seen: set[tuple[str, str]] = set()
        for line in source.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            m = re.match(r"(?:import|use|from|require|include)\s+['\"]?([\w./\\-]+)", stripped)
            if m:
                key = (file_path, m.group(1))
                if key not in seen:
                    seen.add(key)
                    imports.append(
                        ImportInfo(
                            source_file=file_path,
                            imported_module=m.group(1),
                        )
                    )


def _find_block_end(lines: list[str], start: int) -> int:
    """Find the end of an indented block starting at line index `start`.

    Best-effort heuristic for the regex fallback — tree-sitter produces
    exact ranges, so this is only used when tree-sitter can't parse.
    """
    if start + 1 >= len(lines):
        return start + 1
    base_indent = len(lines[start]) - len(lines[start].lstrip())
    for i in range(start + 1, len(lines)):
        current = lines[i]
        if current.strip() == "":
            continue
        current_indent = len(current) - len(current.lstrip())
        if current_indent <= base_indent and not current.rstrip().endswith(":"):
            return i
    return start + 1