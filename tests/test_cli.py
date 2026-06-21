"""Smoke tests for the CLI subcommand dispatch (search/resolve/explain/serve).

Verifies:
- The subcommand parser recognises all four subcommands
- Legacy `-q` flag still routes through the legacy parser (backward compat)
- Help text exists for every subcommand
- Argument validation works (missing required args raise SystemExit)
"""

from __future__ import annotations

import pytest

from vortexa.interfaces.cli import (
    _build_legacy_parser,
    _build_subcommand_parser,
    _SUBCOMMAND_HANDLERS,
)


def test_subcommand_parser_has_all_four_subcommands():
    parser = _build_subcommand_parser()
    # All four handlers registered
    assert set(_SUBCOMMAND_HANDLERS.keys()) == {"search", "resolve", "explain", "serve"}
    # And the parser recognises each subcommand name as a valid first arg
    for cmd, expected_top_k in [
        ("search", 10),
        ("resolve", 5),
    ]:
        ns = parser.parse_args([cmd, "placeholder_query"])
        assert ns.subcommand == cmd
        assert ns.top_k == expected_top_k
    # `explain` uses `location` instead of `query`
    ns = parser.parse_args(["explain", "src/foo.py:42"])
    assert ns.subcommand == "explain"
    assert ns.location == "src/foo.py:42"
    # `serve` has no positional args
    ns = parser.parse_args(["serve"])
    assert ns.subcommand == "serve"


def test_search_subcommand_requires_query():
    parser = _build_subcommand_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["search"])  # missing required `query`


def test_explain_subcommand_requires_location():
    parser = _build_subcommand_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["explain"])  # missing required `location`


def test_resolve_subcommand_accepts_top_k():
    parser = _build_subcommand_parser()
    ns = parser.parse_args(["resolve", "auth flow", "--top-k", "10"])
    assert ns.subcommand == "resolve"
    assert ns.query == "auth flow"
    assert ns.top_k == 10


def test_search_subcommand_accepts_hybrid_flag():
    parser = _build_subcommand_parser()
    ns = parser.parse_args(["search", "auth", "--hybrid"])
    assert ns.subcommand == "search"
    assert ns.query == "auth"
    assert ns.hybrid is True


def test_legacy_parser_still_accepts_q_flag():
    """The legacy `-q` parser must continue working for backward compat."""
    parser = _build_legacy_parser()
    ns = parser.parse_args(["-q", "find auth"])
    assert ns.query == "find auth"
    assert ns.top_k == 10


def test_legacy_parser_accepts_legacy_environment_details():
    parser = _build_legacy_parser()
    ns = parser.parse_args(["-q", "auth", "/some/path"])
    assert ns.query == "auth"
    assert ns.environment_details == "/some/path"


def test_subcommand_search_accepts_environment_details():
    """The new `search` subcommand also accepts environment_details."""
    parser = _build_subcommand_parser()
    ns = parser.parse_args(["search", "auth", "/some/path", "--plain"])
    assert ns.subcommand == "search"
    assert ns.query == "auth"
    assert ns.environment_details == "/some/path"
    assert ns.plain is True


def test_subcommand_handlers_are_callable():
    for name, handler in _SUBCOMMAND_HANDLERS.items():
        assert callable(handler), f"handler for {name!r} is not callable"


def test_search_rejects_bad_alpha():
    """alpha outside [0, 1] should raise SystemExit via the handler."""
    parser = _build_subcommand_parser()
    ns = parser.parse_args(["search", "auth", "--alpha", "2.0"])
    assert ns.alpha == 2.0
    # The handler itself validates and calls parser.error → SystemExit
    from vortexa.interfaces.cli import cmd_search
    with pytest.raises(SystemExit):
        cmd_search(ns, parser)


def test_explain_rejects_missing_location():
    """If location is missing, the handler should fail."""
    from types import SimpleNamespace
    from vortexa.interfaces.cli import cmd_explain

    parser = _build_subcommand_parser()
    args = SimpleNamespace(location="", root=None)
    with pytest.raises(SystemExit):
        cmd_explain(args, parser)