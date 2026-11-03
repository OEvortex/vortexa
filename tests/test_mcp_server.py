from __future__ import annotations

import asyncio

from vortexa.interfaces.mcp_server import mcp


def test_mcp_exposes_search_resolve_explain_status_and_reindex_tools():
    tools = asyncio.run(mcp.list_tools())
    assert {tool.name for tool in tools} >= {
        "search",
        "resolve",
        "explain",
        "status",
        "reindex",
    }


def test_mcp_context_is_not_exposed_as_a_tool_argument():
    tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}
    reindex_params = tools["reindex"].parameters.get("properties", {})
    assert "ctx" not in reindex_params
    assert set(reindex_params) == {"force", "include_text"}
