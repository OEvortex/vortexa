from __future__ import annotations

import asyncio

from vortexa.interfaces.mcp_server import mcp


def test_mcp_exposes_search_resolve_and_explain_tools():
    tools = asyncio.run(mcp.list_tools())
    assert {tool.name for tool in tools} >= {"search", "resolve", "explain"}
