import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import json

from app.finbot.mcp_finbot.mcp_servers_finbot import handle_request, debug_log


@pytest.mark.asyncio
async def test_handle_request_initialize():
    """Test that the MCP server handles initialize requests correctly."""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"roots": {"listChanged": True}},
            "clientInfo": {"name": "test-client", "version": "0.1.0"},
        },
    }

    with patch("app.finbot.mcp_finbot.mcp_servers_finbot.debug_log"):
        response = await handle_request(request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert "protocolVersion" in response["result"]
        assert "serverInfo" in response["result"]
        assert response["result"]["serverInfo"]["name"] == "finbot-mcp"


@pytest.mark.asyncio
async def test_handle_request_tools_list():
    """Test that the MCP server handles tools/list requests correctly."""
    request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}

    with patch("app.finbot.mcp_finbot.mcp_servers_finbot.debug_log"):
        response = await handle_request(request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 2
        assert "result" in response
        assert "tools" in response["result"]

        tools = response["result"]["tools"]
        assert isinstance(tools, list)
        assert len(tools) > 0

        # Verify the structure of the first tool
        tool = tools[0]
        assert "name" in tool
        assert "description" in tool
        assert "inputSchema" in tool


@pytest.mark.asyncio
async def test_handle_request_tools_call():
    """Test that the MCP server handles tools/call requests correctly."""
    request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "tool_name": "get_historical_prices",
            "arguments": {"ticker": "AAPL", "day": "30"},
        },
    }

    mock_result = json.dumps(
        {"dates": ["2023-01-01", "2023-01-02"], "prices": [150.25, 152.75]}
    )

    with patch(
        "app.finbot.mcp_finbot.mcp_tools.get_historical_prices",
        return_value=mock_result,
    ), patch("app.finbot.mcp_finbot.mcp_servers_finbot.debug_log"):
        response = await handle_request(request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 3
        assert "result" in response
        assert "output" in response["result"]


@pytest.mark.asyncio
async def test_handle_request_invalid_method():
    """Test that the MCP server handles invalid method requests correctly."""
    request = {"jsonrpc": "2.0", "id": 4, "method": "invalid_method"}

    with patch("app.finbot.mcp_finbot.mcp_servers_finbot.debug_log"):
        response = await handle_request(request)

        assert response["id"] == 4
        assert "error" in response
        assert "message" in response["error"]
        assert "Unknown method" in response["error"]["message"]


@pytest.mark.asyncio
async def test_handle_request_invalid_tool():
    """Test that the MCP server handles invalid tool requests correctly."""
    request = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {"tool_name": "invalid_tool", "arguments": {}},
    }

    with patch("app.finbot.mcp_finbot.mcp_servers_finbot.debug_log"):
        response = await handle_request(request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 5
        assert "error" in response
        assert "message" in response["error"]
        assert "Unknown tool" in response["error"]["message"]
