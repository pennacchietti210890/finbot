import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import json

from app.finbot.nodes import stock_price_node, stock_price_chart_node, load_tools_from_mcp_server
from app.finbot.mcp_agents import ReActAgent
from langgraph.types import Command

class MockReActResult:
    """Mock result from a ReAct agent invoke."""
    def __init__(self, stock_data=None, message=None):
        self.stock_data = stock_data or json.dumps({
            "dates": ["2023-01-01", "2023-01-02"],
            "prices": [150.25, 152.75]
        })
        self.message = message or "Here is the stock price data for AAPL"
        self.messages = [
            MagicMock(content=self.stock_data),
            MagicMock(content=self.message)
        ]
    
    def __getitem__(self, key):
        if key == "messages":
            return self.messages
        return None


@pytest.mark.asyncio
async def test_stock_price_node_fetches_data(mock_llm, mock_state, mock_proc, mock_tools):
    """Test that stock_price_node correctly fetches and processes stock data."""
    # Mock the dependencies
    with patch('app.finbot.nodes.asyncio.create_subprocess_exec', return_value=mock_proc), \
         patch('app.finbot.nodes.load_tools_from_mcp_server', return_value=[mock_tools]), \
         patch('app.finbot.nodes.ReActAgent') as mock_react_agent:
        
        # Configure the mock ReAct agent
        mock_agent = MagicMock()
        mock_agent.agent = MagicMock()
        mock_agent.agent.ainvoke = AsyncMock(return_value=MockReActResult())
        mock_react_agent.return_value = mock_agent
        
        # Execute the node
        result = await stock_price_node(mock_state)
        
        # Verify the result
        assert result.goto == "supervisor"
        assert "stock_data" in result.update
        assert "messages" in result.update
        assert len(result.update["messages"]) == 1
        assert result.update["messages"][0]["name"] == "stock_price"


@pytest.mark.asyncio
async def test_stock_price_chart_node_with_data(mock_state):
    """Test that stock_price_chart_node correctly processes stock data for charts."""
    # Prepare state with stock data
    mock_state["stock_data"] = json.dumps({
        "dates": ["2023-01-01", "2023-01-02"],
        "prices": [150.25, 152.75]
    })
    
    # Execute the node
    result = stock_price_chart_node(mock_state)
    
    # Verify the result
    assert result.goto == "supervisor"
    assert "stock_data" in result.update
    assert "messages" in result.update
    assert len(result.update["messages"]) == 1
    assert result.update["messages"][0]["name"] == "stock_price_chart"
    assert "plotted" in result.update["messages"][0]["content"].lower()


@pytest.mark.asyncio
async def test_stock_price_chart_node_with_error_data(mock_state):
    """Test that stock_price_chart_node handles error data gracefully."""
    # Prepare state with error in stock data
    mock_state["stock_data"] = json.dumps({"error": "No data available"})
    
    # Execute the node
    result = stock_price_chart_node(mock_state)
    
    # Verify the result
    assert result.goto == "supervisor"
    assert "messages" in result.update
    assert len(result.update["messages"]) == 1
    assert result.update["messages"][0]["name"] == "stock_price_chart"
    assert "no valid" in result.update["messages"][0]["content"].lower()


@pytest.mark.asyncio
async def test_stock_price_chart_node_with_no_data(mock_state):
    """Test that stock_price_chart_node handles missing data gracefully."""
    # Prepare state with no stock data
    mock_state["stock_data"] = ""
    
    # Execute the node
    result = stock_price_chart_node(mock_state)
    
    # Verify the result
    assert result.goto == "supervisor"
    assert "messages" in result.update
    assert len(result.update["messages"]) == 1
    assert result.update["messages"][0]["name"] == "stock_price_chart"
    assert "no valid" in result.update["messages"][0]["content"].lower() 