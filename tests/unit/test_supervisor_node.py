import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio

from app.finbot.nodes import make_supervisor_node, State
from langgraph.types import Command

@pytest.mark.asyncio
async def test_supervisor_node_routes_to_stock_price(mock_llm, mock_state):
    """Test that supervisor routes to stock_price_data for stock price queries."""
    # Configure mock to return stock_price_data
    mock_llm.with_structured_output = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke = MagicMock(return_value={"next": "stock_price_data"})
    mock_llm.with_structured_output.return_value = mock_structured
    
    # Create supervisor node with mock LLM
    members = ["stock_price_data", "financials", "news_search"]
    supervisor_node = make_supervisor_node(mock_llm, members)
    
    # Execute the node with our mock state
    mock_state["messages"] = [{"role": "user", "content": "What is the stock price of AAPL?"}]
    result = supervisor_node(mock_state)
    
    # Verify the result routes to stock_price_data
    assert result.goto == "stock_price_data"
    assert result.update["next"] == "stock_price_data"
    assert result.update["stock_ticker"] == "AAPL"


@pytest.mark.asyncio
async def test_supervisor_node_routes_to_financials(mock_llm, mock_state):
    """Test that supervisor routes to financials for financial statement queries."""
    # Configure mock to return financials
    mock_llm.with_structured_output = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke = MagicMock(return_value={"next": "financial_statements_and_metrics"})
    mock_llm.with_structured_output.return_value = mock_structured
    
    # Create supervisor node with mock LLM
    members = ["stock_price_data", "financial_statements_and_metrics", "news_search"]
    supervisor_node = make_supervisor_node(mock_llm, members)
    
    # Execute the node with our mock state
    mock_state["messages"] = [{"role": "user", "content": "What are the financial statements for AAPL?"}]
    result = supervisor_node(mock_state)
    
    # Verify the result routes to financials
    assert result.goto == "financial_statements_and_metrics"
    assert result.update["next"] == "financial_statements_and_metrics"


@pytest.mark.asyncio
async def test_supervisor_node_handles_invalid_query(mock_llm, mock_state):
    """Test that supervisor correctly handles invalid queries."""
    # Configure mock to return INVALID_QUERY
    mock_llm.with_structured_output = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke = MagicMock(return_value={"next": "INVALID_QUERY"})
    mock_llm.with_structured_output.return_value = mock_structured
    
    # Create supervisor node with mock LLM
    members = ["stock_price_data", "financials", "news_search"]
    supervisor_node = make_supervisor_node(mock_llm, members)
    
    # Execute the node with our mock state
    mock_state["messages"] = [{"role": "user", "content": "How do I bake a chocolate cake?"}]
    result = supervisor_node(mock_state)
    
    # Verify the result has the correct behavior for invalid queries
    assert result.goto == "__end__"
    assert "messages" in result.update
    assert len(result.update["messages"]) > len(mock_state["messages"])
    
    # Get the last message
    last_message = result.update["messages"][-1]
    assert last_message["role"] == "assistant"
    assert "cannot be processed" in last_message["content"]
    assert last_message["name"] == "system"


@pytest.mark.asyncio
async def test_supervisor_node_handles_finish(mock_llm, mock_state):
    """Test that supervisor correctly handles FINISH routing."""
    # Configure mock to return FINISH
    mock_llm.with_structured_output = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke = MagicMock(return_value={"next": "FINISH"})
    mock_llm.with_structured_output.return_value = mock_structured
    
    # Create supervisor node with mock LLM
    members = ["stock_price_data", "financials", "news_search"]
    supervisor_node = make_supervisor_node(mock_llm, members)
    
    # Execute the node with our mock state
    result = supervisor_node(mock_state)
    
    # Verify the result goes to END
    assert result.goto == "__end__"


@pytest.mark.asyncio
async def test_supervisor_node_handles_error(mock_llm, mock_state):
    """Test that supervisor handles errors gracefully."""
    # Configure mock to raise an exception
    mock_llm.invoke = MagicMock(side_effect=Exception("Test error"))
    
    # Create supervisor node with mock LLM
    members = ["stock_price_data", "financials", "news_search"]
    supervisor_node = make_supervisor_node(mock_llm, members)
    
    # Execute the node with our mock state
    result = supervisor_node(mock_state)
    
    # Verify the result handles the error gracefully
    assert result.goto == "__end__"
    assert "messages" in result.update
    assert len(result.update["messages"]) > len(mock_state["messages"])
    
    # Get the last message
    last_message = result.update["messages"][-1]
    assert last_message["role"] == "assistant"
    assert "error" in last_message["content"].lower()
    assert last_message["name"] == "system" 