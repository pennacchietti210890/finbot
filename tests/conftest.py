import asyncio
import os
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

import json
from typing import Dict, Any, List

from app.llm.llm_service import LLMService
from app.finbot.graphs import create_graph
from app.finbot.nodes import State
from app.finbot.services import RAGEngineService
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool

class MockLLM:
    """Mock LLM for testing LangGraph nodes and workflows."""
    
    def __init__(self, responses=None):
        """
        Initialize with predefined responses.
        
        Args:
            responses: Dict mapping input patterns to output responses
        """
        self.responses = responses or {}
        self.invoke_calls = []
        self.with_structured_output_calls = []
    
    def invoke(self, messages):
        """Mock LLM invoke method."""
        self.invoke_calls.append(messages)
        
        # For ticker extraction, return a default ticker
        if isinstance(messages, list) and len(messages) > 0:
            last_msg = messages[-1]
            if isinstance(last_msg, dict) and "content" in last_msg:
                if "ticker" in last_msg["content"].lower():
                    return MagicMock(content="AAPL")
        
        # Default response
        mock_response = MagicMock()
        mock_response.content = "This is a mock response from the LLM"
        return mock_response
    
    def with_structured_output(self, output_type):
        """Mock LLM structured output method."""
        self.with_structured_output_calls.append(output_type)
        
        # Return a mock that will produce structured responses
        mock = MagicMock()
        mock.invoke = MagicMock(return_value={"next": "stock_price_data"})
        return mock


@pytest.fixture
def mock_llm():
    """Fixture that provides a MockLLM instance."""
    return MockLLM()


@pytest.fixture
def mock_state():
    """Fixture that provides a mock State for testing nodes."""
    return {
        "next": None,
        "messages": [{"role": "user", "content": "What is the stock price of AAPL?"}],
        "stock_data": "",
        "financials": "",
        "stock_ticker": "AAPL",
        "financials_chart_data": "",
        "macroeconomics_data": ""
    }


@pytest.fixture
def mock_proc():
    """Fixture that provides a mock subprocess for MCP communication."""
    mock = AsyncMock()
    mock.stdin = AsyncMock()
    mock.stdin.write = AsyncMock()
    mock.stdin.drain = AsyncMock()
    mock.stdout = AsyncMock()
    mock.stdout.readline = AsyncMock(return_value=json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "tools": [{
                "name": "get_historical_prices",
                "description": "Fetches historical stock prices",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "day": {"type": "string"}
                    },
                    "required": ["ticker", "day"]
                }
            }]
        }
    }).encode())
    
    return mock


@pytest.fixture
def finbot_graph(mock_llm):
    """Fixture that creates a test FinBot graph with mocked components."""
    with patch('app.finbot.nodes.agents_llm', mock_llm):
        return create_graph(mock_llm) 


@pytest.fixture
def mock_tools():
    """Fixture that provides a set of mock tools."""
    return Tool(
            name="get_historical_prices",
            description="Fetches historical stock prices",
            func=MagicMock(return_value={"dates": ["2023-01-01", "2023-01-02"], "prices": [150.25, 152.75]}),
        )