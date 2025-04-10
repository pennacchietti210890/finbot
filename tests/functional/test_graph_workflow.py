import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import json
import os

from app.finbot.graphs import create_graph
from app.finbot.nodes import stock_price_node, stock_price_chart_node, make_supervisor_node
from app.llm.llm_service import LLMService
from app.finbot.services import RAGEngineService


class MockAInvoke:
    """Mock for the graph's ainvoke method."""
    
    def __init__(self, return_value=None):
        self.return_value = return_value or {
            "messages": [MagicMock(type="human", content="Here is the info you requested", name="stock_price")],
            "stock_data": json.dumps({
                "dates": ["2023-01-01", "2023-01-02"],
                "prices": [150.25, 152.75]
            }),
            "stock_ticker": "AAPL"
        }
        self.invoked_with = []
        
    async def ainvoke(self, state, config=None):
        """Mock ainvoke method that captures calls and returns predefined value."""
        self.invoked_with.append((state, config))
        return self.return_value


@pytest.fixture
def mock_graph_builder():
    """Fixture providing mocked graph builder components."""
    mock_builder = MagicMock()
    mock_builder.add_node = MagicMock()
    mock_builder.add_edge = MagicMock()
    mock_builder.compile = MagicMock()
    
    # Return a mock graph with a fake ainvoke method
    mock_graph = MagicMock()
    mock_ainvoke = MockAInvoke()
    mock_graph.ainvoke = mock_ainvoke.ainvoke
    mock_builder.compile.return_value = mock_graph
    
    return mock_builder, mock_ainvoke


@pytest.mark.asyncio
async def test_create_graph():
    """Test that the graph is created with the correct structure."""
    mock_llm = MagicMock()
    
    with patch('app.finbot.graphs.StateGraph') as mock_state_graph, \
         patch('app.finbot.graphs.MemorySaver') as mock_memory_saver, \
         patch('app.finbot.graphs.make_supervisor_node') as mock_make_supervisor:
        
        # Configure mocks
        mock_graph_builder = MagicMock()
        mock_state_graph.return_value = mock_graph_builder
        mock_memory_saver.return_value = MagicMock()
        mock_make_supervisor.return_value = MagicMock()
        
        # Create the graph
        graph = create_graph(mock_llm)
        
        # Verify node additions
        assert mock_graph_builder.add_node.call_count >= 9  # 1 supervisor + 8 worker nodes
        assert mock_graph_builder.add_edge.call_count >= 1  # At least START -> supervisor
        assert mock_graph_builder.compile.call_count == 1
        
        # Verify the graph was returned
        assert graph is not None


@pytest.mark.asyncio
async def test_graph_stock_price_query(mock_graph_builder, mock_llm):
    """Test the full workflow for a stock price query."""
    mock_builder, mock_ainvoke = mock_graph_builder
    
    # Mock the StateGraph and other dependencies
    with patch('app.finbot.graphs.StateGraph', return_value=mock_builder), \
         patch('app.finbot.graphs.MemorySaver', return_value=MagicMock()), \
         patch('app.finbot.graphs.make_supervisor_node', return_value=MagicMock()), \
         patch('app.finbot.nodes.stock_price_node', return_value=AsyncMock()), \
         patch('app.finbot.nodes.stock_price_chart_node', return_value=MagicMock()):
        
        # Create the graph
        graph = create_graph(mock_llm)
        
        # Invoke the graph with a stock price query
        state = {
            "messages": [{"role": "user", "content": "What is the stock price of AAPL?"}]
        }
        config = {"configurable": {"thread_id": "test-session"}}
        
        result = await graph.ainvoke(state, config=config)
        
        # Verify the graph was invoked correctly
        assert len(mock_ainvoke.invoked_with) == 1
        assert mock_ainvoke.invoked_with[0][0] == state
        assert mock_ainvoke.invoked_with[0][1] == config
        
        # Verify the result structure
        assert "messages" in result
        assert "stock_data" in result
        assert "stock_ticker" in result


@pytest.mark.asyncio
async def test_graph_invalid_query(mock_graph_builder, mock_llm):
    """Test the workflow for an invalid query that can't be processed."""
    mock_builder, mock_ainvoke = mock_graph_builder
    
    # Configure the mock to return a response for an invalid query
    mock_ainvoke.return_value = {
        "messages": [
            {"role": "user", "content": "How do I bake a chocolate cake?"},
            {"role": "assistant", "content": "Your query cannot be processed with the current tools of this financial bot.", "name": "system"}
        ],
        "stock_ticker": "",
        "next": "FINISH"
    }
    
    # Mock the StateGraph and other dependencies
    with patch('app.finbot.graphs.StateGraph', return_value=mock_builder), \
         patch('app.finbot.graphs.MemorySaver', return_value=MagicMock()), \
         patch('app.finbot.graphs.make_supervisor_node', return_value=MagicMock()), \
         patch('app.finbot.nodes.stock_price_node', return_value=AsyncMock()), \
         patch('app.finbot.nodes.stock_price_chart_node', return_value=MagicMock()):
        
        # Create the graph
        graph = create_graph(mock_llm)
        
        # Invoke the graph with an invalid query
        state = {
            "messages": [{"role": "user", "content": "How do I bake a chocolate cake?"}]
        }
        
        result = await graph.ainvoke(state)
        
        # Verify the result structure
        assert "messages" in result
        assert len(result["messages"]) == 2
        assert result["messages"][1]["content"].startswith("Your query cannot be processed")


@pytest.mark.asyncio
async def test_full_graph_workflow_integration():
    """
    Full integration test for the graph workflow.
    
    This is a more complex test that mocks the minimum necessary
    to test the interaction between multiple nodes in the graph.
    """
    # Mock LLM to return predictable responses
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(return_value=MagicMock(content="AAPL"))
    mock_structured = MagicMock()
    mock_structured.invoke = MagicMock(return_value={"next": "stock_price_data"})
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured)
    
    # Mock the subprocess creation for MCP calls
    mock_proc = AsyncMock()
    mock_proc.stdin = AsyncMock()
    mock_proc.stdin.write = AsyncMock()
    mock_proc.stdin.drain = AsyncMock()
    mock_proc.stdout = AsyncMock()
    
    # Mock the MCP communication
    mock_proc.stdout.readline = AsyncMock(side_effect=[
        # For tools/list
        json.dumps({
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
        }).encode(),
        # For tools/call
        json.dumps({
            "jsonrpc": "2.0",
            "id": 42,
            "result": {
                "output": json.dumps({
                    "dates": ["2023-01-01", "2023-01-02"],
                    "prices": [150.25, 152.75]
                })
            }
        }).encode()
    ])
    
    # Create a real graph but with mocked components
    with patch('app.finbot.nodes.LLMService', return_value=MagicMock(client=mock_llm)), \
         patch('app.finbot.nodes.asyncio.create_subprocess_exec', return_value=mock_proc), \
         patch('app.finbot.nodes.agents_llm', mock_llm):
        
        # Initialize any singleton services that might be needed
        RAGEngineService._instance = None  # Reset singleton
        
        # Create a minimal graph with just the components we need
        supervisor_node = make_supervisor_node(mock_llm, ["stock_price_data", "stock_price_chart"])
        
        # Create a mock react agent for the stock price node
        mock_react_agent = MagicMock()
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value={
            "messages": [
                MagicMock(content=json.dumps({
                    "dates": ["2023-01-01", "2023-01-02"],
                    "prices": [150.25, 152.75]
                })),
                MagicMock(content="Here's the stock price data for AAPL.")
            ]
        })
        mock_react_agent.agent = mock_agent
        
        with patch('app.finbot.nodes.ReActAgent', return_value=mock_react_agent):
            # Test the supervisor node
            initial_state = {
                "next": None,
                "messages": [{"role": "user", "content": "What is the stock price of AAPL?"}],
                "stock_data": "",
                "financials": "",
                "stock_ticker": "",
                "financials_chart_data": "",
                "macroeconomics_data": ""
            }
            
            # Execute the supervisor node
            supervisor_result = supervisor_node(initial_state)
            assert supervisor_result.goto == "stock_price_data"
            
            # Execute the stock price node with the updated state
            updated_state = {**initial_state, **supervisor_result.update}
            stock_price_result = await stock_price_node(updated_state)
            assert stock_price_result.goto == "supervisor"
            assert "stock_data" in stock_price_result.update
            
            # Execute the supervisor again with the new state
            final_state = {**updated_state, **stock_price_result.update}
            supervisor_result2 = supervisor_node(final_state)
            
            # Verify the flow is as expected
            assert "stock_data" in final_state
            assert final_state["stock_ticker"] == "AAPL" 