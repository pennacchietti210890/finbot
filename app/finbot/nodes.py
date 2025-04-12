import asyncio
import json
import logging
import os
import shutil
from typing import Annotated, Any, Dict, List, Literal, TypedDict

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, Tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pydantic import BaseModel, Field, create_model

from app.finbot.mcp_agents import ReActAgent
from app.llm.llm_service import LLMService
from app.llm.rag_query_engine import RAGEngine
import aiohttp

# Get the logger for this module
logger = logging.getLogger(__name__)

env_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"
)
load_dotenv(env_path)

agents_llm = LLMService(
    llm_provider="openai", model_name="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")
).client


MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:5005")


async def fetch_tool_list() -> dict:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{MCP_SERVER_URL}/tools/list", timeout=10) as resp:
                if resp.status != 200:
                    return {"error": f"HTTP {resp.status}"}
                return await resp.json()
    except asyncio.TimeoutError:
        return {"error": "Timeout while requesting tool list"}
    except Exception as e:
        return {"error": f"Error: {str(e)}"}

async def call_tool_http(tool_name: str, arguments: dict) -> dict:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{MCP_SERVER_URL}/tools/call",
                json={"tool_name": tool_name, "arguments": arguments},
                timeout=15,
            ) as resp:
                if resp.status != 200:
                    return {"error": f"HTTP {resp.status}"}
                return await resp.json()
    except asyncio.TimeoutError:
        return {"error": "Timeout calling MCP tool"}
    except Exception as e:
        return {"error": f"Error calling MCP tool: {str(e)}"}


# Wrap raw MCP tool schema into LangChain @tool
class MCPTool(BaseTool):
    name: str
    description: str
    tool_name: str
    input_schema: Dict[str, Any]

    def __init__(self, **kwargs):
        type_map = {
            "string": str,
            "integer": int,
            "int": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        schema_fields = {
            key: (type_map.get(py_type["type"], str), ...)
            for key, py_type in kwargs["input_schema"]["properties"].items()
        }

        schema = create_model(f"{kwargs['tool_name'].title()}Schema", **schema_fields)
        kwargs["args_schema"] = schema
        super().__init__(**kwargs)

    async def _arun(self, *args, **kwargs):
        logger.info(f"Calling MCP tool: {self.tool_name} with args: {kwargs}")
        response = await call_tool_http(self.tool_name, kwargs)
        if "error" in response:
            return json.dumps({"error": response["error"]})
        return response["output"]

    def _run(self, tool_input: dict, **kwargs):
        raise NotImplementedError("Only async supported")



def wrap_tool_for_langchain(tool_schema):
    return MCPTool(
        name=tool_schema["name"],
        description=tool_schema.get("description", ""),
        tool_name=tool_schema["name"],
        input_schema=tool_schema["inputSchema"],
    )


# ✅ Main helper: load LangChain tools from MCP server
async def load_tools_from_mcp_server():
    response = await fetch_tool_list()

    if "error" in response:
        return [create_error_tool(f"MCP server error: {response['error']}")]

    tools = response.get("tools", [])
    if not tools:
        return [create_error_tool("No tools returned from MCP server")]

    wrapped = []
    for tool_def in tools:
        try:
            tool = wrap_tool_for_langchain(tool_def)
            wrapped.append(tool)
        except Exception as e:
            logger.error(f"Error wrapping tool {tool_def['name']}: {e}")

    if not wrapped:
        return [create_error_tool("No MCP tools could be loaded")]
    return wrapped



# Create a dummy tool that returns an error message
def create_error_tool(error_message):
    class ErrorTool(BaseTool):
        name = "error_tool"
        description = "This tool reports an error with the MCP server"
        
        def _run(self, *args, **kwargs):
            return json.dumps({"error": error_message})
        
        async def _arun(self, *args, **kwargs):
            return json.dumps({"error": error_message})
    
    return ErrorTool()


class FinancialsChartStruct(BaseModel):
    """Dates and Values of the financials time series to chart."""

    dates: List[str] = Field(
        description="The dates of the financials time series to chart."
    )
    values: List[float] = Field(
        description="The values of the financials time series to chart. Just the values, no other text."
    )


class State(TypedDict):
    next: str
    messages: Annotated[list, add_messages]
    stock_data: str
    financials: str
    stock_ticker: str
    financials_chart_data: str
    macroeconomics_data: str


def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> StateGraph:
    options = ["FINISH", "INVALID_QUERY"] + members
    system_prompt = """You are a supervisor tasked with managing a conversation between the following workers: {workers}.
    
    Given the user request and the conversation history so far, respond with the worker to act next or FINISH if the request has been fully addressed.
    Only select one worker at a time. Wait for that worker to complete before selecting another.
    
    Important rules:
    1. If a user asks to plot data about stock price, then first use stock_price to retrieve the data, then use stock_price_chart to plot the data, then FINISH.
    2. If a user asks to plot data about financial metrics or statements, then first use financial_statements_and_metrics to retrieve the data, then use financials_chart to plot the data, then FINISH.
    3. If a user asks to plot data about macroeconomic indicators, then first use macroeconomics to retrieve the data, then use macroeconomics_chart to plot the data, then FINISH.
    4. If a user asks about data from the annual report, then first use annual_report to retrieve the data, then FINISH.
    5. ONLY proceed to the next worker if it's truly necessary to complete the user's request.
    6. If the user's request is about a different stock from what has been asked in the previous request, re-start the workflow with the new stock ticker.
    7. If no more workers are needed or the request has been addressed, respond with FINISH.
    8. If the user's request is outside the scope of this financial assistant's capabilities or cannot be processed with available tools, respond with INVALID_QUERY.
    
    Current workers: {workers}
    """

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router."""
        # Format the system prompt with the worker names
        formatted_prompt = system_prompt.format(workers=members)

        # Otherwise, ask the LLM what to do next
        messages = [
            {"role": "system", "content": formatted_prompt},
        ] + state["messages"]

        logger.info(f"SUPERVISOR NODE - Incoming messages: {messages}")

        ticker_prompt = f"Fetch the stock ticker from the user's request. Only return the ticker, no other text."
        ticker_messages = state["messages"] + [
            {"role": "user", "content": ticker_prompt},
        ]

        try:
            if "stock_ticker" in state:
                logger.info(
                    f"SUPERVISOR NODE - Previous Stock ticker: {state['stock_ticker']}"
                )
            ticker_response = llm.invoke(ticker_messages)
            # if not state["stock_ticker"]:
            state["stock_ticker"] = ticker_response.content
            logger.info(f"SUPERVISOR NODE - Raw Ticker Response: {ticker_response}")
            logger.info(f"SUPERVISOR NODE - Stock ticker: {state['stock_ticker']}")
            if (
                state["stock_ticker"]
                and state["stock_ticker"] != ticker_response.content
            ):
                logger.info(
                    f"SUPERVISOR NODE - Stock ticker has changed. Restarting workflow with new ticker: {ticker_response.content}"
                )
                messages = [
                    {"role": "system", "content": formatted_prompt},
                ]
                state["messages"] = messages

            logger.info("Calling LLM for routing decision...")
            response = llm.with_structured_output(Router).invoke(messages)
            logger.info(f"SUPERVISOR NODE - Raw Response: {response}")

            # Handle both response formats
            if isinstance(response, dict):
                if "next" in response:
                    goto = response["next"]
                elif (
                    "type" in response
                    and "properties" in response
                    and isinstance(response["properties"], dict)
                ):
                    # Handle nested format
                    if "next" in response["properties"]:
                        goto = response["properties"]["next"]
                    else:
                        logger.warning(
                            "Cannot find 'next' in nested properties, defaulting to FINISH"
                        )
                        goto = "FINISH"
                else:
                    logger.warning("Unexpected response format, defaulting to FINISH")
                    goto = "FINISH"
            else:
                logger.warning(
                    f"Response is not a dict: {type(response)}, defaulting to FINISH"
                )
                goto = "FINISH"

            # Handle special INVALID_QUERY case
            if goto == "INVALID_QUERY":
                logger.info("Router detected an invalid or unprocessable query")
                return Command(
                    goto=END,
                    update={
                        "next": "FINISH",
                        "stock_ticker": state["stock_ticker"],
                        "messages": state["messages"]
                        + [
                            {
                                "role": "assistant",
                                "content": "Your query cannot be processed with the current tools of this financial bot. Please try asking about stock prices, financial statements, macroeconomic indicators, or company annual reports.",
                                "name": "system",
                            }
                        ],
                    },
                )
            elif goto == "FINISH":
                logger.info("Router decided to FINISH")
                goto = END
            else:
                logger.info(f"Router selected next worker: {goto}")

            return Command(
                goto=goto, update={"next": goto, "stock_ticker": state["stock_ticker"]}
            )
        except Exception as e:
            logger.error(f"Error in supervisor node: {str(e)}", exc_info=True)
            # If there's an error, default to ending the workflow with an error message
            return Command(
                goto=END,
                update={
                    "next": "FINISH",
                    "messages": state["messages"]
                    + [
                        {
                            "role": "assistant",
                            "content": "I encountered an error processing your request. Please try rephrasing your question or ask about a different topic.",
                            "name": "system",
                        }
                    ],
                },
            )

    return supervisor_node


async def stock_price_node(state: State) -> Command[Literal["supervisor"]]:
    logger.info("STOCK PRICE NODE - Processing request")
    
    try:        
        # Load tools from the MCP server
        logger.info("Loading tools from MCP server")
        wrapped_tools = await load_tools_from_mcp_server()
        
        if not wrapped_tools:
            logger.error("No tools were loaded from the MCP server")
            return Command(
                update={
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "I'm sorry, but I couldn't access the stock price tools. Please try again later.",
                            "name": "stock_price"
                        }
                    ]
                },
                goto="supervisor",
            )
        
        # Check if any tool is an error tool
        for tool in wrapped_tools:
            if tool.name == "error_tool":
                logger.error(f"Error tool detected: {tool.description}")
                return Command(
                    update={
                        "messages": [
                            {
                                "role": "assistant",
                                "content": f"I'm sorry, but there was an issue with the stock price service: {tool._run()}",
                                "name": "stock_price"
                            }
                        ]
                    },
                    goto="supervisor",
                )
        
        # Map tools by name
        tool_map = {tool.name: tool for tool in wrapped_tools}
        
        # Check if the required tool is available
        if "get_historical_prices" not in tool_map:
            logger.error("get_historical_prices tool not found in loaded tools")
            return Command(
                update={
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "I'm sorry, but the stock price tool is not available. Please try again later.",
                            "name": "stock_price"
                        }
                    ]
                },
                goto="supervisor",
            )
        
        # Create and run the agent
        logger.info("Creating ReActAgent with get_historical_prices tool")
        react = ReActAgent(agents_llm, [tool_map["get_historical_prices"]])
        stock_price_agent = react.agent
        result = await stock_price_agent.ainvoke(state)
        
        # Process the result
        logger.info("Stock price agent execution completed")
        if "messages" not in result or len(result["messages"]) < 2:
            logger.error(f"Invalid result format from stock_price_agent: {result}")
            return Command(
                update={
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "I'm sorry, but I couldn't retrieve the stock price data in the expected format.",
                            "name": "stock_price"
                        }
                    ]
                },
                goto="supervisor",
            )
        
        # Extract data from result
        stock_data = result["messages"][-2].content
        last_message = result["messages"][-1].content
        logger.info("STOCK PRICE NODE - Retrieved stock data and message")
        logger.info(f"Stock data: {stock_data}")
        
        return Command(
            update={
                "messages": [
                    {"role": "assistant", "content": last_message, "name": "stock_price"}
                ],
                "stock_data": stock_data,
            },
            goto="supervisor",
        )
        
    except Exception as e:
        logger.error(f"Error in stock_price_node: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        return Command(
            update={
                "messages": [
                    {
                        "role": "assistant",
                        "content": f"I'm sorry, but I encountered an error while processing your stock price request: {str(e)}",
                        "name": "stock_price"
                    }
                ]
            },
            goto="supervisor",
        )


def stock_price_chart_node(state: State) -> Command[Literal["supervisor"]]:
    logger.info("STOCK PRICE CHART NODE - Processing request")
    stock_data = state.get("stock_data", None)
    if not stock_data or "error" in stock_data:
        logger.warning("STOCK PRICE CHART NODE - No valid stock data available to plot")
        return Command(
            update={
                "messages": [
                    {
                        "role": "assistant",
                        "content": "No valid stock data available to plot.",
                        "name": "stock_price_chart",
                    }
                ]
            },
            goto="supervisor",
        )

    logger.info("STOCK PRICE CHART NODE - Data available for plotting")
    return Command(
        update={
            "messages": [
                {
                    "role": "assistant",
                    "content": "✅ Data is ready to be plotted.",
                    "name": "stock_price_chart",
                }
            ],
            "stock_data": stock_data,
        },
        goto="supervisor",
    )


async def financials_node(state: State) -> Command[Literal["supervisor"]]:
    logger.info("FINANCIALS NODE - Processing request")

    wrapped_tools = await load_tools_from_mcp_server()
    tool_map = {tool.name: tool for tool in wrapped_tools}

    react = ReActAgent(agents_llm, [tool_map["get_financials"]])
    financials_agent = react.agent
    result = await financials_agent.ainvoke(state)

    # ✅ Accessing Intermediate response tool call observation
    financials_data = result["messages"][-2].content  # Already a JSON dict
    last_message = result["messages"][-1].content  # Already a JSON dict
    logger.info("FINANCIALS NODE - Retrieved financial data and message")

    return Command(
        update={
            "messages": [HumanMessage(content=last_message, name="financials")],
            "financials": financials_data,  # ✅ Directly store tool output
        },
        goto="supervisor",
    )


async def financials_chart_node(state: State) -> Command[Literal["supervisor"]]:
    logger.info("FINANCIALS CHART NODE - Processing request")

    wrapped_tools = await load_tools_from_mcp_server()
    tool_map = {tool.name: tool for tool in wrapped_tools}

    react = ReActAgent(agents_llm, [tool_map["get_financials"]], FinancialsChartStruct)
    financials_chart_agent = react.agent
    financials_chart_response = financials_chart_agent.invoke(state)

    last_message = financials_chart_response["messages"][-1].content
    financials_chart_data = financials_chart_response.get("structured_response", None)
    logger.info(
        f"FINANCIALS CHART NODE - Retrieved financials chart data: {financials_chart_data}"
    )

    if not financials_chart_data:
        logger.warning(
            "FINANCIALS CHART NODE - No valid financials data available to plot"
        )
        return Command(
            update={
                "messages": [
                    {
                        "role": "assistant",
                        "content": "No valid financials data available to plot.",
                        "name": "financials_chart",
                    }
                ]
            },
            goto="supervisor",
        )

    logger.info("FINANCIALS  CHART NODE - Data available for plotting")
    return Command(
        update={
            "messages": [
                {
                    "role": "assistant",
                    "content": "✅ Data is ready to be plotted.",
                    "name": "financials_chart",
                }
            ],
            "financials_chart_data": financials_chart_data.json(),
        },
        goto="supervisor",
    )


async def macroeconomics_node(state: State) -> Command[Literal["supervisor"]]:
    logger.info("MACROECONOMICS NODE - Processing request")

    wrapped_tools = await load_tools_from_mcp_server()
    tool_map = {tool.name: tool for tool in wrapped_tools}

    react = ReActAgent(agents_llm, [tool_map["get_macroeconomic_series"]])
    macroeconomics_agent = react.agent
    result = await macroeconomics_agent.ainvoke(state)

    # ✅ Accessing Intermediate response tool call observation
    macroeconomics_data = result["messages"][-2].content  # Already a JSON dict
    last_message = result["messages"][-1].content  # Already a JSON dict

    logger.info(f"MACROECONOMICS DATA FOUND: {macroeconomics_data}")

    return Command(
        update={
            "messages": [HumanMessage(content=last_message, name="macroeconomics")],
            "macroeconomics_data": macroeconomics_data,  # ✅ Directly store tool output
        },
        goto="supervisor",
    )


def macroeconomics_chart_node(state: State) -> Command[Literal["supervisor"]]:
    logger.info("MACROECONOMIS CHART NODE - Processing request")
    macroeconomics_data = state.get("macroeconomics_data", None)
    if not macroeconomics_data or "error" in macroeconomics_data:
        logger.warning(
            "MACROECONOMIS CHART NODE - No valid macroeconomic data available to plot"
        )
        return Command(
            update={
                "messages": [
                    {
                        "role": "assistant",
                        "content": "No valid macroeconomic data available to plot.",
                        "name": "macroeconomics_chart",
                    }
                ]
            },
            goto="supervisor",
        )

    logger.info("MACROECONOMICS CHART NODE - Data available for plotting")
    return Command(
        update={
            "messages": [
                {
                    "role": "assistant",
                    "content": "✅ Data is ready to be plotted.",
                    "name": "macroeconomics_chart",
                }
            ],
            "macroeconomics_data": macroeconomics_data,
        },
        goto="supervisor",
    )


async def news_search_node(state: State) -> Command[Literal["supervisor"]]:
    logger.info("NEWS SEARCH NODE - Processing request")

    wrapped_tools = await load_tools_from_mcp_server()
    tool_map = {tool.name: tool for tool in wrapped_tools}

    react = ReActAgent(agents_llm, [tool_map["search_news"]])
    news_search_agent = react.agent
    result = await news_search_agent.ainvoke(state)

    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="news_search")
            ],
        },
        goto="supervisor",
    )


async def annual_report_node(state: State) -> Command[Literal["supervisor"]]:
    """
    Node for handling annual report queries using the RAG engine
    """
    logger.info("ANNUAL REPORT NODE - Processing request")

    # Get the RAG engine service
    from app.finbot.services import RAGEngineService

    rag_service = RAGEngineService.get_instance()

    # Get the RAG engine for this ticker
    rag_engine = rag_service.get_engine(state["stock_ticker"])

    if not rag_engine:
        # If the engine doesn't exist, we need to fetch the annual report first
        logger.info(
            f"No RAG engine found for {state['stock_ticker']}, need to fetch annual report first"
        )

        wrapped_tools = await load_tools_from_mcp_server()
        tool_map = {tool.name: tool for tool in wrapped_tools}

        react = ReActAgent(agents_llm, [tool_map["get_annual_report"]])
        annual_report_agent = react.agent
        result = await annual_report_agent.ainvoke(state)
        latest_path = result["messages"][-2].content

        # Check if the report was successfully fetched
        with open(latest_path, "r", encoding="utf-8") as f:
            content = f.read()

        if os.path.isdir(latest_path):
            shutil.rmtree(latest_path)
        else:
            logger.warning(f"❗ Skipping delete, not a directory: {latest_path}")
        # Remove HTML tags
        soup = BeautifulSoup(content, "html.parser")
        text_only = soup.get_text()
        # Create and store RAG engine in the service
        rag_service.add_engine(state["stock_ticker"], documents=[text_only])

        logger.info(f"Created RAG engine for {state['stock_ticker']} annual report")
        rag_engine = rag_service.get_engine(state["stock_ticker"])
        if not rag_engine:
            logger.error(f"Failed to create RAG engine for {state['stock_ticker']}")
            return Command(
                update={
                    "messages": [
                        {
                            "role": "assistant",
                            "content": f"I couldn't fetch the annual report for {state['stock_ticker']}. Please try again or check if the ticker is correct.",
                            "name": "annual_report",
                        }
                    ],
                },
                goto="supervisor",
            )

    # Get the query from the last message
    rag_query = state["messages"][-1].content
    logger.info(f"Querying RAG engine for {state['stock_ticker']} with: {rag_query}")

    # Query the RAG engine
    rag_response = rag_engine.custom_query(rag_query)

    # Log the response
    logger.info(
        f"ANNUAL REPORT NODE - Retrieved RAG Response for {state['stock_ticker']}: {rag_response}"
    )

    # Return the response
    return Command(
        update={
            "messages": [
                {
                    "role": "assistant",
                    "content": rag_response.response,
                    "name": "annual_report",
                }
            ],
        },
        goto="supervisor",
    )
