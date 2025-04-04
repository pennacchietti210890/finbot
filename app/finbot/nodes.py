from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import Tool, BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function

from langchain_openai import ChatOpenAI

from typing import TypedDict, Annotated, Literal, Any, Dict
from app.llm.llm_service import LLMService
from app.finbot.agents import (
    create_stock_price_agent,
    create_financials_agent,
    create_financials_chart_agent,
    create_macroeconomics_agent,
    create_news_search_agent,
    create_annual_report_agent,
)
from app.finbot.mcp_agents import (
    create_mcp_stock_price_agent,
    create_mcp_financials_agent,
    create_mcp_financials_chart_agent,
    create_mcp_macroeconomics_agent,
    create_mcp_news_search_agent,
    create_mcp_annual_report_agent,
)
from app.llm.rag_query_engine import RAGEngine
from pydantic import BaseModel, Field, create_model
import os
import logging
from dotenv import load_dotenv
import json
import asyncio
from bs4 import BeautifulSoup
import shutil
# Get the logger for this module
logger = logging.getLogger(__name__)

env_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"
)
load_dotenv(env_path)

agents_llm = LLMService(
    llm_provider="openai", model_name="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")
).client


mcp_script_path = os.path.abspath("app/finbot/mcp_finbot/mcp_servers_finbot.py")


# Low-level MCP request
async def send_mcp_message(proc, message: dict):
    proc.stdin.write((json.dumps(message) + "\n").encode())
    await proc.stdin.drain()
    line = await proc.stdout.readline()
    return json.loads(line)

# Tool calling function
async def call_tool(proc, tool_name: str, arguments: dict) -> str:
    msg = {
        "jsonrpc": "2.0",
        "id": 42,  # could randomize or increment
        "method": "tools/call",
        "params": {
            "tool_name": tool_name,
            "arguments": arguments
        }
    }
    response = await send_mcp_message(proc, msg)
    logger.info("🧪 MCP raw response:", response)
    # Check if result exists
    if "result" not in response:
        raise ValueError(f"❌ MCP tool call failed, response: {response}")
    return response["result"]["output"]

# Wrap raw MCP tool schema into LangChain @tool
class MCPTool(BaseTool):
    name: str
    description: str
    proc: Any
    tool_name: str
    input_schema: Dict[str, Any]

    def __init__(self, **kwargs):
        # Dynamically create a Pydantic model for input schema
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

        kwargs["args_schema"] = schema  # ✅ tell LangChain this is your input model
        super().__init__(**kwargs)

    async def _arun(self, *args, **kwargs):
        tool_input = args[0] if args else kwargs.get("tool_input", {})
        logger.info(f"{args}")
        logger.info(f"{kwargs}")
        result = await call_tool(self.proc, self.tool_name, kwargs)
        return result if isinstance(result, str) else json.dumps(result)

    def _run(self, tool_input: dict, **kwargs):
        raise NotImplementedError("Only async supported")

def wrap_tool_for_langchain(tool_schema, proc):
    tool_name = tool_schema["name"]
    description = tool_schema.get("description", "")

    return MCPTool(
        name=tool_name,
        description=description,
        proc=proc,
        tool_name=tool_schema["name"],
        input_schema=tool_schema["inputSchema"],  # from MCP tools/list
    )

# ✅ Main helper: load LangChain tools from MCP server
async def load_tools_from_mcp_server(proc):
    msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list"
    }
    response = await send_mcp_message(proc, msg)
    raw_tools = response["result"]["tools"]
    wrapped = []
    for tool_def in raw_tools:
        try:
            tool = wrap_tool_for_langchain(tool_def, proc)
            if tool is None:
                logger.info(f"⚠️ Tool {tool_def['name']} returned None!")
            else:
                wrapped.append(tool)
        except Exception as e:
            logger.info(f"❌ Failed to wrap tool {tool_def['name']}: {e}")

    return wrapped


class State(TypedDict):
    next: str
    messages: Annotated[list, add_messages]
    stock_data: str
    financials: str
    stock_ticker: str
    financials_chart_data: str
    macroeconomics_data: str


def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> StateGraph:
    options = ["FINISH"] + members
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

            if goto == "FINISH":
                logger.info("Router decided to FINISH")
                goto = END
            else:
                logger.info(f"Router selected next worker: {goto}")

            return Command(
                goto=goto, update={"next": goto, "stock_ticker": state["stock_ticker"]}
            )
        except Exception as e:
            logger.error(f"Error in supervisor node: {str(e)}", exc_info=True)
            # If there's an error, default to ending the workflow
            return Command(goto=END, update={"next": "FINISH"})

    return supervisor_node


async def stock_price_node(state: State) -> Command[Literal["supervisor"]]:
    logger.info("STOCK PRICE NODE - Processing request")
    
    proc = await asyncio.create_subprocess_exec(
        "python", "app/finbot/mcp_finbot/mcp_servers_finbot.py",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    wrapped_tools = await load_tools_from_mcp_server(proc)
    tool_map = {tool.name: tool for tool in wrapped_tools}
    
    stock_price_agent = create_mcp_stock_price_agent(agents_llm, tool_map)
    result = await stock_price_agent.ainvoke(state)

    # ✅ Accessing Intermediate response tool call observation
    stock_data = result["messages"][-2].content  # Already a JSON dict
    last_message = result["messages"][-1].content  # Already a JSON dict
    logger.info("STOCK PRICE NODE - Retrieved stock data and message")
    logger.info(f"{stock_data}")
    
    return Command(
        update={
            "messages": [
                {"role": "assistant", "content": last_message, "name": "stock_price"}
            ],
            # [HumanMessage(content=last_message, name="stock_price")],
            "stock_data": stock_data,  # ✅ Directly store tool output
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

    proc = await asyncio.create_subprocess_exec(
        "python", "app/finbot/mcp_finbot/mcp_servers_finbot.py",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    wrapped_tools = await load_tools_from_mcp_server(proc)
    tool_map = {tool.name: tool for tool in wrapped_tools}
    
    financials_agent = create_mcp_financials_agent(agents_llm, tool_map)
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


def financials_chart_node(state: State) -> Command[Literal["supervisor"]]:
    logger.info("FINANCIALS CHART NODE - Processing request")

    financials_chart_agent = create_financials_chart_agent(agents_llm)
    financials_chart_response = financials_chart_agent.invoke(state)

    last_message = financials_chart_response["messages"][-1].content
    financials_chart_data = financials_chart_response.get("structured_response", None)
    logger.info(f"FINANCIALS CHART NODE - Retrieved financials chart data: {financials_chart_data}")

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

    proc = await asyncio.create_subprocess_exec(
        "python", "app/finbot/mcp_finbot/mcp_servers_finbot.py",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    wrapped_tools = await load_tools_from_mcp_server(proc)
    tool_map = {tool.name: tool for tool in wrapped_tools}
    
    macroeconomics_agent = create_mcp_macroeconomics_agent(agents_llm, tool_map)
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
    
    proc = await asyncio.create_subprocess_exec(
        "python", "app/finbot/mcp_finbot/mcp_servers_finbot.py",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    wrapped_tools = await load_tools_from_mcp_server(proc)
    tool_map = {tool.name: tool for tool in wrapped_tools}
    
    news_search_agent = create_mcp_news_search_agent(agents_llm, tool_map)
    result = await news_search_agent.ainvoke(state)

    return Command(
        update={
            "messages": [
                HumanMessage(content= result["messages"][-1].content, name="news_search")
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
        logger.info(f"No RAG engine found for {state['stock_ticker']}, need to fetch annual report first")
        
        proc = await asyncio.create_subprocess_exec(
            "python", "app/finbot/mcp_finbot/mcp_servers_finbot.py",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        wrapped_tools = await load_tools_from_mcp_server(proc)
        tool_map = {tool.name: tool for tool in wrapped_tools}
        
        annual_report_agent = create_mcp_annual_report_agent(agents_llm, tool_map)
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
                        {"role": "assistant", "content": f"I couldn't fetch the annual report for {state['stock_ticker']}. Please try again or check if the ticker is correct.", "name": "annual_report"}
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
    logger.info(f"ANNUAL REPORT NODE - Retrieved RAG Response for {state['stock_ticker']}: {rag_response}")
    
    # Return the response
    return Command(
        update={
            "messages": [
                {"role": "assistant", "content": rag_response.response, "name": "annual_report"}
            ],
        },
        goto="supervisor",
    )
