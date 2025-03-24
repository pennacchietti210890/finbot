from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated, Literal
from app.llm.llm_service import LLMService
from app.finbot.agents import create_stock_price_agent, create_financials_agent
import os
import logging
from dotenv import load_dotenv

# Get the logger for this module
logger = logging.getLogger(__name__)

env_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"
)
load_dotenv(env_path)

agents_llm = LLMService(
    llm_provider="openai", model_name="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")
).client


class State(TypedDict):
    next: str
    messages: Annotated[list, add_messages]
    stock_data: str
    financials: str
    stock_ticker: str 


def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> StateGraph:
    options = ["FINISH"] + members
    system_prompt = """You are a supervisor tasked with managing a conversation between the following workers: {workers}.
    
    Given the user request and the conversation history so far, respond with the worker to act next or FINISH if the request has been fully addressed.
    
    Only select one worker at a time. Wait for that worker to complete before selecting another.
    
    Important rules:
    1. If a user asks for stock price or chart, use stock_price first, then chart, then FINISH.
    2. If a user asks for financial metrics or statements, use financial_statements_and_metrics, then FINISH.
    3. ONLY proceed to the next worker if it's truly necessary to complete the user's request.
    4. After chart is called, the request for stock price and charts should be considered complete - use FINISH.
    5. After financial_statements_and_metrics is called, the request for financial data should be considered complete - use FINISH.
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
            logger.info(f"SUPERVISOR NODE - Previous Stock ticker: {state['stock_ticker']}")
            ticker_response = llm.invoke(ticker_messages)
            #if not state["stock_ticker"]:
            state["stock_ticker"] = ticker_response.content
            logger.info(f"SUPERVISOR NODE - Raw Ticker Response: {ticker_response}")
            logger.info(f"SUPERVISOR NODE - Stock ticker: {state['stock_ticker']}")
            if state["stock_ticker"] and state["stock_ticker"] != ticker_response.content:
                logger.info(f"SUPERVISOR NODE - Stock ticker has changed. Restarting workflow with new ticker: {ticker_response.content}")
                messages = [
                    {"role": "system", "content": formatted_prompt},
                ]
                state["messages"] = messages


            logger.info("Calling LLM for routing decision...")
            response = llm.with_structured_output(Router).invoke(messages)
            logger.info(f"SUPERVISOR NODE - Raw Response: {response}")

            ticker_response = llm.invoke(ticker_messages)
            logger.info(f"SUPERVISOR NODE - Raw Ticker Response: {ticker_response}")
            
            # Handle both response formats
            if isinstance(response, dict):
                if "next" in response:
                    goto = response["next"]
                elif "type" in response and "properties" in response and isinstance(response["properties"], dict):
                    # Handle nested format
                    if "next" in response["properties"]:
                        goto = response["properties"]["next"]
                    else:
                        logger.warning("Cannot find 'next' in nested properties, defaulting to FINISH")
                        goto = "FINISH"
                else:
                    logger.warning("Unexpected response format, defaulting to FINISH")
                    goto = "FINISH"
            else:
                logger.warning(f"Response is not a dict: {type(response)}, defaulting to FINISH")
                goto = "FINISH"
            
            if goto == "FINISH":
                logger.info("Router decided to FINISH")
                goto = END
            else:
                logger.info(f"Router selected next worker: {goto}")
            
            return Command(goto=goto, update={"next": goto, "stock_ticker": state["stock_ticker"]})
        except Exception as e:
            logger.error(f"Error in supervisor node: {str(e)}", exc_info=True)
            # If there's an error, default to ending the workflow
            return Command(goto=END, update={"next": "FINISH"})

    return supervisor_node


def stock_price_node(state: State) -> Command[Literal["supervisor"]]:
    logger.info("STOCK PRICE NODE - Processing request")

    stock_price_agent = create_stock_price_agent(agents_llm)
    result = stock_price_agent.invoke(state)

    # ✅ Accessing Intermediate response tool call observation
    stock_data = result["messages"][-2].content  # Already a JSON dict
    last_message = result["messages"][-1].content  # Already a JSON dict
    logger.info("STOCK PRICE NODE - Retrieved stock data and message")
    
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


def chart_node(state: State) -> Command[Literal["supervisor"]]:
    logger.info("CHART NODE - Processing request")
    stock_data = state.get("stock_data", None)
    if not stock_data or "error" in stock_data:
        logger.warning("CHART NODE - No valid stock data available to plot")
        return Command(
            update={
                "messages": [
                    {
                        "role": "assistant",
                        "content": "No valid stock data available to plot.",
                        "name": "chart",
                    }
                ]
            },
            goto="supervisor",
        )

    logger.info("CHART NODE - Data available for plotting")
    return Command(
        update={
            "messages": [
                {
                    "role": "assistant",
                    "content": "✅ Data is ready to be plotted.",
                    "name": "chart",
                }
            ],
            "stock_data": stock_data,
        },
        goto="supervisor",
    )


def financials_node(state: State) -> Command[Literal["supervisor"]]:
    logger.info("FINANCIALS NODE - Processing request")
    financials_agent = create_financials_agent(agents_llm)
    result = financials_agent.invoke(state)

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
