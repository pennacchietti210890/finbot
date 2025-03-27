from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Union
from app.finbot.graphs import create_graph
import logging
import uvicorn
import os
import uuid
import json
import sys
from datetime import datetime
from dotenv import load_dotenv
from app.llm.llm_service import LLMService
from app.finbot.nodes import State
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Configure detailed logging
# Make sure we're using absolute paths for log files
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(base_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"finbot_{datetime.now().strftime('%Y%m%d')}.log")

# Create a file handler for the log file
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create a formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Configure the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Remove any existing handlers to avoid duplicates
if root_logger.handlers:
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Get a logger for this module
logger = logging.getLogger(__name__)
logger.info("Logger initialized - This message should appear in the log file")
logger.info(f"Logging to file: {log_file}")

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(env_path)

app = FastAPI(title="FinBot API", description="Financial chatbot API using LLMs")

# CORS to allow requests from Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Store active graphs by session ID
active_graphs = {}


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    text: str
    charts_data: str = "{}"
    financials_charts_data: str = "{}"
    macroeconomics_charts_data: str = "{}"
    session_id: str


def convert_messages_to_openai_format(
    messages: List[Union[HumanMessage, AIMessage, dict]]
) -> List[dict]:
    formatted_messages = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            formatted_messages.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, dict):  # Handle dict-based messages
            formatted_messages.append(
                {
                    "role": msg.get(
                        "role", "user"
                    ),  # Default to 'user' if role is missing
                    "content": msg["content"],
                }
            )
        else:
            raise ValueError(f"Unsupported message type: {type(msg)}")

    return formatted_messages


@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {
        "message": "FinBot API is running. Use /chat endpoint to interact with the bot."
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a user query about financial information using LangGraph.
    """
    # Process the user query through the LangGraph
    logger.info(f"Processing query: {request.query}")
    try:
        # Get or create session ID
        session_id = request.session_id
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.info(f"Created new session ID: {session_id}")
        else:
            logger.info(f"Using existing session ID: {session_id}")

        # Get or create graph for this session
        if session_id not in active_graphs:
            logger.info(f"Creating new graph for session {session_id}")
            finbot_graph = create_graph(
                LLMService(
                    llm_provider="openai",
                    model_name="gpt-4o-mini",
                    api_key=os.getenv("OPENAI_API_KEY"),
                ).client
            )
            # Store [graph, message_history, last_state, stock_data]
            active_graphs[session_id] = {
                "graph": finbot_graph,
                "message_history": [],
                "stock_data": None,
                "financials": None,
                "stock_ticker": None,
                "macroeconomics_data": None,
            }

            # Create the initial state with the user's query
            initial_state = State(
                messages=[{"role": "user", "content": request.query}],
                next=None,
                stock_data=None,
                financials=None,
                stock_ticker=None,
                macroeconomics_data=None,
            )

            logger.info(f"SESSION {session_id} - INITIAL STATE:")
            logger.info(f"Messages: {json.dumps(initial_state['messages'])}")
            logger.info(f"Next: {initial_state['next']}")
            logger.info(f"Stock data: {initial_state['stock_data']}")
            logger.info(f"Financials: {initial_state['financials']}")
            logger.info(f"Stock ticker: {initial_state['stock_ticker']}")
            logger.info(f"Macroeconomics data: {initial_state['macroeconomics_data']}")
            # Process the user query through the LangGraph
            response = finbot_graph.invoke(
                initial_state,
                {"recursion_limit": 100},
            )

        else:
            logger.info(f"Using existing graph for session {session_id}")
            session_data = active_graphs[session_id]
            finbot_graph = session_data["graph"]
            graph_message_history = session_data["message_history"]
            stock_data = session_data["stock_data"]
            financials = session_data["financials"]
            stock_ticker = session_data["stock_ticker"]
            macroeconomics_data = session_data["macroeconomics_data"]
            # Log current message history in session
            logger.info(f"SESSION {session_id} - CURRENT MESSAGE HISTORY:")
            logger.info(f"Message history length: {len(graph_message_history)}")
            for i, msg in enumerate(graph_message_history):
                if hasattr(msg, "content"):
                    logger.info(
                        f"Message {i + 1}: Role={msg.type if hasattr(msg, 'type') else 'unknown'}, Content={msg.content[:100]}..."
                    )
                else:
                    logger.info(f"Message {i + 1}: {str(msg)[:100]}...")

            # Create state for this invocation - must include all keys from the State type
            invoke_state = State(
                messages=convert_messages_to_openai_format(graph_message_history)
                + [{"role": "user", "content": request.query}],
                next=None,  # Initialize next to None explicitly
                stock_data=stock_data,
                financials=financials,
                stock_ticker=stock_ticker,
                macroeconomics_data=macroeconomics_data,
            )

            logger.info(f"SESSION {session_id} - INVOKING GRAPH WITH STATE:")
            logger.info(f"Last user message: {request.query}")
            logger.info(f"Stock data present: {stock_data is not None}")
            logger.info(f"Stock data: {stock_data}")
            logger.info(f"Financials data present: {financials is not None}")
            logger.info(f"Financials data: {financials}")
            logger.info(f"Macroeconomics data present: {macroeconomics_data is not None}")
            logger.info(f"Macroeconomics data: {macroeconomics_data}")
            logger.info(
                f"Next value: {invoke_state.get('next')}"
            )  # Log the 'next' value

            # Process the user query through the LangGraph
            response = finbot_graph.invoke(
                invoke_state,
                {"recursion_limit": 100},
            )

        # Log the response state
        logger.info(f"SESSION {session_id} - GRAPH RESPONSE STATE:")
        logger.info(f"Response keys: {list(response.keys())}")

        # Log message history
        if "messages" in response:
            logger.info(f"Response messages count: {len(response['messages'])}")
            if response["messages"]:
                last_msg = response["messages"][-1]
                logger.info(f"Last message role: {last_msg.type}")
                logger.info(f"Last message content: {last_msg.content[:100]}...")
                if hasattr(last_msg, "name"):
                    logger.info(f"Last message name: {last_msg.name}")

        # Log stock data
        if "stock_data" in response:
            stock_data_str = str(response.get("stock_data", ""))
            logger.info(
                f"Stock data available: {bool(stock_data_str and stock_data_str != '{}')}"
            )
            logger.info(
                f"Stock data preview: {stock_data_str[:100]}..."
                if stock_data_str and len(stock_data_str) > 100
                else stock_data_str
            )

        # Log financials data
        if "financials" in response:
            financials_str = str(response.get("financials", ""))
            logger.info(
                f"Financials data available: {bool(financials_str and financials_str != '{}')}"
            )
            logger.info(
                f"Financials data preview: {financials_str[:100]}..."
                if financials_str and len(financials_str) > 100
                else financials_str
            )

        # Log Macroeconomics data
        if "macroeconomics_data" in response:
            macroeconomics_str = str(response.get("macroeconomics_data", ""))
            logger.info(
                f"Macroeconomics data available: {bool(macroeconomics_str and macroeconomics_str != '{}')}"
            )
            logger.info(
                f"Macroeconomics data preview: {macroeconomics_str[:100]}..."
                if macroeconomics_str and len(macroeconomics_str) > 100
                else macroeconomics_str
            )

        # Extract the response text from the messages
        response_text = "I couldn't process your request."
        if response.get("messages") and len(response["messages"]) > 0:
            logger.info(response.get("stock_ticker", "{}"))
            response_text = response["messages"][-1].content
            active_graphs[session_id]["message_history"].extend(response["messages"])
            active_graphs[session_id]["stock_data"] = response.get("stock_data", "{}")
            active_graphs[session_id]["financials"] = response.get("financials", "{}")
            active_graphs[session_id]["stock_ticker"] = response.get(
                "stock_ticker", "{}"
            )
            active_graphs[session_id]["macroeconomics_data"] = response.get(
                "macroeconomics_data", "{}"
            )

        # Extract chart data from stock_data
        if response["messages"][-1].name == "stock_price_chart":
            charts_data = response.get("stock_data", "{}")
            financials_charts_data = "{}"
            macroeconomics_charts_data = "{}"
            logger.info(f"Chart data detected, size: {len(str(charts_data))} chars")
        elif response["messages"][-1].name == "financials_chart":
            charts_data = "{}"
            macroeconomics_charts_data = "{}"
            financials_charts_data = response.get("financials_chart_data", "{}")
            logger.info(
                f"Financials chart data detected, size: {len(str(financials_charts_data))} chars"
            )
            logger.info(f"Financials chart data: {financials_charts_data}")
        elif response["messages"][-1].name == "macroeconomics_chart":
            charts_data = "{}"
            financials_charts_data = "{}"
            macroeconomics_charts_data = response.get("macroeconomics_data", "{}")
            logger.info(
                f"Macroeconomics chart data detected, size: {len(str(macroeconomics_charts_data))} chars"
            )
            logger.info(f"Macroeconomics chart data: {macroeconomics_charts_data}")
        else:
            charts_data = "{}"
            financials_charts_data = "{}"
            macroeconomics_charts_data = "{}"
            logger.info("No chart data in response")

        logger.info(f"Response processed: {response_text[:50]}...")

        return ChatResponse(
            text=response_text,
            charts_data=charts_data,
            financials_charts_data=financials_charts_data,
            macroeconomics_charts_data=macroeconomics_charts_data,
            session_id=session_id,
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Entry point for running the FastAPI application"""
    logger.info("Starting FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
