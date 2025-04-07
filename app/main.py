import json
import logging
import os
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from app.finbot.graphs import create_graph
from app.finbot.nodes import State
from app.llm.llm_service import LLMService

# Configure detailed logging
# Make sure we're using absolute paths for log files
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(base_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"finbot_{datetime.now().strftime('%Y%m%d')}.log")

# root logger config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
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
active_graph_config = {}


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
            logger.info(f"Creating graph...")
            finbot_graph = create_graph(
                LLMService(
                    llm_provider="openai",
                    model_name="gpt-4o-mini",
                    api_key=os.getenv("OPENAI_API_KEY"),
                ).client
            )
            logger.info(f"Creating new graph Config for new session ID: {session_id}")
            session_config = {"configurable": {"thread_id": f"{session_id}"}}
            active_graph_config[session_id] = [session_config, finbot_graph]
        else:
            session_config = active_graph_config[session_id][0]
            finbot_graph = active_graph_config[session_id][1]
            logger.info(f"Using existing config for session ID: {session_config}")

        # Process the user query through the LangGraph
        response = await finbot_graph.ainvoke(
            {"messages": [{"role": "user", "content": request.query}]},
            config=session_config,
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
