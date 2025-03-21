from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from app.finbot.graphs import create_graph
import logging
import uvicorn
import os
import uuid
from dotenv import load_dotenv
from app.llm.llm_service import LLMService
logger = logging.getLogger(__name__)

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
message_history = []

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    text: str
    charts_data: str = "{}"
    session_id: str


@app.get("/")
async def root():
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
        
        # Get or create graph for this session
        if session_id not in active_graphs:
            logger.info(f"Creating new graph for session {session_id}")
            finbot_graph = create_graph(
                LLMService(
                    llm_provider="openai",
                    model_name="gpt-4o",
                    api_key=os.getenv("OPENAI_API_KEY"),
                ).client
            )
            active_graphs[session_id] = finbot_graph

            # Create the initial state with the user's query
            initial_state = {
                "messages": [{"role": "user", "content": request.query}],
                "next": None,
                "stock_data": None
            }
            # Process the user query through the LangGraph        # Execute the graph
            response = finbot_graph.invoke(
                initial_state,
                {"recursion_limit": 100},
            )
        else:
            logger.info(f"Using existing graph for session {session_id}")
            finbot_graph = active_graphs[session_id]
    
            # Process the user query through the LangGraph        # Execute the graph
            response = finbot_graph.invoke(
                {"messages": message_history + [{"role": "user", "content": request.query}]},
                {"recursion_limit": 100},
            )

        # Extract the response text from the messages
        response_text = "I couldn't process your request."
        if response.get("messages") and len(response["messages"]) > 0:
            response_text = response["messages"][-1].content
            message_history.append({"role": "assistant", "content": response_text})
        
        # Extract chart data from stock_data
        charts_data = response.get("stock_data", "{}")
        
        logger.info(f"Response processed: {response_text[:50]}...")
        
        return ChatResponse(
            text=response_text, 
            charts_data=charts_data,
            session_id=session_id
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
