from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
from app.finbot.graphs import create_graph
import logging
import uvicorn
import os
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


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    text: str
    charts_data: str = "{}"


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
    try:
        # Initialize the LangGraph with the LLM
        finbot_graph = create_graph(
            LLMService(
                llm_provider="openai",
                model_name="gpt-4o",
                api_key=os.getenv("OPENAI_API_KEY"),
            ).client
        )
        
        # Process the user query through the LangGraph
        logger.info(f"Processing query: {request.query}")
        
        # Create the initial state with the user's query
        initial_state = {
            "messages": [{"role": "user", "content": request.query}],
            "next": None,
            "stock_data": None
        }
        
        # Execute the graph
        response = finbot_graph.invoke(
            initial_state,
            {"recursion_limit": 100},
        )
        
        # Extract the response text from the messages
        response_text = "I couldn't process your request."
        if response.get("messages") and len(response["messages"]) > 0:
            response_text = response["messages"][-1].content
        
        # Extract chart data from stock_data
        charts_data = response.get("stock_data", "{}")
        
        logger.info(f"Response processed: {response_text[:50]}...")
        
        return ChatResponse(
            text=response_text, 
            charts_data=charts_data
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
