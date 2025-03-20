from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
from .llm_service import LLMService
import logging
import uvicorn

logger = logging.getLogger(__name__)

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
    charts: List[Dict[str, Any]] = []

@app.get("/")
async def root():
    return {"message": "FinBot API is running. Use /chat endpoint to interact with the bot."}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a user query about financial information.
    """
    try:
        # Use the LLM service to process the query
        response = LLMService.process_query(request.query)
        return ChatResponse(**response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 


def main():
    """Entry point for running the FastAPI application"""
    logger.info("Starting FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()