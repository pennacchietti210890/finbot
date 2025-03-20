import os
from typing import Dict, Any, List
from openai import OpenAI
import yfinance as yf
import pandas as pd
import json
import logging
from dotenv import load_dotenv
import os

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(env_path)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class LLMService:
    """
    Service for interacting with Large Language Models and processing financial data.
    """
    
    @staticmethod
    def process_query(query: str) -> Dict[str, Any]:
        """
        Process a user query about financial information using LLM.
        """
        try:
            # This is a placeholder implementation
            # In a real implementation, you would:
            # 1. Parse the user query to identify companies/tickers using LLM
            # 2. Fetch relevant financial data
            # 3. Generate charts if needed
            # 4. Use LLM to generate a comprehensive response
            
            logger.info(f"Processing query: {query}")
            
            # Placeholder response
            response = {
                "text": f"I received your query: '{query}'. However, I'm still learning about financial data. Soon I'll be able to provide detailed information and charts for S&P 500 companies.",
                "charts": []
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}")
            raise e
    
    @staticmethod
    def get_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
        """
        Fetch stock data for a given ticker.
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            return hist
        except Exception as e:
            logger.error(f"Error fetching stock data for {ticker}: {str(e)}")
            raise e
    
    @staticmethod
    def generate_chart_data(data: pd.DataFrame, chart_type: str = "line") -> Dict[str, Any]:
        """
        Generate chart data from pandas DataFrame.
        """
        # Placeholder - actual implementation would convert the data
        # to a format understood by the frontend charting library
        return {
            "type": chart_type,
            "data": json.loads(data.to_json(orient="table"))
        } 