import asyncio
import json
import logging
import os
import sys
import traceback
from typing import Dict, Any

from fastapi import FastAPI, Request
from pydantic import BaseModel

import requests
from dotenv import load_dotenv

from app.finbot.mcp_finbot.mcp_tools import (
    get_annual_report,
    get_financials,
    get_historical_prices,
    get_macroeconomic_series,
    search_news,
)
import time
# Configure logging
logger = logging.getLogger(__name__)

env_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"
)
load_dotenv(env_path)

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}


def debug_log(msg: str) -> None:
    """
    Print debug information to stderr with MCP prefix for visibility.

    Args:
        msg: The message to log
    """
    logger.info(f"[MCP-DEBUG] {msg}", file=sys.stderr, flush=True)


@app.post("/tools/list")
def list_tools():
    debug_log("🔍 Handling tools/list")
    return  {
            "tools": [
                {
                    "name": "get_historical_prices",
                    "description": """
                        Fetches historical stock prices given a stock ticker and a number of days. Returns a JSON string containing:
                - "dates": List of date strings (YYYY-MM-DD)
                - "prices": Corresponding stock closing prices
                Only return the JSON.
                Example output:
                {"dates": ["2025-02-06", "2025-02-07"], "prices": [414.9879150390625, 408.9300537109375]}
            """,
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Stock ticker symbol (e.g., AAPL, TSLA).",
                            },
                            "day": {
                                "type": "string",
                                "description": "Number of days to fetch historical prices for.",
                            },
                        },
                        "required": ["ticker", "day"],
                    },
                },
                {
                    "name": "get_financials",
                    "description": """
                Fetches the current and historical financial statements for one single stock, given the stock ticker, and returns a JSON string containing:
                    - "balance sheet": A Table containing information on the Balance Sheet
                    - "income statement": A Table containing information on the Income Statement
                    - "cash flow": A Table containing information on the Cash Flow Statement
                    - "metrics": A Dictionary containing key financial metrics and ratios such as price to earnings ratio, leverage ratio


                    **Only choose this tool if the user asks for financial statements or ratios.**
            """,
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "The stock ticker symbol (e.g., AAPL, TSLA).",
                            }
                        },
                        "required": ["ticker"],
                    },
                },
                {
                    "name": "get_macroeconomic_series",
                    "description": """
                Fetches the a time series for the chosen macro economic indicator from FRED API.
                        Args:
                        macro_indicator: The MacroEconomic data series to fetch from FRED API
                        start_year: Start Year for the data series, in format YYYY
                        end_year: End year for the data series, in format YYYY


                        The Macro Economic indicators available, and their meaning, are listed in the below list:


                        'GDPC1': 'Real Gross Domestic Product',
                        'UNRATE': 'Unemployment Rate',
                        'DGS10': '10-Year Treasury Constant Maturity Rate',
                        'M2SL': 'M2 Money stock',
                        'WALCL': 'Weekly FED Balance sheet total assets',
                        'TREAST': 'USTs holdings at the FED',
                        'MBST': 'MBS Holdings of the FED',
                        'RESBALNS': 'Reserve Balances held at the FED',
                        'BOPGSTB': 'Trade Balance of goods and services',
                        'IEABC': 'Total value of imports',
                        'IEABCEX': 'Total value of exports',
                        'NAPM': 'ISM Manufacturing PMI',
                        'INDPRO': 'Industrial Production Index',
                        'IPMAN': 'Manufacturing Output',
                        'DGORDER': 'Durable Goods Orders',
                        'NHSL': 'New Home Sales',
                        'EXHOSLUSM495S': 'Existing Home Sales',
                        'PHSI': 'Pending Home Sales Index',
                        'HOUST': 'Total Housing Starts',
                        'PERMIT': 'Building Permits',
                        'FGRECPT': 'Federal Government Tax Receipts',
                        'FGEXPND': 'Federal Government Tax Expenditures',
                        'MTSDS133FMS': 'Federal Government Budget Surplus (or Deficit)',
                        'GFDEBTN': 'Federal Debt (Total Public Debt)',
                        'FGRECPT_ALT': 'Federal Government Receipts: Budget Receipts',
                        'A091RC1Q027SBEA': 'Federal Government Current Expenditures: Interest Payments'


                        Output:
                            Dictionary object of Dates:Values for the desired data series
            """,
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "macro_indicator": {
                                "type": "string",
                                "description": "The FRED macro indicator series ID (e.g., GDPC1 for Real Gross Domestic Product).",
                            },
                            "start_year": {
                                "type": "int",
                                "description": "Start year for the data series, in format YYYY.",
                            },
                            "end_year": {
                                "type": "int",
                                "description": "End year for the data series, in format YYYY.",
                            },
                        },
                        "required": ["macro_indicator", "start_year", "end_year"],
                    },
                },
                {
                    "name": "search_news",
                    "description": """Searches for recent news articles using Tavily based on a full user query. \
                            Use this tool whenever the user asks about current events, headlines, or what's happening. \
                            Pass the user's entire question or phrase as the `query` argument.
            """,
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query (e.g., tell me some news about AAPL).",
                            }
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "get_annual_report",
                    "description": """
            Fetches the latest annual report for one stock, given the stock ticker, and return a string containing the text content of the report.
            **Only choose this tool if the user asks questions about a company's or stock annual report (also known as 10k).**
            """,
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "The stock ticker symbol (e.g., AAPL, TSLA).",
                            }
                        },
                        "required": ["ticker"],
                    },
                },
            ]
        }

class ToolCall(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]



@app.post("/tools/call")
def call_tool(call: ToolCall):
    try:
        if call.tool_name  == "get_historical_prices":
            print(f"get_historical_prices: {call.arguments}")
            return {"output": get_historical_prices(call.arguments["ticker"], int(call.arguments["day"]))}

        elif call.tool_name == "get_financials":
            return {"output": get_financials(call.arguments["ticker"])}

        elif tool_name == "get_macroeconomic_series":
            return {
                "output": get_macroeconomic_series(
                    call.arguments["macro_indicator"],
                    call.arguments["start_year"],
                    call.arguments["end_year"],
                )
            }

        elif call.tool_name == "search_news":
            return {"output": search_news(call.arguments["query"])}
        elif call.tool_name == "get_annual_report":
            return {"output": get_annual_report(call.arguments["ticker"])}
        else:
            return {"error": f"Unknown tool: {call.tool_name}"}
    except Exception as e:
        return {"error": str(e)}

def main():
    """Entry point for running MCP Server via Fat   """
    logger.info("Starting MCP server")
    return app

def main_local():
    # Only run uvicorn directly when script is executed, not when imported
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5005)


if __name__ == "__main__":
    # Only run uvicorn directly when script is executed, not when imported
    main_local()
