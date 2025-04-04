import asyncio
import json
import os
import sys
from typing import Dict
import traceback

from app.finbot.mcp_finbot.mcp_tools import get_historical_prices, get_financials, get_macroeconomic_series, search_news, get_annual_report

import requests
from dotenv import load_dotenv
import logging

# Configure logging
logger = logging.getLogger(__name__)

env_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"
)
load_dotenv(env_path)


def debug_log(msg):
    print(f"[MCP-DEBUG] {msg}", file=sys.stderr, flush=True)

async def handle_request(request: dict):
    debug_log(f"📦 Handling request: {request}")
    method = request.get("method")
    id_ = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": id_,
            "result": {
                "protocolVersion": "2024-11-05",  # ✅ REQUIRED
                "serverInfo": {
                    "name": "finbot-mcp",
                    "version": "0.1.0"
                },
                "capabilities": {
                    "listChanged": True
                }
            }
        }
   # Optional notification (no response required — but some clients expect it)
    elif method == "notifications/initialized":
        return {"id": None, "result": {}}

    elif method == "tools/list":
        debug_log("🔍 Handling tools/list")
        return  {
            "jsonrpc": "2.0",
            "id": id_,
            "result": {
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
                           "description": "Stock ticker symbol (e.g., AAPL, TSLA)."
                       },
                       "day": {
                           "type": "string",
                           "description": "Number of days to fetch historical prices for."
                       }
                   },
                   "required": ["ticker", "day"]
               }
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
                           "description": "The stock ticker symbol (e.g., AAPL, TSLA)."
                       }
                   },
                   "required": ["ticker"]
               }
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
                           "description": "The FRED macro indicator series ID (e.g., GDPC1 for Real Gross Domestic Product)."
                       },
                       "start_year": {
                           "type": "int",
                           "description": "Start year for the data series, in format YYYY."
                       },
                       "end_year": {
                           "type": "int",
                           "description": "End year for the data series, in format YYYY."
                       }
                   },
                   "required": ["macro_indicator", "start_year", "end_year"]
               }
           },
           {
               "name": "search_news",
               "description": "Search for recent news articles using the Tavily API.",
               "inputSchema": {
                   "type": "object",
                   "properties": {
                       "query": {
                           "type": "string",
                           "description": "The search query (e.g., inflation, AI, interest rates)."
                       }
                   },
                   "required": ["query"]
               }
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
                           "description": "The stock ticker symbol (e.g., AAPL, TSLA)."
                       }
                   },
                   "required": ["ticker"]
               }
           }
            ]
        }
    }
        

    elif method == "tools/call":
        params = request.get("params", {})
        tool_name = params.get("tool_name")
        arguments = params.get("arguments", {})
        try:
            
            if tool_name == "echo":
                output_data = arguments["message"]
                return {"jsonrpc": "2.0", "id": id_, "result": {"output": output_data}}
            
            elif tool_name == "get_historical_prices":
                print(arguments)
                ticker = arguments["ticker"]
                day = int(arguments["day"])
                output_data = get_historical_prices(ticker, day)
                return {"jsonrpc": "2.0", "id": id_, "result": {"output": output_data}}

            elif tool_name == "get_financials":
                ticker = arguments["ticker"]
                output_data = get_financials(ticker)
                return {"jsonrpc": "2.0", "id": id_, "result": {"output": output_data}}

            elif tool_name == "get_macroeconomic_series":
                series_id = arguments["macro_indicator"]
                start_year = arguments["start_year"]
                end_year = arguments["end_year"]
                output_data = get_macroeconomic_series(series_id, start_year, end_year)
                return {"jsonrpc": "2.0", "id": id_, "result": {"output": output_data}}

            elif tool_name == "search_news":
                query = arguments.get("query")
                output_data = search_news(query)
                return {"jsonrpc": "2.0", "id": id_, "result": {"output": output_data}}

            elif tool_name == "get_annual_report":
                ticker = arguments["ticker"]
                output_data = get_annual_report(ticker)
                return {"jsonrpc": "2.0", "id": id_, "result": {"output": output_data}}

            else:
                debug_log(f"⚠️ Unknown method: {method}")
                return {"jsonrpc": "2.0", "id": id_, "error": {"message": f"Unknown tool: {tool_name}"}}

        except Exception as e:
            import traceback
            err = {
                "jsonrpc": "2.0",
                "error": {
                    "message": f"Server error: {str(e)}",
                    "trace": traceback.format_exc()
                }
            }
            return err
    return {"id": id_, "error": {"message": "Unknown method"}}


async def run_server():
    debug_log("🟢 MCP Server started and waiting for input...")

    while True:
        line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
        if not line:
            debug_log("⚪️ No more input. Exiting server.")
            break
        try:
            request = json.loads(line)
            response = await handle_request(request)
            debug_log(f"📤 Sending response: {response}")
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err = {
                "jsonrpc": "2.0",
                "error": {"message": f"Server error: {str(e)}"},
            }
            debug_log(f"❌ Error: {e}")
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(run_server())
