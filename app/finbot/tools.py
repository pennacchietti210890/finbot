from typing import Annotated, List

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool

import yfinance as yf
import json


@tool
def get_historical_prices(ticker: str, day: int = 365) -> str:
    """
    Fetches historical stock prices and returns a JSON string containing:
    - "dates": List of date strings (YYYY-MM-DD)
    - "prices": Corresponding stock closing prices

    Only return the JSON.
    Example output:
    {"dates": ["2025-02-06", "2025-02-07"], "prices": [414.9879150390625, 408.9300537109375]}


    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{day}d")

        if not hist.empty:
            stock_data = {
                "dates": hist.index.strftime("%Y-%m-%d").tolist(),
                "prices": hist["Close"].values.tolist(),
            }
            return json.dumps(stock_data)  # 🔹 Enforce JSON output
        else:
            return json.dumps({"error": "No data available"})
    except Exception as e:
        return json.dumps({"error": str(e)})


repl = PythonREPL()


def python_repl_tool(
    code: Annotated[
        str, "The Python code to execute to generate your chart using Plotly."
    ]
):
    """Executes Python code in a controlled REPL environment and returns the result properly formatted."""

    try:
        local_vars = {}  # Store execution results
        exec(code, globals(), local_vars)  # ✅ Execute code

        # ✅ Ensure we return the Plotly figure HTML output inside a dictionary
        if "fig" in local_vars and isinstance(local_vars["fig"], go.Figure):
            return {
                "chart_html": local_vars["fig"].to_html()
            }  # ✅ Return dictionary instead of raw HTML

        return {"error": "Execution succeeded but no figure was detected."}

    except IndentationError as e:
        return {"error": f"IndentationError: {str(e)}"}

    except SyntaxError as e:
        return {"error": f"SyntaxError: {str(e)}"}

    except BaseException as e:
        return {"error": f"Execution failed: {repr(e)}"}
