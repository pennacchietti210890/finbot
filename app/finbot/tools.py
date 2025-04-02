from typing import Annotated, List, TypedDict

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool

import yfinance as yf
import pandas_datareader as pdr
import json
import os
from dotenv import load_dotenv

from langchain_tavily import TavilySearch

env_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"
)
load_dotenv(env_path)

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


class KeyMetrics(TypedDict):
    market_cap: float
    pe_ratio: float
    forward_pe_ratio: float
    eps: float
    revenues: float
    profit_margin: float
    debt_to_equity: float
    current_ratio: float


@tool
def get_financials(ticker: str) -> str:
    """
    Fetches the current and historical financial statements for one single stock, given the stock ticker, and returns a JSON string containing:
    - "balance sheet": A Table containing information on the Balance Sheet
    - "income statement": A Table containing information on the Income Statement
    - "cash flow": A Table containing information on the Cash Flow Statement
    - "metrics": A Dictionary containing key financial metrics and ratios such as price to earnings ratio, leverage ratio

    **Only choose this tool if the user asks for financial statements or ratios.**
    """
    try:
        stock = yf.Ticker(ticker)
        financials = {
            "balance sheet": stock.balance_sheet.to_dict(),
            "income statement": stock.financials.to_dict(),
            "cash flow": stock.cashflow.to_dict(),
        }

        financials["balance sheet"] = {
            key.strftime("%Y-%m-%d"): financials["balance sheet"][key]
            for key in financials["balance sheet"]
        }
        financials["income statement"] = {
            key.strftime("%Y-%m-%d"): financials["income statement"][key]
            for key in financials["income statement"]
        }
        financials["cash flow"] = {
            key.strftime("%Y-%m-%d"): financials["cash flow"][key]
            for key in financials["cash flow"]
        }

        info = stock.info
        # Ensure info is not empty
        if not info:
            print(f"No financial metrics available for {ticker}")
            return financials

        financials["key_metrics"] = KeyMetrics(
            market_cap=info.get("marketCap", 0.0),
            pe_ratio=info.get("trailingPE", 0.0),
            forward_pe_ratio=info.get("forwardPE", 0.0),
            eps=info.get("trailingEps", 0.0),
            profit_margin=info.get("profitMargins", 0.0),
            debt_to_equity=info.get("debtToEquity", 0.0),
            current_ratio=info.get("currentRatio", 0.0),
        )

        return json.dumps(financials)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_macroeconomic_series(
    macro_indicator: str, start_year: str, end_year: str) -> str:
    """
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
    """
    try:
        macro_data = pdr.get_data_fred(macro_indicator, start_year, end_year)

        if macro_data.shape[0] == 0:
            return json.dumps({"error": "No data available"})

        macro_dict = {
            "dates": [ts.strftime("%Y-%m-%d") for ts in list(macro_data.index)],
            "values": [float(val[0]) for val in macro_data.values],
        }

        return json.dumps(macro_dict)
    except Exception as e:
        return json.dumps({"error": str(e)})


search_tool = TavilySearch(
    max_results=3,
    topic="general",
)