import pytest
from unittest.mock import patch, MagicMock
import json
import os

from app.finbot.mcp_finbot.mcp_tools import (
    get_historical_prices,
    get_financials,
    get_macroeconomic_series,
    search_news,
    get_annual_report,
    KeyMetrics
)
import numpy as np
import pandas as pd

@pytest.fixture
def mock_yf_ticker():
    """Fixture providing a mock yfinance Ticker object."""
    mock_ticker = MagicMock()
    
    # Configure history method
    mock_history = MagicMock()
    mock_history.empty=False
    mock_history.index.strftime.return_value = np.array(["2023-01-01", "2023-01-02"])
    mock_history["Close"].values = np.array([150.25, 152.75])
    mock_ticker.history.return_value = mock_history
    
    # Configure balance_sheet, financials, and cashflow
    mock_ticker.balance_sheet = MagicMock()
    mock_ticker.balance_sheet.to_dict.return_value = {pd.Timestamp("2023-01-01"): {"assets": 1000}}
    
    mock_ticker.financials = MagicMock()
    mock_ticker.financials.to_dict.return_value = {pd.Timestamp("2023-01-01"): {"revenue": 500}}
    
    mock_ticker.cashflow = MagicMock()
    mock_ticker.cashflow.to_dict.return_value = {pd.Timestamp("2023-01-01"): {"operating_cash": 200}}
    
    # Configure info
    mock_ticker.info = {
        "marketCap": 2000000000,
        "trailingPE": 25.5,
        "forwardPE": 20.1,
        "trailingEps": 12.5,
        "profitMargins": 0.25,
        "debtToEquity": 0.5,
        "currentRatio": 1.5,
    }
    
    return mock_ticker


def test_get_historical_prices_success(mock_yf_ticker):
    """Test successful retrieval of historical prices."""
    
    with patch('app.finbot.mcp_finbot.mcp_tools.yf.Ticker', return_value=mock_yf_ticker):
        result = get_historical_prices("AAPL", 30)
        result_json = json.loads(result)
        
        assert "dates" in result_json
        assert "prices" in result_json
        assert len(result_json["dates"]) == 2
        assert len(result_json["prices"]) == 2
        assert result_json["dates"][0] == "2023-01-01"
        assert result_json["prices"][0] == 150.25


def test_get_historical_prices_empty():
    """Test handling of empty history data."""
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = MagicMock(empty=True)
    
    with patch('app.finbot.mcp_finbot.mcp_tools.yf.Ticker', return_value=mock_ticker):
        result = get_historical_prices("AAPL", 30)
        result_json = json.loads(result)
        
        assert "error" in result_json
        assert result_json["error"] == "No data available"


def test_get_historical_prices_exception():
    """Test handling of exceptions during data retrieval."""
    with patch('app.finbot.mcp_finbot.mcp_tools.yf.Ticker', side_effect=Exception("Test error")):
        result = get_historical_prices("AAPL", 30)
        result_json = json.loads(result)
        
        assert "error" in result_json
        assert "Test error" in result_json["error"]


def test_get_financials_success(mock_yf_ticker):
    """Test successful retrieval of financial data."""
    with patch('app.finbot.mcp_finbot.mcp_tools.yf.Ticker', return_value=mock_yf_ticker):
        result = get_financials("AAPL")
        result_json = json.loads(result)
        
        assert "balance sheet" in result_json
        assert "income statement" in result_json
        assert "cash flow" in result_json
        assert "key_metrics" in result_json
        assert result_json["key_metrics"]["market_cap"] == 2000000000
        assert result_json["key_metrics"]["pe_ratio"] == 25.5


def test_get_financials_no_info(mock_yf_ticker):
    """Test handling of missing info data."""
    mock_yf_ticker.info = {}
    
    with patch('app.finbot.mcp_finbot.mcp_tools.yf.Ticker', return_value=mock_yf_ticker):
        result = get_financials("AAPL")
        
        # Result should still be valid financials without metrics
        assert isinstance(result, dict)
        assert "balance sheet" in result
        assert "income statement" in result
        assert "cash flow" in result


def test_get_macroeconomic_series_success():
    """Test successful retrieval of macroeconomic data."""
    mock_macro_data = MagicMock()
    mock_macro_data.shape = (2, 1)
    mock_macro_data.index = [MagicMock(), MagicMock()]
    mock_macro_data.index[0].strftime.return_value = "2023-01-01"
    mock_macro_data.index[1].strftime.return_value = "2023-01-02"
    mock_macro_data.values = [[1.5], [2.5]]
    
    with patch('app.finbot.mcp_finbot.mcp_tools.pdr.get_data_fred', return_value=mock_macro_data):
        result = get_macroeconomic_series("GDPC1", "2022", "2023")
        result_json = json.loads(result)
        
        assert "dates" in result_json
        assert "values" in result_json
        assert len(result_json["dates"]) == 2
        assert len(result_json["values"]) == 2
        assert result_json["dates"][0] == "2023-01-01"
        assert result_json["values"][0] == 1.5


def test_get_macroeconomic_series_empty():
    """Test handling of empty macroeconomic data."""
    mock_macro_data = MagicMock()
    mock_macro_data.shape = (0, 0)
    
    with patch('app.finbot.mcp_finbot.mcp_tools.pdr.get_data_fred', return_value=mock_macro_data):
        result = get_macroeconomic_series("GDPC1", "2022", "2023")
        result_json = json.loads(result)
        
        assert "error" in result_json
        assert result_json["error"] == "No data available"


def test_search_news_success():
    """Test successful news search."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"title": "Test Article 1", "url": "https://example.com/1"},
            {"title": "Test Article 2", "url": "https://example.com/2"}
        ]
    }
    
    with patch('app.finbot.mcp_finbot.mcp_tools.requests.post', return_value=mock_response):
        result = search_news("AAPL news")
        result_json = json.loads(result)
        
        assert len(result_json) == 2
        assert result_json[0]["title"] == "Test Article 1"
        assert result_json[1]["url"] == "https://example.com/2"


def test_search_news_exception():
    """Test handling of exceptions during news search."""
    with patch('app.finbot.mcp_finbot.mcp_tools.requests.post', side_effect=Exception("Test error")):
        result = search_news("AAPL news")
        result_json = json.loads(result)
        
        assert "error" in result_json
        assert "Test error" in result_json["error"] 