from langgraph.prebuilt import create_react_agent
from app.finbot.tools import (
    get_historical_prices,
    get_financials,
    get_macroeconomic_series,
    get_annual_report,
)
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field
from typing import List
from app.finbot.tools import search_tool


def create_mcp_stock_price_agent(llm: BaseChatModel, tool_map: dict):
    stock_price_agent = create_react_agent(
        llm, tools=[tool_map["get_historical_prices"]]
    )
    return stock_price_agent


def create_mcp_financials_agent(llm: BaseChatModel, tool_map: dict):
    financials_agent = create_react_agent(llm, tools=[tool_map["get_financials"]])
    return financials_agent


def create_mcp_macroeconomics_agent(llm: BaseChatModel, tool_map: dict):
    macroeconomics_agent = create_react_agent(
        llm, tools=[tool_map["get_macroeconomic_series"]]
    )
    return macroeconomics_agent


class FinancialsChartStruct(BaseModel):
    """Dates and Values of the financials time series to chart."""

    dates: List[str] = Field(
        description="The dates of the financials time series to chart."
    )
    values: List[float] = Field(
        description="The values of the financials time series to chart. Just the values, no other text."
    )


def create_mcp_financials_chart_agent(llm: BaseChatModel, tool_map: dict):
    financials_chart_agent = create_react_agent(
        llm,
        tools=[tool_map["get_financials"]],
        response_format=FinancialsChartStruct,
    )
    return financials_chart_agent


def create_mcp_news_search_agent(llm: BaseChatModel, tool_map: dict):
    news_search_agent = create_react_agent(
        llm,
        tools=[tool_map["search_news"]],
    )
    return news_search_agent


def create_mcp_annual_report_agent(llm: BaseChatModel, tool_map: dict):
    annual_report_agent = create_react_agent(
        llm,
        tools=[tool_map["get_annual_report"]],
    )
    return annual_report_agent
