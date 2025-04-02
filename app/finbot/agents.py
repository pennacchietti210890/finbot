from langgraph.prebuilt import create_react_agent
from app.finbot.tools import (
    get_historical_prices,
    get_financials,
    get_macroeconomic_series,
)
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field
from typing import List
from app.finbot.tools import search_tool


def create_stock_price_agent(llm: BaseChatModel):
    stock_price_agent = create_react_agent(
        llm,
        tools=[get_historical_prices],
    )
    return stock_price_agent


def create_financials_agent(llm: BaseChatModel):
    financials_agent = create_react_agent(
        llm,
        tools=[get_financials],
    )
    return financials_agent


def create_macroeconomics_agent(llm: BaseChatModel):
    macroeconomics_agent = create_react_agent(
        llm,
        tools=[get_macroeconomic_series],
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


def create_financials_chart_agent(llm: BaseChatModel):
    financials_chart_agent = create_react_agent(
        llm,
        tools=[get_financials],
        response_format=FinancialsChartStruct,
    )
    return financials_chart_agent

def create_news_search_agent(llm: BaseChatModel):
    news_search_agent = create_react_agent(
        llm,
        tools=[search_tool],
    )
    return news_search_agent