from langgraph.prebuilt import create_react_agent
from app.finbot.tools import get_historical_prices, get_financials
from langchain_core.language_models.chat_models import BaseChatModel


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
