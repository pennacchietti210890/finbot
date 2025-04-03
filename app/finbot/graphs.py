from app.finbot.nodes import (
    stock_price_node,
    stock_price_chart_node,
    make_supervisor_node,
    State,
    financials_node,
    financials_chart_node,
    macroeconomics_chart_node,
    macroeconomics_node,
    news_search_node,
    annual_report_node,
)
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

def create_graph(llm):
    stock_supervisor_node = make_supervisor_node(
        llm,
        [
            "stock_price_data",
            "stock_price_chart",
            "financial_statements_and_metrics",
            "financials_chart",
            "macroeconomics",
            "macroeconomics_chart",
            "news_search",
            "annual_report",
        ],
    )

    finbot_builder = StateGraph(State)
    finbot_builder.add_node("supervisor", stock_supervisor_node)
    finbot_builder.add_node("stock_price_data", stock_price_node)
    finbot_builder.add_node("stock_price_chart", stock_price_chart_node)
    finbot_builder.add_node("financial_statements_and_metrics", financials_node)
    finbot_builder.add_node("financials_chart", financials_chart_node)
    finbot_builder.add_node("macroeconomics", macroeconomics_node)
    finbot_builder.add_node("macroeconomics_chart", macroeconomics_chart_node)
    finbot_builder.add_node("news_search", news_search_node)
    finbot_builder.add_node("annual_report", annual_report_node)
    finbot_builder.add_edge(START, "supervisor")
    memory_saver = MemorySaver()
    finbot_graph = finbot_builder.compile(checkpointer=memory_saver)

    return finbot_graph
