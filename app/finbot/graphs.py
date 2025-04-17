from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph

from app.finbot.nodes import (
    State,
    annual_report_node,
    financials_chart_node,
    financials_node,
    macroeconomics_chart_node,
    macroeconomics_node,
    make_supervisor_node,
    news_search_node,
    stock_price_chart_node,
    stock_price_node,
    general_comment_node,
)


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
    finbot_builder.add_node("general_comment", general_comment_node)
    finbot_builder.add_edge(START, "supervisor")
    memory_saver = MemorySaver()
    finbot_graph = finbot_builder.compile(checkpointer=memory_saver)

    return finbot_graph
