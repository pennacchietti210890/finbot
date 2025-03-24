from app.finbot.nodes import stock_price_node, stock_price_chart_node, make_supervisor_node, State, financials_node
from langgraph.graph import StateGraph, START


def create_graph(llm):
    stock_supervisor_node = make_supervisor_node(
        llm, ["stock_price_data", "stock_price_chart", "financial_statements_and_metrics"]
    )  

    finbot_builder = StateGraph(State)
    finbot_builder.add_node("supervisor", stock_supervisor_node)
    finbot_builder.add_node("stock_price_data", stock_price_node)
    finbot_builder.add_node("stock_price_chart", stock_price_chart_node)
    finbot_builder.add_node("financial_statements_and_metrics", financials_node)

    finbot_builder.add_edge(START, "supervisor")
    finbot_graph = finbot_builder.compile()

    return finbot_graph
