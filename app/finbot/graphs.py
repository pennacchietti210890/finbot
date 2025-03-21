from app.finbot.nodes import stock_price_node, chart_node, make_supervisor_node, State
from langgraph.graph import StateGraph, START

def create_graph(llm):
    stock_supervisor_node = make_supervisor_node(llm, ["stock_price"])  # "chart"])

    finbot_builder = StateGraph(State)
    finbot_builder.add_node("supervisor", stock_supervisor_node)
    finbot_builder.add_node("stock_price", stock_price_node)
    # finbot_builder.add_node("chart", chart_node)

    finbot_builder.add_edge(START, "supervisor")
    finbot_graph = finbot_builder.compile()

    return finbot_graph
