from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated, Literal
from app.llm.llm_service import LLMService
from app.finbot.agents import create_stock_price_agent
import os
from dotenv import load_dotenv

env_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"
)
load_dotenv(env_path)

agents_llm = LLMService(
    llm_provider="openai", model_name="gpt-4o", api_key=os.getenv("OPENAI_API_KEY")
).client


class State(TypedDict):
    next: str
    messages: Annotated[list, add_messages]
    stock_data: str


def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router."""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})

    return supervisor_node


def stock_price_node(state: State) -> Command[Literal["supervisor"]]:
    stock_price_agent = create_stock_price_agent(agents_llm)
    result = stock_price_agent.invoke(state)

    # ✅ Accessing Intermediate response tool call observation
    stock_data = result["messages"][-2].content  # Already a JSON dict
    last_message = result["messages"][-1].content  # Already a JSON dict
    return Command(
        update={
            "messages": [HumanMessage(content=last_message, name="stock_price")],
            "stock_data": stock_data,  # ✅ Directly store tool output
        },
        goto="supervisor",
    )


def chart_node(state: State) -> Command[Literal["supervisor"]]:
    stock_data = json.loads(state.get("stock_data", None))
    if not stock_data or "error" in stock_data:
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content="No valid stock data available to plot.", name="chart"
                    )
                ]
            },
            goto="supervisor",
        )

    python_code = textwrap.dedent(
        f"""\
        import plotly.graph_objects as go
        from IPython.display import display, HTML

        dates = {stock_data["dates"]}
        prices = {stock_data["prices"]}

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name='Stock Price'))

        fig.update_layout(
            title="Stock Price Time Series",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark"
        )

        html_output = fig.to_html()
        display(HTML(html_output))  # Display in Jupyter
    """
    )

    # ✅ Print debug output before execution
    # print("Generated Python Code:\n", python_code)

    # ✅ Execute the corrected Python code
    result = python_repl_tool(python_code)

    return Command(
        update={
            "messages": [
                HumanMessage(content="✅ Chart generated successfully.", name="chart")
            ],
            "stock_data": stock_data,  # ✅ Preserve stock_data
        },
        goto="supervisor",
    )
