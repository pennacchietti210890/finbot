from langgraph.prebuilt import create_react_agent
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field
from typing import List
from langchain_core.tools import BaseTool


class ReActAgent:
    """A ReAct agent that can be used to create a MCP agent."""

    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[BaseTool],
        structured_response: BaseModel = None,
    ):
        self.llm = llm
        self.tools = tools
        if structured_response:
            self._agent = create_react_agent(
                self.llm, self.tools, response_format=structured_response
            )
        else:
            self._agent = create_react_agent(self.llm, self.tools)

    @property
    def agent(self):
        return self._agent
