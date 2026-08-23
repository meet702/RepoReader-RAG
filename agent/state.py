from typing import TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    question: str
    chat_history: list[BaseMessage]
    context: str
    answer: str
