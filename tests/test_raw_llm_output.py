import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from agent.graph import SYSTEM_PROMPT

@tool
def code_search_tool(query: str) -> str:
    """Searches the repository for code."""
    return "Dummy result"

@tool
def github_search_tool(query: str) -> str:
    """Searches GitHub."""
    return "Dummy result"

@tool
def graph_search_tool(entity: str) -> str:
    """Searches the code graph."""
    return "Dummy result"

llm = ChatOllama(model="qwen2.5-coder:7b").bind_tools([code_search_tool, github_search_tool, graph_search_tool])

messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content="do this project has repository folder?")
]

response = llm.invoke(messages)
print("--- RAW LLM CONTENT REPR ---")
print(repr(response.content))
