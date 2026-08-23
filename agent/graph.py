import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.outputs import ChatResult
from langgraph.prebuilt import create_react_agent

from agent.tools import code_search_tool, github_search_tool, graph_search_tool

class QwenToolChatOllama(ChatOllama):
    def _generate(self, messages, stop, run_manager, **kwargs) -> ChatResult:
        result = super()._generate(messages, stop, run_manager, **kwargs)
        for gen in result.generations:
            msg = gen.message
            if isinstance(msg, AIMessage) and isinstance(msg.content, str):
                content = msg.content.strip()
                # Check for Qwen JSON tool call format
                if content.startswith('{"name":') and '"arguments":' in content:
                    try:
                        tool_data = json.loads(content)
                        if "name" in tool_data and "arguments" in tool_data:
                            # Manually construct tool_calls
                            msg.content = ""
                            msg.tool_calls = [{
                                "name": tool_data["name"],
                                "args": tool_data["arguments"],
                                "id": "call_" + tool_data["name"]
                            }]
                    except Exception:
                        pass
        return result

SYSTEM_PROMPT = """You are a helpful software engineering assistant with access to three tools:
- code_search_tool: searches the ingested repository for code, functions, classes, and documentation.
- github_search_tool: searches GitHub issues, PRs, and commit history.
- graph_search_tool: finds callers, callees, and structural relationships between code elements.

IMPORTANT RULES:
1. For ANY question about this specific repository's code, files, or documentation, you MUST use a tool first, even if you think you know the answer. Only skip tools for purely general programming concepts unrelated to this repository.
2. Always use a tool to answer questions about the repository before responding.
3. If a tool returns "not yet implemented", you MUST say plainly that the capability is not yet available. Do NOT invent, guess, or hallucinate an answer based on general knowledge.
4. Only answer from general knowledge (without calling a tool) if the question is clearly not about this specific repository at all."""

def build_agent():
    """Build and return a LangGraph ReAct agent with the three tools."""
    llm = QwenToolChatOllama(model="qwen2.5-coder:7b")
    tools = [code_search_tool, github_search_tool, graph_search_tool]
    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    return agent
