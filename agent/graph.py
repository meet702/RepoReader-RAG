import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.outputs import ChatResult
from langgraph.prebuilt import create_react_agent

class QwenToolChatOllama(ChatOllama):
    def _generate(self, messages, stop, run_manager, **kwargs) -> ChatResult:
        import re
        result = super()._generate(messages, stop, run_manager, **kwargs)
        for gen in result.generations:
            msg = gen.message
            if isinstance(msg, AIMessage) and isinstance(msg.content, str):
                content = msg.content.strip()
                # Use regex to find a tool call JSON block anywhere in the text
                match = re.search(r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}', content, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    try:
                        tool_data = json.loads(json_str)
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
4. Only answer from general knowledge (without calling a tool) if the question is clearly not about this specific repository at all.
5. If your first search does not return clearly relevant results, try again with a more specific query — for example, a likely class name, annotation (like '@Entity'), or a specific keyword. Only respond that information is unavailable after at least one retry with a different, more targeted query."""

def build_agent(tools):
    """Build and return a LangGraph ReAct agent with the provided tools."""
    llm = QwenToolChatOllama(model="qwen2.5-coder:7b")
    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    return agent
