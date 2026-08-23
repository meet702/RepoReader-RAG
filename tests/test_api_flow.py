import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Ingestion')))

from langchain_core.messages import HumanMessage
from api.main import get_repo_state
from agent.tools import build_tools
from agent.graph import build_agent
from retrieval.query_rewriter import rewrite_query

repo_name = "sampleproject"
state = get_repo_state(repo_name)
tools = build_tools(state["persist_dir"], state["all_chunks"], state["graph"])
agent = build_agent(tools)

question = "do this project has repository folder?"
rewritten = rewrite_query(question, [])
print(f"Rewritten query: {rewritten}")

# Run agent
messages = [HumanMessage(content=rewritten)]

try:
    result = agent.invoke({"messages": messages})
    print("\n--- FINAL AGENT MESSAGES ---")
    for msg in result["messages"]:
        print(f"[{type(msg).__name__}]")
        print(repr(msg.content))
        if getattr(msg, 'tool_calls', None):
            print("Tool calls:", msg.tool_calls)
except Exception as e:
    print(f"Error: {e}")
