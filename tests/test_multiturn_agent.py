import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Ingestion')))

from langchain_core.messages import HumanMessage, AIMessage
from api.main import get_repo_state
from agent.tools import build_tools
from agent.graph import build_agent
from retrieval.query_rewriter import rewrite_query

repo_name = "To-Do-list"
state = get_repo_state(repo_name)
tools = build_tools(state["persist_dir"], state["all_chunks"], state["graph"])
agent = build_agent(tools)

history = []

def run_turn(turn_idx, question):
    print(f"\n=================== TURN {turn_idx} ===================")
    print(f"QUESTION: {question}")
    
    rewritten = rewrite_query(question, history)
    print(f"REWRITTEN: {rewritten}")
    
    messages = [HumanMessage(content=rewritten)]
    result = agent.invoke({"messages": messages})
    
    print("\n--- AGENT MESSAGES TRACE ---")
    final_answer = ""
    for msg in result["messages"]:
        print(f"[{type(msg).__name__}]")
        if isinstance(msg, AIMessage) and getattr(msg, 'tool_calls', None):
            print("Tool calls:", msg.tool_calls)
        elif getattr(msg, 'content', None):
            print(msg.content)
        
        if isinstance(msg, AIMessage) and not getattr(msg, 'tool_calls', None):
            final_answer = msg.content
            
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=final_answer))


run_turn(1, "do this project has repository folder?")
run_turn(2, "okay so which all files are present in that particular folder")
