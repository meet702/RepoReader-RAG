"""
Targeted test: Q4 (unambiguous code search) + corrected Q2 (lint caller) with system prompt fix.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

from agent.tools import init_tools
from agent.graph import build_agent


def setup():
    repo_name = "sampleproject"
    persist_directory = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'db', 'chroma_db', repo_name)
    )
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    collection_data = vectorstore.get()
    all_chunks = [
        Document(page_content=content, metadata=meta)
        for content, meta in zip(collection_data['documents'], collection_data['metadatas'])
    ]
    print(f"Loaded {len(all_chunks)} chunks.\n")
    init_tools(persist_directory, all_chunks)
    return build_agent()


def run_question(agent, question):
    print("=" * 60)
    print(f"Q: {question}")
    print("=" * 60)

    for step in agent.stream({"messages": [HumanMessage(content=question)]}, stream_mode="values"):
        last_msg = step["messages"][-1]
        msg_type = last_msg.__class__.__name__

        if msg_type == "AIMessage" and hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                print(f"\n[Agent calling tool: {tc['name']}]")
                print(f"  Query: {tc['args'].get('query', '')}")

        elif msg_type == "ToolMessage":
            print(f"\n[Tool result from: {last_msg.name}]")
            preview = last_msg.content[:300].replace('\n', ' ')
            if len(last_msg.content) > 300:
                preview += "..."
            print(f"  Preview: {preview}")

        elif msg_type == "AIMessage" and last_msg.content and not (hasattr(last_msg, 'tool_calls') and last_msg.tool_calls):
            print(f"\n[Final Answer]\n{last_msg.content}")

    print()


if __name__ == "__main__":
    agent = setup()

    print("--- Q1: what encoding should the readme file use ---")
    run_question(agent, "what encoding should the readme file use")
