"""
Non-interactive test: runs three questions through the LangGraph agent
one at a time, fresh chat_history for each.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from agent.tools import init_tools
from agent.graph import build_agent

def setup():
    repo_name = "sampleproject"
    persist_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db', 'chroma_db', repo_name))

    print("Loading vector database and chunks...")
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    collection_data = vectorstore.get()
    all_chunks = []
    for doc_content, metadata in zip(collection_data['documents'], collection_data['metadatas']):
        all_chunks.append(Document(page_content=doc_content, metadata=metadata))

    print(f"Loaded {len(all_chunks)} chunks.\n")
    graph_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db', 'code_graph', f"{repo_name}.pkl"))
    init_tools(persist_directory, all_chunks, graph_path)
    return build_agent()


def run_question(agent, question):
    print("=" * 60)
    print(f"Q: {question}")
    print("=" * 60)

    messages = [HumanMessage(content=question)]
    final_answer = ""

    for step in agent.stream({"messages": messages}, stream_mode="values"):
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
            final_answer = last_msg.content
            print(f"\n[Final Answer]\n{final_answer}")

    print()


if __name__ == "__main__":
    agent = setup()

    questions = [
        "what encoding should the readme file use",
        "what calls the lint function",
        "explain what a virtual environment is",
        "show me the tests function in noxfile.py"
    ]

    for q in questions:
        run_question(agent, q)
