import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Ingestion')))

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from code_graph.graph_store import load_graph
from agent.tools import build_tools
from agent.graph import build_agent

repo_name = "To-Do-list"
persist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db', 'chroma_db', repo_name))
graph_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db', 'code_graph', f"{repo_name}.pkl"))

print(f"Loading chunks from {persist_dir}...")
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
collection_data = vectorstore.get()
all_chunks = [
    Document(page_content=content, metadata=meta)
    for content, meta in zip(collection_data['documents'], collection_data['metadatas'])
]
print(f"Loaded {len(all_chunks)} chunks.")

# --- STEP 1: Show updated Task.java chunk content ---
print("\n--- Task.java chunk content (after fix) ---")
for chunk in all_chunks:
    if "Task.java" in chunk.metadata.get("file", "") and chunk.metadata.get("chunk_type") == "class":
        print(f"FILE: {chunk.metadata['file']}")
        print(f"TYPE: {chunk.metadata['chunk_type']}")
        print(f"CONTENT:\n{chunk.page_content}")
        print()

# --- STEP 2: Full agent trace ---
graph = load_graph(graph_path) if os.path.exists(graph_path) else None
tools = build_tools(persist_dir, all_chunks, graph)
agent = build_agent(tools)

question = "in which file all the models are present? also give me the path of that file"
print(f"\nQuestion: {question}")
print("="*60)

result = agent.invoke({"messages": [HumanMessage(content=question)]})

print("\n--- FULL AGENT TRACE ---\n")
for i, msg in enumerate(result["messages"]):
    msg_type = type(msg).__name__
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    tool_calls = getattr(msg, 'tool_calls', None)

    print(f"[{i+1}] {msg_type}")
    if tool_calls:
        print(f"  tool_calls: {tool_calls}")
    if content:
        print(f"  content: {content[:800]}")
    print()
