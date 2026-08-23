import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import langchain_core

from agent.tools import code_search_tool, init_tools
from agent.graph import build_agent
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

# Setup basics
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
init_tools(persist_directory, all_chunks)


print("=" * 60)
print("TEST 1: RAW bind_tools DIRECTORY TEST (no LangGraph)")
print("=" * 60)
llm = ChatOllama(model="qwen2.5-coder:7b")
llm_with_tools = llm.bind_tools([code_search_tool])

msg = HumanMessage(content="show me the tests function in noxfile.py")
res = llm_with_tools.invoke([msg])

print("raw .content:")
print(repr(res.content))
print("raw .tool_calls:")
print(repr(res.tool_calls))


print("\n" + "=" * 60)
print("TEST 2: LANGGRAPH FULL TRACE FOR Q1")
print("=" * 60)

langchain_core.globals.set_debug(True)

agent = build_agent()
q1 = "what encoding should the readme file use"

print(f"Running Q1: {q1}")
try:
    for step in agent.stream({"messages": [HumanMessage(content=q1)]}, stream_mode="values"):
        # Not doing custom printing here, relying on set_debug(True) output
        pass
except Exception as e:
    print("Error during execution:", e)

langchain_core.globals.set_debug(False)
