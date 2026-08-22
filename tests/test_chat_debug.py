import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document

from retrieval.query_rewriter import rewrite_query
from retrieval.pipeline import run_pipeline

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

llm = ChatOllama(model="llama3.2")
chat_history = []


def run_turn(question, show_chunk_content=False):
    print(f"{'='*60}")
    print(f"Q: {question}")
    print(f"{'='*60}")

    search_query = rewrite_query(question, chat_history)
    if search_query != question:
        print(f"Rewritten query: {search_query}")
    else:
        print("(No rewrite — empty history)")

    top_docs = run_pipeline(
        query=search_query,
        persist_directory=persist_directory,
        all_chunks=all_chunks,
        dense_k=15,
        sparse_k=15,
        rerank_top_n=5
    )

    print(f"\nTop {len(top_docs)} chunks used as context:")
    for i, doc in enumerate(top_docs, 1):
        file = doc.metadata.get('file', 'unknown')
        section = doc.metadata.get('section', '')
        label = f"{file}" + (f" [section: {section}]" if section else "")
        if show_chunk_content:
            print(f"\n  --- Chunk {i}: {label} ---")
            print(f"  FULL CONTENT:\n{doc.page_content}")
            print()
        else:
            print(f"  {i}. {label}")

    combined_input = f"""Based on the following documents, please answer this question: {question}

Documents:
{"\n".join([f"- {doc.page_content}" for doc in top_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

    messages = [
        SystemMessage(content="You are a helpful software engineering assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    result = llm.invoke(messages)
    answer = result.content

    print(f"\nAnswer: {answer}")

    # Append ORIGINAL question (not rewritten) to history
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))


# --- TURN 1 ---
run_turn("what encoding should the readme file use", show_chunk_content=False)

# --- TURN 2: show full chunk contents ---
run_turn("why does that matter", show_chunk_content=True)

# --- Print chat_history ---
print("\n" + "="*60)
print("Final chat_history:")
print("="*60)
for i, msg in enumerate(chat_history):
    role = "Human" if isinstance(msg, HumanMessage) else "AI"
    print(f"\n[{i+1}] {role}: {msg.content[:300]}{'...' if len(msg.content) > 300 else ''}")
