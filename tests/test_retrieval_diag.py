import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Ingestion')))

from retrieval.pipeline import run_pipeline
from retrieval.dense_retriever import get_dense_results
from retrieval.sparse_retriever import get_sparse_results
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

repo_name = "To-Do-list"
persist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db', 'chroma_db', repo_name))

embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
collection_data = vectorstore.get()
all_chunks = [
    Document(page_content=content, metadata=meta)
    for content, meta in zip(collection_data['documents'], collection_data['metadatas'])
]
print(f"Loaded {len(all_chunks)} chunks.\n")


def print_results(label, results):
    print(f"\n{'='*60}")
    print(f"QUERY: {label}")
    print('='*60)
    if not results:
        print("  No results.")
        return
    for j, doc in enumerate(results, 1):
        print(f"  [{j}] FILE: {doc.metadata.get('file', '?')} | TYPE: {doc.metadata.get('chunk_type', '?')}")
        print(f"       CONTENT: {doc.page_content[:200]}")
        print()


# --- CHECK 1: Full original question through run_pipeline ---
full_question = "in which file all the models are present? also give me the path of that file"
results = run_pipeline(full_question, persist_dir, all_chunks)
print_results(f"run_pipeline() with FULL QUESTION: '{full_question}'", results)


# --- CHECK 2a: Dense-only for "find the models" ---
dense_results = get_dense_results("find the models", persist_dir, k=5)
print_results("DENSE ONLY: 'find the models'", dense_results)

# --- CHECK 2b: Sparse (BM25)-only for "find the models" ---
sparse_results = get_sparse_results("find the models", all_chunks, k=5)
print_results("SPARSE (BM25) ONLY: 'find the models'", sparse_results)
