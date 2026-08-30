import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Ingestion')))

from retrieval.pipeline import run_pipeline
from langchain_chroma import Chroma
from llm.provider import get_embeddings
from langchain_core.documents import Document

repo_name = "To-Do-list"
persist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db', 'chroma_db', repo_name))

embedding_model = get_embeddings()
vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
collection_data = vectorstore.get()
all_chunks = [
    Document(page_content=content, metadata=meta)
    for content, meta in zip(collection_data['documents'], collection_data['metadatas'])
]
print(f"Loaded {len(all_chunks)} chunks.\n")

for query in ["find the models", "@Entity", "Task"]:
    print(f"\n{'='*60}")
    print(f"QUERY: '{query}'")
    print('='*60)
    results = run_pipeline(query, persist_dir, all_chunks)
    if not results:
        print("No results.")
    for j, doc in enumerate(results, 1):
        print(f"  [{j}] FILE: {doc.metadata.get('file', '?')} | TYPE: {doc.metadata.get('chunk_type', '?')}")
        print(f"       CONTENT: {doc.page_content[:300]}")
        print()
