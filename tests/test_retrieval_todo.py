import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Ingestion')))

from retrieval.pipeline import run_pipeline
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

repo_name = "To-Do-list"
persist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db', 'chroma_db', repo_name))

print(f"Loading chunks from {persist_dir}...")
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
collection_data = vectorstore.get()
all_chunks = [
    Document(page_content=content, metadata=meta)
    for content, meta in zip(collection_data['documents'], collection_data['metadatas'])
]
print(f"Loaded {len(all_chunks)} chunks.\n")
print("All chunks:")
for i, chunk in enumerate(all_chunks):
    print(f"[{i+1}] file={chunk.metadata.get('file', '?')} type={chunk.metadata.get('chunk_type', '?')}")
    print(f"     content_start: {chunk.page_content[:80].strip()}\n")

for query in ["model class", "entity class", "@Entity"]:
    print(f"\n{'='*60}")
    print(f"QUERY: '{query}'")
    print('='*60)
    results = run_pipeline(query, persist_dir, all_chunks)
    if not results:
        print("No results returned.")
    for doc in results:
        print(f"  FILE: {doc.metadata.get('file', '?')}")
        print(f"  TYPE: {doc.metadata.get('chunk_type', '?')}")
        print(f"  CONTENT:\n{doc.page_content[:300]}")
        print()
