import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Ingestion')))

from retrieval.dense_retriever import get_dense_results
from retrieval.sparse_retriever import get_sparse_results
from retrieval.rrf import reciprocal_rank_fusion
from sentence_transformers import CrossEncoder
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

query = "find the models"

# Step 1: Dense + Sparse retrieval
dense_results = get_dense_results(query, persist_dir, k=15)
sparse_results = get_sparse_results(query, all_chunks, k=15)

# Step 2: RRF fusion — full untruncated list
fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60)

print(f"(A) FULL RRF-FUSED LIST — {len(fused)} documents total")
print("="*60)
for i, doc in enumerate(fused, 1):
    f = doc.metadata.get('file', '?')
    t = doc.metadata.get('chunk_type', '?')
    flag = " <--- Task.java" if "Model/Task.java" in f else ""
    print(f"  [{i:2d}] {f} | {t}{flag}")
    print(f"        content: {doc.page_content[:80].strip()}")
print()

# Step 3: Cross-encoder scores for ALL fused candidates
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [[query, doc.page_content] for doc in fused]
scores = cross_encoder.predict(pairs)
doc_scores = sorted(zip(fused, scores), key=lambda x: x[1], reverse=True)

print(f"(B) CROSS-ENCODER SCORES (all {len(fused)} candidates, sorted by score)")
print("="*60)
for i, (doc, score) in enumerate(doc_scores, 1):
    f = doc.metadata.get('file', '?')
    t = doc.metadata.get('chunk_type', '?')
    flag = " <--- Task.java" if "Model/Task.java" in f else ""
    print(f"  [{i:2d}] score={score:8.4f} | {f} | {t}{flag}")
