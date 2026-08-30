import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'retrieval')))

from retrieval.dense_retriever import get_dense_results
from retrieval.sparse_retriever import get_sparse_results
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.reranker import rerank
from langchain_chroma import Chroma
from llm.provider import get_embeddings
from langchain_core.documents import Document

def test_retrieval():
    repo_name = "sampleproject"
    persist_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db', 'chroma_db', repo_name))
    
    if not os.path.exists(persist_directory):
        print(f"Error: No ChromaDB found at {persist_directory}. Please run ingest.py on sampleproject first.")
        return
        
    print(f"Loading chunks from ChromaDB at {persist_directory}...")
    embedding_model = get_embeddings()
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    # Extract all chunks for BM25
    collection_data = vectorstore.get()
    all_chunks = []
    for doc_content, metadata in zip(collection_data['documents'], collection_data['metadatas']):
        all_chunks.append(Document(page_content=doc_content, metadata=metadata))
        
    print(f"Successfully loaded {len(all_chunks)} chunks for sparse retrieval.\n")
    
    query = "what encoding should the readme file use"
    print(f'=== Query: "{query}" ===\n')
    
    # 1. Dense Retrieval
    dense_results = get_dense_results(query, persist_directory, k=15)
    print("--- Top 5 Dense Results ---")
    for i, doc in enumerate(dense_results[:5], 1):
        file = doc.metadata.get('file', 'unknown')
        c_type = doc.metadata.get('chunk_type', 'unknown')
        name = doc.metadata.get('function') or doc.metadata.get('class') or doc.metadata.get('method') or ''
        tag = f"{file} [{c_type} {name}]".strip()
        print(f"{i}. {tag}")
        
    # 2. Sparse Retrieval
    sparse_results = get_sparse_results(query, all_chunks, k=15)
    print("\n--- Top 5 Sparse Results ---")
    for i, doc in enumerate(sparse_results[:5], 1):
        file = doc.metadata.get('file', 'unknown')
        c_type = doc.metadata.get('chunk_type', 'unknown')
        name = doc.metadata.get('function') or doc.metadata.get('class') or doc.metadata.get('method') or ''
        tag = f"{file} [{c_type} {name}]".strip()
        print(f"{i}. {tag}")
        
    # 3. RRF Fusion
    fused_results = reciprocal_rank_fusion(dense_results, sparse_results, k=60)
    print("\n--- Top 5 Fused Results (RRF) ---")
    for i, doc in enumerate(fused_results[:5], 1):
        file = doc.metadata.get('file', 'unknown')
        c_type = doc.metadata.get('chunk_type', 'unknown')
        name = doc.metadata.get('function') or doc.metadata.get('class') or doc.metadata.get('method') or ''
        tag = f"{file} [{c_type} {name}]".strip()
        print(f"{i}. {tag}")
        
    # 4. Reranking
    reranked_results = rerank(query, fused_results, top_n=5)
    print("\n--- Top 5 Reranked Results ---")
    for i, doc in enumerate(reranked_results[:5], 1):
        file = doc.metadata.get('file', 'unknown')
        c_type = doc.metadata.get('chunk_type', 'unknown')
        name = doc.metadata.get('function') or doc.metadata.get('class') or doc.metadata.get('method') or ''
        tag = f"{file} [{c_type} {name}]".strip()
        preview = doc.page_content[:150].replace('\n', ' ')
        if len(doc.page_content) > 150:
            preview += "..."
        print(f"{i}. {tag}\n   Content: {preview}")

if __name__ == "__main__":
    test_retrieval()
