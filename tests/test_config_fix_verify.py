import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Ingestion')))

from retrieval.dense_retriever import get_dense_results
from retrieval.sparse_retriever import get_sparse_results
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.pipeline import run_pipeline
from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

def load_chunks(repo_name):
    persist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db', 'chroma_db', repo_name))
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    collection_data = vectorstore.get()
    all_chunks = [
        Document(page_content=content, metadata=meta)
        for content, meta in zip(collection_data['documents'], collection_data['metadatas'])
    ]
    return persist_dir, all_chunks

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def run_rerank_test(label, query, persist_dir, all_chunks, flag_file_substr):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"QUERY: '{query}'")
    print('='*60)

    dense_results = get_dense_results(query, persist_dir, k=15)
    sparse_results = get_sparse_results(query, all_chunks, k=15)
    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60)

    # Show which config chunks are excluded
    excluded = [d for d in fused if d.metadata.get("chunk_type") == "config"]
    candidates = [d for d in fused if d.metadata.get("chunk_type") != "config"]
    print(f"  Fused total: {len(fused)} | Excluded (config): {len(excluded)} | Reranker input: {len(candidates)}")
    if excluded:
        for d in excluded:
            print(f"    EXCLUDED: {d.metadata.get('file')} [{d.metadata.get('chunk_type')}]")

    pairs = [[query, doc.page_content] for doc in candidates]
    scores = cross_encoder.predict(pairs)
    doc_scores = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    print(f"\n  Cross-encoder top results (all {len(candidates)} candidates):")
    for i, (doc, score) in enumerate(doc_scores[:10], 1):
        f = doc.metadata.get('file', '?')
        t = doc.metadata.get('chunk_type', '?')
        flag = f"  <--- {flag_file_substr}" if flag_file_substr in f else ""
        print(f"  [{i:2d}] score={score:8.4f} | {f} | {t}{flag}")

    print(f"\n  Final top-5 returned by run_pipeline():")
    final = run_pipeline(query, persist_dir, all_chunks)
    for i, doc in enumerate(final, 1):
        f = doc.metadata.get('file', '?')
        t = doc.metadata.get('chunk_type', '?')
        flag = f"  <--- {flag_file_substr}" if flag_file_substr in f else ""
        print(f"  [{i}] {f} | {t}{flag}")
        print(f"       content: {doc.page_content[:120].strip()}")


# --- TEST 1: To-Do-list "find the models" ---
persist_dir_todo, chunks_todo = load_chunks("To-Do-list")
run_rerank_test(
    label="To-Do-list: main test",
    query="find the models",
    persist_dir=persist_dir_todo,
    all_chunks=chunks_todo,
    flag_file_substr="Model/Task.java"
)

# --- TEST 2: sampleproject README regression check ---
persist_dir_sp, chunks_sp = load_chunks("sampleproject")
run_rerank_test(
    label="sampleproject: regression check",
    query="what encoding should the readme file use",
    persist_dir=persist_dir_sp,
    all_chunks=chunks_sp,
    flag_file_substr="README"
)
