from langchain_core.documents import Document
from retrieval.dense_retriever import get_dense_results
from retrieval.sparse_retriever import get_sparse_results
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.reranker import rerank

def run_pipeline(
    query: str, 
    persist_directory: str, 
    all_chunks: list[Document], 
    dense_k: int = 15, 
    sparse_k: int = 15, 
    rerank_top_n: int = 5
) -> list[Document]:
    
    dense_results = get_dense_results(query, persist_directory, k=dense_k)
    sparse_results = get_sparse_results(query, all_chunks, k=sparse_k)
    fused_results = reciprocal_rank_fusion(dense_results, sparse_results, k=60)
    final_results = rerank(query, fused_results, top_n=rerank_top_n)
    
    return final_results
