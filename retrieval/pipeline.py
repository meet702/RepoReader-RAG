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
    rerank_top_n: int = 12
) -> list[Document]:
    
    dense_results = get_dense_results(query, persist_directory, k=dense_k)
    sparse_results = get_sparse_results(query, all_chunks, k=sparse_k)
    fused_results = reciprocal_rank_fusion(dense_results, sparse_results, k=60)
    
    # Exclude config files (pom.xml, application.yml, etc.) from reranking —
    # they are rarely the answer to code questions and their keyword content
    # (e.g. <modelVersion>) actively misleads the cross-encoder.
    rerank_candidates = [doc for doc in fused_results if doc.metadata.get("chunk_type") != "config"]
    
    final_results = rerank(query, rerank_candidates, top_n=rerank_top_n)
    
    return final_results
