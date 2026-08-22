from langchain_core.documents import Document

def reciprocal_rank_fusion(dense_results: list[Document], sparse_results: list[Document], k: int = 60) -> list[Document]:
    def get_doc_id(doc: Document) -> str:
        file_name = doc.metadata.get('file', '')
        start_line = str(doc.metadata.get('start_line', ''))
        # Include page_content hash to be extra safe on deduplication
        return f"{file_name}::{start_line}::{hash(doc.page_content)}"
        
    doc_scores = {}
    doc_map = {}
    
    # Process dense results
    for rank, doc in enumerate(dense_results, 1):
        doc_id = get_doc_id(doc)
        if doc_id not in doc_scores:
            doc_scores[doc_id] = 0.0
            doc_map[doc_id] = doc
        doc_scores[doc_id] += 1.0 / (k + rank)
        
    # Process sparse results
    for rank, doc in enumerate(sparse_results, 1):
        doc_id = get_doc_id(doc)
        if doc_id not in doc_scores:
            doc_scores[doc_id] = 0.0
            doc_map[doc_id] = doc
        doc_scores[doc_id] += 1.0 / (k + rank)
        
    # Sort descending by score
    sorted_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
    
    fused_results = [doc_map[doc_id] for doc_id in sorted_doc_ids]
    return fused_results
