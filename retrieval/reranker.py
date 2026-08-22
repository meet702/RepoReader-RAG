from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

# Load the CrossEncoder model once at the module level.
# This runs locally and doesn't require an API key.
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, documents: list[Document], top_n: int = 5) -> list[Document]:
    if not documents:
        return []

    # Create pairs of (query, document_text)
    pairs = [[query, doc.page_content] for doc in documents]
    
    # Score each document against the query using the cross-encoder
    scores = cross_encoder.predict(pairs)
    
    # Attach scores to documents for sorting
    doc_scores = list(zip(documents, scores))
    
    # Sort documents by score descending
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top_n documents
    reranked_docs = [doc for doc, score in doc_scores[:top_n]]
    
    return reranked_docs
