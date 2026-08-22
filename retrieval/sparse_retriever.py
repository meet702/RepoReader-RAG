from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

def get_sparse_results(query: str, all_chunks: list[Document], k: int = 15) -> list[Document]:
    if not all_chunks:
        return []
    retriever = BM25Retriever.from_documents(all_chunks)
    retriever.k = k
    return retriever.invoke(query)
