from langchain_chroma import Chroma
from langchain_core.documents import Document
from llm.provider import get_embeddings

def get_dense_results(query: str, persist_directory: str, k: int = 15) -> list[Document]:
    embedding_model = get_embeddings()
    vectorstore = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    results = vectorstore.similarity_search(query, k=k)
    return results
