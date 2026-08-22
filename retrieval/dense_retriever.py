from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

def get_dense_results(query: str, persist_directory: str, k: int = 15) -> list[Document]:
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    results = vectorstore.similarity_search(query, k=k)
    return results
