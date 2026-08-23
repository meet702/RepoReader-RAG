import os
import sys

# Ensure imports work from the root dir
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document

from retrieval.query_rewriter import rewrite_query
from retrieval.pipeline import run_pipeline

def start_chat():
    repo_name = "sampleproject"
    persist_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), 'db', 'chroma_db', repo_name))
    
    if not os.path.exists(persist_directory):
        print(f"Error: No ChromaDB found at {persist_directory}. Run ingest.py first.")
        return

    print("Loading vector database and chunks...")
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
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
        
    print(f"Loaded {len(all_chunks)} chunks successfully.")
    
    llm = ChatOllama(model="qwen2.5-coder:7b")
    chat_history = []
    
    print("\nAsk me questions about the codebase! Type 'quit' to exit.")
    
    while True:
        try:
            question = input("\nYour question: ").strip()
        except EOFError:
            break
            
        if not question:
            continue
            
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        print(f"\n--- Processing ---")
        # 1. Rewrite query
        search_query = rewrite_query(question, chat_history)
        if search_query != question:
            print(f"Rewritten query: {search_query}")
        
        # 2. Run retrieval pipeline
        top_docs = run_pipeline(
            query=search_query,
            persist_directory=persist_directory,
            all_chunks=all_chunks,
            dense_k=15,
            sparse_k=15,
            rerank_top_n=5
        )
        
        print(f"Found {len(top_docs)} relevant chunks for context.")
        for i, doc in enumerate(top_docs, 1):
            file = doc.metadata.get('file', 'unknown')
            c_type = doc.metadata.get('chunk_type', 'unknown')
            print(f"  - {file} [{c_type}]")
            
        # 3. Generate Answer
        combined_input = f"""Based on the following documents, please answer this question: {question}

Documents:
{"\n".join([f"- {doc.page_content}" for doc in top_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""
        
        messages = [
            SystemMessage(content="You are a helpful software engineering assistant that answers questions based on provided documents and conversation history."),
        ] + chat_history + [
            HumanMessage(content=combined_input)
        ]
        
        print("\n--- Answer ---")
        result = llm.invoke(messages)
        answer = result.content
        print(answer)
        
        # 4. Update history
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))

if __name__ == "__main__":
    start_chat()
