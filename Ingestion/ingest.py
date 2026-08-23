import os
import sys
from collections import Counter

from dotenv import load_dotenv
load_dotenv()

from github_loader import clone_repo
from file_filter import get_parseable_files
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ast_chunker import ASTChunker
from code_graph.graph_builder import build_graph
from code_graph.graph_store import save_graph

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def main(repo_url: str):
    print(f"=== Starting Ingestion for {repo_url} ===")
    
    # 1 & 2. Clone repository
    print("\n--- 1. Cloning Repository ---")
    dest_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cloned_repos'))
    repo_path = clone_repo(repo_url, dest_dir)
    print(f"Repository local path: {repo_path}")
    
    # 3. Get parseable files
    print("\n--- 2. Filtering Files ---")
    files = get_parseable_files(repo_path)
    print(f"Found {len(files)} files to parse.")
    
    if not files:
        print("No parseable files found. Exiting.")
        return
        
    # 4. Chunk files
    print("\n--- 3. Chunking Files ---")
    chunker = ASTChunker()
    all_chunks = []
    
    for i, file_path in enumerate(files, 1):
        try:
            chunks = chunker.chunk_file(file_path, repo_path)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  [!] Failed to chunk {file_path}: {e}")
            continue
            
        if i % 10 == 0 or i == len(files):
            print(f"Chunked {i}/{len(files)} files...")
            
    # 5. Print summary
    print("\n--- 4. Summary ---")
    print(f"Total files processed: {len(files)}")
    print(f"Total chunks created: {len(all_chunks)}")
    
    chunk_types = Counter(chunk.metadata.get("chunk_type", "unknown") for chunk in all_chunks)
    print("Chunk types breakdown:")
    for c_type, count in chunk_types.items():
        print(f"  - {c_type}: {count}")
        
    if not all_chunks:
        print("No chunks created. Exiting.")
        return
        
    # 6. Generate embeddings and store in ChromaDB
    print("\n--- 5. Creating Vector Store ---")
    repo_name = os.path.basename(repo_path)
    persist_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db', 'chroma_db', repo_name))
    
    if os.path.exists(persist_directory):
        print(f"Found existing vector store at {persist_directory}. Deleting to prevent duplication...")
        import shutil
        shutil.rmtree(persist_directory, ignore_errors=True)
    
    print(f"Initializing Ollama embeddings (nomic-embed-text)...")
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    
    batch_size = 20
    print(f"Storing chunks in {persist_directory} in batches of {batch_size}...")
    
    vectorstore = Chroma.from_documents(
        documents=all_chunks[:batch_size],
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"  Embedded chunks 1-{min(batch_size, len(all_chunks))}")
    
    for i in range(batch_size, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        vectorstore.add_documents(batch)
        end = min(i + batch_size, len(all_chunks))
        print(f"  Embedded chunks {i + 1}-{end}")
        
    print("\n--- 6. Verification ---")
    # Actually count from the collection directly to verify
    try:
        count = vectorstore._collection.count()
        print(f"Verification successful: {count} chunks currently stored in ChromaDB for '{repo_name}'.")
    except Exception as e:
        print(f"Could not verify count: {e}")
        
    print("\n--- 7. Building Code Relationship Graph ---")
    graph_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db', 'code_graph'))
    os.makedirs(graph_dir, exist_ok=True)
    graph_path = os.path.join(graph_dir, f"{repo_name}.pkl")
    
    print("Parsing files to build graph...")
    graph = build_graph(repo_path, files)
    print(f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    
    save_graph(graph, graph_path)
    print(f"Graph saved to {graph_path}")
        
    print("\n=== Ingestion Complete ===")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <repo_url>")
        sys.exit(1)
        
    main(sys.argv[1])