import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Ingestion')))

from github_loader import clone_repo
from file_filter import get_parseable_files
from ast_chunker import ASTChunker

def test_ingestion_pipeline():
    repo_url = "https://github.com/pypa/sampleproject"
    dest_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cloned_repos'))
    
    print(f"--- 1. Cloning Repository ---")
    repo_path = clone_repo(repo_url, dest_dir)
    print(f"Repo path: {repo_path}")
    
    print(f"\n--- 2. Filtering Files ---")
    files = get_parseable_files(repo_path)
    print(f"Found {len(files)} parseable files.")
    
    print(f"\n--- 3. Chunking Files ---")
    chunker = ASTChunker()
    all_chunks = []
    
    for file in files:
        chunks = chunker.chunk_file(file, repo_path)
        all_chunks.extend(chunks)
        
    print(f"Total chunks created: {len(all_chunks)}")
    
    print(f"\n--- 4. Sample Chunks ---")
    for i, chunk in enumerate(all_chunks[:5]):
        print(f"\n[Chunk {i+1}]")
        print(f"Metadata: {chunk.metadata}")
        content_preview = chunk.page_content[:200].replace('\n', ' ')
        if len(chunk.page_content) > 200:
            content_preview += "..."
        print(f"Content: {content_preview}")

if __name__ == "__main__":
    test_ingestion_pipeline()
