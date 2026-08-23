import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Ingestion')))

from Ingestion.ast_chunker import ASTChunker

file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dummy_python_decorators.py'))
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

chunker = ASTChunker()
chunks = chunker.chunk_file(file_path, repo_path)

print(f"Loaded {len(chunks)} chunks.")
for i, chunk in enumerate(chunks, 1):
    print(f"\n--- Chunk {i} ({chunk.metadata['chunk_type']}) ---")
    print(chunk.page_content)
