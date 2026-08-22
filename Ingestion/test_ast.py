from ast_chunker import ASTChunker

file_path = "UserService.java"
repo_path = "."

chunker = ASTChunker()
chunks = chunker.chunk_file(file_path=file_path, repo_path=repo_path)

print("Total chunks created:", len(chunks))

for i, chunk in enumerate(chunks, 1):
    print(f"\n--- Chunk {i} ---")
    print(f"Metadata: {chunk.metadata}")
    print(f"Content:\n{chunk.page_content}")