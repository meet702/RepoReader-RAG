import os
import sys

# Add Ingestion to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Ingestion')))

from ast_chunker import ASTChunker

def test_ast_chunking():
    # 1. Create a sample Python file
    python_fixture_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'fixtures', 'sample_fixture.py'))
    java_fixture_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'fixtures', 'UserService.java'))
    
    python_code = """
def standalone_function():
    print("I am a standalone function")

class MyTestClass(BaseClass):
    \"\"\"This is a test class.\"\"\"
    
    def __init__(self, name):
        self.name = name
    
    def method_one(self, a, b):
        return a + b
        
    async def method_two(self):
        await asyncio.sleep(1)
        return "done"
"""
    with open(python_fixture_path, "w") as f:
        f.write(python_code.strip())
        
    chunker = ASTChunker()
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # 2. Test Python file
    print("="*60)
    print("Testing Python Chunker")
    print("="*60)
    python_chunks = chunker.chunk_file(python_fixture_path, repo_path)
    for i, chunk in enumerate(python_chunks):
        print(f"\n--- Chunk {i+1} ---")
        print("Metadata:", chunk.metadata)
        print("Content:\n" + chunk.page_content)
        
    # 3. Test Java file
    print("\n" + "="*60)
    print("Testing Java Chunker")
    print("="*60)
    if os.path.exists(java_fixture_path):
        java_chunks = chunker.chunk_file(java_fixture_path, repo_path)
        for i, chunk in enumerate(java_chunks):
            print(f"\n--- Chunk {i+1} ---")
            print("Metadata:", chunk.metadata)
            print("Content:\n" + chunk.page_content)
    else:
        print(f"Could not find Java fixture at {java_fixture_path}")
        
    # Cleanup
    if os.path.exists(python_fixture_path):
        os.remove(python_fixture_path)

if __name__ == "__main__":
    test_ast_chunking()
