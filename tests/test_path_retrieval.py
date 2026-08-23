import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Ingestion')))

from api.main import get_repo_state
from retrieval.pipeline import run_pipeline

state = get_repo_state("To-Do-list")
query = "src/main/java/com/toDo/To_Do/list/Repository"

results = run_pipeline(query, state["persist_dir"], state["all_chunks"], rerank_top_n=12)

print(f"--- CHUNKS RETURNED FOR QUERY: {query} ---")
for doc in results:
    print(f"[{doc.metadata.get('file')}]")
    print(doc.page_content)
    print("---")
