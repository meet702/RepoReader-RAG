import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.tools import tool

# These are set once at startup via init_tools() before the agent runs.
_persist_directory: str = ""
_all_chunks: list = []

def init_tools(persist_directory: str, all_chunks: list):
    """Call this once at startup to configure the tools with runtime state."""
    global _persist_directory, _all_chunks
    _persist_directory = persist_directory
    _all_chunks = all_chunks


@tool
def code_search_tool(query: str) -> str:
    """Search the ingested code repository for chunks relevant to the query.
    Use this tool for questions about code, functions, classes, methods,
    configuration, or documentation found in the repository."""
    from retrieval.pipeline import run_pipeline
    
    if not _persist_directory or not _all_chunks:
        return "Error: tools not initialized. Call init_tools() first."
    
    docs = run_pipeline(
        query=query,
        persist_directory=_persist_directory,
        all_chunks=_all_chunks,
        dense_k=15,
        sparse_k=15,
        rerank_top_n=5
    )
    
    if not docs:
        return "No relevant code chunks found for that query."
    
    parts = []
    for doc in docs:
        file = doc.metadata.get('file', 'unknown')
        section = doc.metadata.get('section', '')
        label = f"[{file}]" + (f" (section: {section})" if section else "")
        parts.append(f"{label}\n{doc.page_content}")
        
    return "\n\n---\n\n".join(parts)


@tool
def github_search_tool(query: str) -> str:
    """Search GitHub issues, pull requests, and commit history for information
    related to the query. Use this for questions about why a change was made,
    bug reports, feature discussions, or contributor history."""
    return "GitHub search is not yet implemented."


@tool
def graph_search_tool(query: str) -> str:
    """Search the code relationship graph to find callers, callees, dependencies,
    and structural relationships between code elements. Use this for questions
    like 'what calls X', 'what does Y depend on', or 'what imports Z'."""
    return "Code relationship graph is not yet implemented."
