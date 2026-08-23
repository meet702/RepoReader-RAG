import os
import sys
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from typing import Dict, List, Any

# Ensure we can import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Ingestion.ingest import main as run_ingestion
from agent.tools import build_tools
from agent.graph import build_agent
from code_graph.graph_store import load_graph
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from retrieval.query_rewriter import rewrite_query

app = FastAPI(title="AI Software Engineering Agent API")

class IngestRequest(BaseModel):
    repo_url: str

class ChatRequest(BaseModel):
    repo_name: str
    session_id: str
    question: str

# In-memory storage
_repo_cache: Dict[str, dict] = {}
sessions: Dict[str, list] = {}

def get_repo_state(repo_name: str) -> dict:
    if repo_name in _repo_cache:
        return _repo_cache[repo_name]
        
    persist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db', 'chroma_db', repo_name))
    graph_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db', 'code_graph', f"{repo_name}.pkl"))
    
    if not os.path.exists(persist_dir):
        raise HTTPException(status_code=404, detail=f"Repository {repo_name} not found. Please ingest first.")
        
    # Load Vectorstore and Chunks
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model
    )
    
    collection_data = vectorstore.get()
    all_chunks = []
    from langchain_core.documents import Document
    if collection_data and 'documents' in collection_data:
        for content, meta in zip(collection_data['documents'], collection_data['metadatas']):
            all_chunks.append(Document(page_content=content, metadata=meta))
            
    # Load Graph
    graph = None
    if os.path.exists(graph_path):
        graph = load_graph(graph_path)
        
    _repo_cache[repo_name] = {
        "persist_dir": persist_dir,
        "all_chunks": all_chunks,
        "graph": graph
    }
    
    return _repo_cache[repo_name]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/repos")
def list_repos():
    db_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db', 'chroma_db'))
    if not os.path.exists(db_dir):
        return {"repos": []}
        
    repos = [d for d in os.listdir(db_dir) if os.path.isdir(os.path.join(db_dir, d))]
    return {"repos": repos}


@app.post("/ingest")
def ingest_repo(req: IngestRequest):
    try:
        stats = run_ingestion(req.repo_url)
        # Clear cache for this repo if it was previously ingested
        repo_name = stats["repo_name"]
        if repo_name in _repo_cache:
            del _repo_cache[repo_name]
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        # Load state from cache (O(1) after first load)
        state = get_repo_state(req.repo_name)
        
        # Build fresh tools closures for this request
        tools = build_tools(state["persist_dir"], state["all_chunks"], state["graph"])
        
        # Build fresh agent
        agent = build_agent(tools)
        
        # Manage history
        if req.session_id not in sessions:
            sessions[req.session_id] = []
            
        history = sessions[req.session_id]
        
        # Rewrite query
        rewritten = rewrite_query(req.question, history)
        
        # Run agent
        messages = [HumanMessage(content=rewritten)]
        result = agent.invoke({"messages": messages})
        
        # Extract answer and tool used
        final_answer = result["messages"][-1].content
        
        tool_used = "None"
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and getattr(msg, 'tool_calls', None):
                tool_used = msg.tool_calls[0]["name"]
                break
                
        # Update history with ORIGINAL question
        history.append(HumanMessage(content=req.question))
        history.append(AIMessage(content=final_answer))
        
        return {
            "answer": final_answer,
            "tool_used": tool_used,
            "rewritten_query": rewritten
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
