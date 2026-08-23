import pickle
import networkx as nx

def save_graph(graph: nx.MultiDiGraph, path: str):
    with open(path, 'wb') as f:
        pickle.dump(graph, f)

def load_graph(path: str) -> nx.MultiDiGraph:
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_callers(graph: nx.MultiDiGraph, function_name: str) -> list[str]:
    """Find all nodes that call the given function_name (substring match on target node)."""
    results = []
    
    for u, v, data in graph.edges(data=True):
        if data.get("type") == "calls" and function_name in str(v):
            file = data.get("source_file", "unknown")
            line = data.get("line_number", "?")
            results.append(f"{u} at line {line} calls {v}")
            
    # Deduplicate while preserving order
    unique_results = []
    seen = set()
    for r in results:
        if r not in seen:
            unique_results.append(r)
            seen.add(r)
            
    return unique_results

def get_callees(graph: nx.MultiDiGraph, function_name: str) -> list[str]:
    """Find all nodes called by the given function_name (substring match on source node)."""
    results = []
    
    for u, v, data in graph.edges(data=True):
        if data.get("type") == "calls" and function_name in str(u):
            file = data.get("source_file", "unknown")
            line = data.get("line_number", "?")
            results.append(f"{u} at line {line} calls {v}")
            
    unique_results = []
    seen = set()
    for r in results:
        if r not in seen:
            unique_results.append(r)
            seen.add(r)
            
    return unique_results
