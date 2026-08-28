import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Ingestion')))

from code_graph.graph_store import load_graph, get_callers, get_callees

graph_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db', 'code_graph', 'To-Do-list.pkl'))
graph = load_graph(graph_path)

print(f"Nodes in graph: {len(graph.nodes)}")
print("Some edges:")
for edge in list(graph.edges)[:10]:
    print(edge)
