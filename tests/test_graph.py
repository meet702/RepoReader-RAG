import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from code_graph.graph_store import load_graph, get_callers

repo_name = "sampleproject"
graph_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db', 'code_graph', f"{repo_name}.pkl"))

print(f"Loading graph from {graph_path}...")
graph = load_graph(graph_path)

print(f"Graph nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")

target_function = "build_and_check_dists"
print(f"\nSearching for callers of '{target_function}'...")
callers = get_callers(graph, target_function)

if callers:
    print("\nCallers found:")
    for caller in callers:
        print(f"  - {caller}")
else:
    print("\nNo callers found.")
