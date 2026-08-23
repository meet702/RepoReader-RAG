import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Ingestion')))

from code_graph.graph_builder import _build_python_graph
import networkx as nx

# Create dummy python file content
dummy_content = """
class A:
    def save(self):
        print("Saving A")

class B:
    def save(self):
        print("Saving B")

def save():
    print("Saving global")
"""

dummy_path = "tests/dummy_classes.py"
with open(dummy_path, "w") as f:
    f.write(dummy_content)

graph = nx.MultiDiGraph()
_build_python_graph(graph, dummy_path, ".")

print("Generated Nodes:")
for node in graph.nodes():
    print(f"  - {node}")

# Clean up
os.remove(dummy_path)
