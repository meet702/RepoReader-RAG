import ast

source = """
@dataclass
class TaskModel:
    pass

@app.route("/api/tasks", methods=["GET"])
def get_all_tasks():
    pass
"""

tree = ast.parse(source)
for node in tree.body:
    print(f"Node: {type(node).__name__}")
    print(f"  node.lineno = {node.lineno}")
    if hasattr(node, 'decorator_list') and node.decorator_list:
        print(f"  node.decorator_list[0].lineno = {node.decorator_list[0].lineno}")
