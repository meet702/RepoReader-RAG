import ast

source = """
@dataclass
# some comment
@app.route(
    "/api/tasks", 
    methods=["GET"]
)
class TaskModel:
    pass
"""
lines = source.splitlines()
tree = ast.parse(source)
node = tree.body[0]
decor_start = min([d.lineno for d in node.decorator_list])
print(f"decor_start: {decor_start}, node.lineno: {node.lineno}")
print("Lines between:")
for line in lines[decor_start - 1 : node.lineno - 1]:
    print(repr(line))
