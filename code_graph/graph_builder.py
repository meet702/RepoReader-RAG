import os
import ast
import javalang
import networkx as nx
from Ingestion.language_detector import detect_language

def _get_call_name(node: ast.Call) -> str:
    """Extract the base function or method name from an ast.Call node."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    elif isinstance(node.func, ast.Attribute):
        return node.func.attr
    return "unknown"

class PythonGraphVisitor(ast.NodeVisitor):
    def __init__(self, graph: nx.MultiDiGraph, file_node_name: str, file_path: str):
        self.graph = graph
        self.file_node_name = file_node_name
        self.file_path = file_path
        self.current_context = None
        self.current_class = None

    def visit_FunctionDef(self, node):
        prev_context = self.current_context
        # Create a unique node name for this function
        if self.current_class:
            self.current_context = f"{self.file_node_name}::{self.current_class}.{node.name}"
        else:
            self.current_context = f"{self.file_node_name}::{node.name}"
        self.generic_visit(node)
        self.current_context = prev_context

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        prev_class = self.current_class
        prev_context = self.current_context
        self.current_class = node.name
        self.current_context = f"{self.file_node_name}::{node.name}"
        self.generic_visit(node)
        self.current_class = prev_class
        self.current_context = prev_context

    def visit_Call(self, node):
        if self.current_context:
            target_name = _get_call_name(node)
            if target_name != "unknown":
                # NOTE: Callee target nodes are unqualified bare names (not file::name).
                # This means we do NOT resolve which specific function is meant when 
                # multiple files have same-named functions. This is an intentional 
                # simplification, not a bug, and should be documented as such for 
                # anyone reading the code later.
                self.graph.add_edge(
                    self.current_context,
                    target_name,
                    type="calls",
                    source_file=self.file_path,
                    line_number=node.lineno
                )
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.graph.add_edge(
                self.file_node_name,
                alias.name,
                type="imports",
                source_file=self.file_path,
                line_number=node.lineno
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            for alias in node.names:
                target = f"{node.module}.{alias.name}"
                self.graph.add_edge(
                    self.file_node_name,
                    target,
                    type="imports",
                    source_file=self.file_path,
                    line_number=node.lineno
                )
        self.generic_visit(node)

def _build_python_graph(graph: nx.MultiDiGraph, file_path: str, repo_path: str):
    relative_path = os.path.relpath(file_path, repo_path).replace("\\", "/")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            content = f.read()
            
    try:
        tree = ast.parse(content)
    except SyntaxError:
        print(f"Skipping invalid Python file for graph: {file_path}")
        return

    visitor = PythonGraphVisitor(graph, relative_path, relative_path)
    visitor.visit(tree)

def _build_java_graph(graph: nx.MultiDiGraph, file_path: str, repo_path: str):
    relative_path = os.path.relpath(file_path, repo_path).replace("\\", "/")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            content = f.read()
            
    try:
        tree = javalang.parse.parse(content)
    except Exception:
        print(f"Skipping invalid Java file for graph: {file_path}")
        return

    # Keep it simple: look for method declarations and then method invocations inside them
    for path, method_node in tree.filter(javalang.tree.MethodDeclaration):
        method_name = method_node.name
        caller_node_name = f"{relative_path}::{method_name}"
        
        for _, invocation in method_node.filter(javalang.tree.MethodInvocation):
            target_name = invocation.member
            
            # NOTE: Callee target nodes are unqualified bare names (not file::name).
            # This means we do NOT resolve which specific function is meant when 
            # multiple files have same-named functions. This is an intentional 
            # simplification, not a bug, and should be documented as such for 
            # anyone reading the code later.
            line_no = invocation.position.line if invocation.position else (method_node.position.line if method_node.position else 1)
            
            graph.add_edge(
                caller_node_name,
                target_name,
                type="calls",
                source_file=relative_path,
                line_number=line_no
            )

def build_graph(repo_path: str, parseable_files: list[str]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    
    for file_path in parseable_files:
        language = detect_language(file_path)
        if language == "python":
            _build_python_graph(graph, file_path, repo_path)
        elif language == "java":
            _build_java_graph(graph, file_path, repo_path)
            
    return graph
