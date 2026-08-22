import ast
import os
import javalang

from langchain_core.documents import Document
from language_detector import detect_language


class ASTChunker:

    def chunk_file(self, file_path: str, repo_path: str):

        language = detect_language(file_path)

        if language == "python":
            return self._chunk_python(
                file_path,
                repo_path
            )

        elif language == "java":
            return self._chunk_java(
                file_path,
                repo_path
            )

        elif language == "text":
            return self._chunk_text(
                file_path,
                repo_path
            )

        return []

    # ---------------------------------------------------------
    # Python
    # ---------------------------------------------------------

    def _chunk_python(
        self,
        file_path,
        repo_path
    ):

        content = self._read_file(file_path)

        try:
            tree = ast.parse(content)
        except SyntaxError:
            print(f"Skipping invalid Python file: {file_path}")
            return []

        relative_path = os.path.relpath(
            file_path,
            repo_path
        )

        chunks = []
        lines = content.splitlines()

        class _Visitor(ast.NodeVisitor):
            def __init__(self):
                self.classes = []
                self.functions = []
                self.methods = []
                self.current_class = None

            def visit_ClassDef(self, node):
                self.classes.append(node)
                prev_class = self.current_class
                self.current_class = node.name
                self.generic_visit(node)
                self.current_class = prev_class

            def visit_FunctionDef(self, node):
                if self.current_class:
                    self.methods.append((self.current_class, node))
                else:
                    self.functions.append(node)

            def visit_AsyncFunctionDef(self, node):
                self.visit_FunctionDef(node)

        visitor = _Visitor()
        visitor.visit(tree)

        # Process top-level functions
        for node in visitor.functions:
            start_line = node.lineno
            end_line = getattr(node, "end_lineno", start_line)
            code = "\n".join(lines[start_line - 1:end_line])

            chunks.append(
                Document(
                    page_content=code,
                    metadata={
                        "file": relative_path,
                        "language": "python",
                        "chunk_type": "function",
                        "function": node.name,
                        "start_line": start_line,
                        "end_line": end_line
                    }
                )
            )

        # Process classes
        for node in visitor.classes:
            start_line = node.lineno

            # Get method names for this class specifically
            # Note: node.body contains the immediate children
            method_names = []
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_names.append(child.name)

            bases = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    bases.append(b.id)
                elif isinstance(b, ast.Attribute):
                    bases.append(b.attr)
            
            base_str = f"({', '.join(bases)})" if bases else ""
            summary_lines = [f"class {node.name}{base_str}:"]

            docstring = ast.get_docstring(node)
            if docstring:
                summary_lines.append('    """')
                for line in docstring.splitlines():
                    summary_lines.append(f"    {line}" if line else "    ")
                summary_lines.append('    """')

            if method_names:
                summary_lines.append(f"    # Methods: {', '.join(method_names)}")
            else:
                summary_lines.append("    pass")

            chunks.append(
                Document(
                    page_content="\n".join(summary_lines),
                    metadata={
                        "file": relative_path,
                        "language": "python",
                        "chunk_type": "class",
                        "class": node.name,
                        "start_line": start_line
                    }
                )
            )

        # Process methods
        for class_name, node in visitor.methods:
            start_line = node.lineno
            end_line = getattr(node, "end_lineno", start_line)
            code = "\n".join(lines[start_line - 1:end_line])

            chunks.append(
                Document(
                    page_content=code,
                    metadata={
                        "file": relative_path,
                        "language": "python",
                        "chunk_type": "method",
                        "class": class_name,
                        "method": node.name,
                        "start_line": start_line,
                        "end_line": end_line
                    }
                )
            )

        return chunks

    # ---------------------------------------------------------
    # Java
    # ---------------------------------------------------------

    def _chunk_java(
        self,
        file_path,
        repo_path
    ):
        content = self._read_file(file_path)

        try:
            tree = javalang.parse.parse(content)
        except Exception as e:
            print(f"\nERROR PARSING JAVA FILE")
            print(f"File: {file_path}")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {e}")
            return []

        relative_path = os.path.relpath(file_path, repo_path)
        lines = content.splitlines()
        chunks = []

        class_nodes = list(tree.filter(javalang.tree.ClassDeclaration))
        interface_nodes = list(tree.filter(javalang.tree.InterfaceDeclaration))
        constructor_nodes = list(tree.filter(javalang.tree.ConstructorDeclaration))
        method_nodes = list(tree.filter(javalang.tree.MethodDeclaration))

        # ---------------------------------------------------------
        # Class / Interface chunks — signature + fields + member list
        # ---------------------------------------------------------
        for path, node in class_nodes + interface_nodes:
            chunk_type = "class" if isinstance(node, javalang.tree.ClassDeclaration) else "interface"
            start_line = node.position.line if node.position else 1

            field_lines = []
            for field in getattr(node, "fields", []):
                field_line = field.position.line if field.position else start_line
                field_lines.append(lines[field_line - 1].strip())

            constructor_signatures = [
                f"{node.name}({', '.join(p.type.name for p in c.parameters)})"
                for _, c in constructor_nodes
                if c.position and self._belongs_to(c, node, lines)
            ]

            method_signatures = [
                f"{m.name}({', '.join(p.type.name for p in m.parameters)})"
                for _, m in method_nodes
                if m.position and self._belongs_to(m, node, lines)
            ]

            summary_lines = [f"{'class' if chunk_type == 'class' else 'interface'} {node.name} {{"]
            summary_lines.extend(f"    {f}" for f in field_lines)
            if constructor_signatures:
                summary_lines.append(f"    // Constructor: {', '.join(constructor_signatures)}")
            if method_signatures:
                summary_lines.append(f"    // Methods: {', '.join(method_signatures)}")
            summary_lines.append("}")

            chunks.append(
                Document(
                    page_content="\n".join(summary_lines),
                    metadata={
                        "file": relative_path,
                        "language": "java",
                        "chunk_type": chunk_type,
                        "class": node.name,
                        "start_line": start_line
                    }
                )
            )

        # ---------------------------------------------------------
        # Constructor chunks
        # ---------------------------------------------------------
        for path, node in constructor_nodes:
            start_line = node.position.line if node.position else 1
            code = self._extract_java_block(lines, start_line - 1)
            chunks.append(
                Document(
                    page_content=code,
                    metadata={
                        "file": relative_path,
                        "language": "java",
                        "chunk_type": "constructor",
                        "constructor": node.name,
                        "start_line": start_line
                    }
                )
            )

        # ---------------------------------------------------------
        # Method chunks
        # ---------------------------------------------------------
        for path, node in method_nodes:
            start_line = node.position.line if node.position else 1
            code = self._extract_java_block(lines, start_line - 1)
            chunks.append(
                Document(
                    page_content=code,
                    metadata={
                        "file": relative_path,
                        "language": "java",
                        "chunk_type": "method",
                        "method": node.name,
                        "start_line": start_line
                    }
                )
            )

        return chunks

    def _belongs_to(self, member_node, class_node, lines):
        """Rough check: does this member's line fall within the class's brace block?"""
        if not member_node.position or not class_node.position:
            return False
        class_start = class_node.position.line - 1
        class_block = self._extract_java_block(lines, class_start)
        class_end = class_start + len(class_block.splitlines())
        return class_start <= member_node.position.line - 1 < class_end

    # ---------------------------------------------------------
    # Text / Markdown / Config
    # ---------------------------------------------------------

    def _chunk_text(
        self,
        file_path,
        repo_path
    ):

        content = self._read_file(file_path)

        relative_path = os.path.relpath(
            file_path,
            repo_path
        )

        # For the first version we keep these files
        # as a single logical document.
        return [
            Document(
                page_content=content,
                metadata={
                    "file": relative_path,
                    "language": "text",
                    "chunk_type": "document"
                }
            )
        ]

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _read_file(self, file_path):

        try:
            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:
                return file.read()

        except UnicodeDecodeError:

            with open(
                file_path,
                "r",
                encoding="latin-1"
            ) as file:
                return file.read()

    def _extract_java_method(
        self,
        lines,
        start_index
    ):

        return self._extract_java_block(
            lines,
            start_index
        )

    def _extract_java_block(
        self,
        lines,
        start_index
    ):

        collected = []

        brace_count = 0
        started = False

        for i in range(
            start_index,
            len(lines)
        ):

            line = lines[i]

            collected.append(line)

            brace_count += line.count("{")
            brace_count -= line.count("}")

            if "{" in line:
                started = True

            if started and brace_count == 0:
                break

        return "\n".join(collected)