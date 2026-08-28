import os
from langchain_core.tools import tool

def build_tools(persist_directory: str, all_chunks: list, graph=None, repo_path: str = ""):
    """
    Factory function that returns a list of configured tools.
    These tools are closures over the specific repo's runtime state,
    eliminating global variables and preventing race conditions during concurrent API requests.
    """

    @tool
    def code_search_tool(query: str) -> str:
        """Search the ingested repository for code, functions, classes, and documentation.
        Use this tool when you need to read the actual code or find where something is defined.
        IMPORTANT: Use specific, targeted queries for best results — use class names (e.g. 'Task'),
        method names (e.g. 'createTask'), or Java/Spring annotations (e.g. '@Entity', '@RestController',
        '@GetMapping'). Vague phrases like 'find the models' or 'search for models' produce poor
        results. If unsure of the exact name, try the most likely class or annotation name."""
        
        # We must import run_pipeline here to avoid circular imports, 
        # or we just assume run_pipeline is available.
        # Actually, let's import it here.
        from retrieval.pipeline import run_pipeline
        
        # We need the vectorstore/chunks to run the pipeline.
        if not persist_directory or not all_chunks:
            return "Error: Tools not properly initialized with vector database context."
            
        try:
            results = run_pipeline(query, persist_directory, all_chunks)
            if not results:
                return "No relevant code found for your query."
                
            formatted = []
            for doc in results:
                file_path = doc.metadata.get("file", "Unknown file")
                start_line = doc.metadata.get("start_line", "")
                
                header = f"[{file_path}]"
                if start_line:
                    header += f" (Line {start_line})"
                    
                formatted.append(f"{header}\n{doc.page_content}")
                
            return "\n\n---\n\n".join(formatted)
        except Exception as e:
            return f"Error executing code search: {str(e)}"

    @tool
    def github_search_tool(query: str) -> str:
        """Search GitHub issues, pull requests, and commit history for context on why 
        changes were made, historical bugs, and feature discussions."""
        return "GitHub search is not yet implemented."

    @tool
    def graph_search_tool(query: str) -> str:
        """Search the code relationship graph to find callers, callees, dependencies,
        and structural relationships between code elements. Use this for questions
        like 'what calls X', 'what does Y depend on', or 'what imports Z'.
        NOTE: Provide the actual function name 'X' you want to lookup in the query."""
        
        if not graph:
            return "Error: graph not initialized or not found."
            
        from code_graph.graph_store import get_callers, get_callees
        
        # Extract likely function name from query (simple heuristic)
        words = query.replace("'", "").replace('"', "").replace("`", "").split()
        target = sorted(words, key=len, reverse=True)[0]
        
        callers = get_callers(graph, target)
        callees = get_callees(graph, target)
        
        result = []
        if callers:
            result.append(f"Found {len(callers)} caller(s) for '{target}':\n" + "\n".join(f"- {c}" for c in callers))
        if callees:
            result.append(f"Found {len(callees)} callee(s) that '{target}' calls:\n" + "\n".join(f"- {c}" for c in callees))
            
        if not result:
            return f"No structural relationships or callers found for '{target}'."
            
        return "\n\n".join(result)
        
    @tool
    def list_directory_tool(path: str = "") -> str:
        """List the files and folders in the repository. Use this tool for questions like:
        'what files exist in this project', 'show me the project structure',
        'what's in the src directory', 'what controller files are present',
        'give me the file structure'. This tool does a real filesystem walk — it is
        deterministic and always shows the actual files on disk.
        Do NOT use code_search_tool for these questions — it retrieves by meaning,
        not by listing files.

        IMPORTANT: If you do not know the exact folder layout, ALWAYS call this tool
        first with path='' (empty string / no argument) to see the real top-level
        structure. Only after seeing the real folder names should you call again with
        a specific subpath (e.g. 'src/main/java'). Never guess or invent a path —
        always derive it from a previous tool response.

        If the output contains '... (max depth reached)' for a directory you care
        about, call this tool again with that directory's exact path to see its
        contents. Never guess or fabricate filenames — always use a follow-up call."""

        if not repo_path:
            return "Error: repo_path not configured. The server may need to be restarted after re-ingesting."

        # Resolve the target directory
        target = os.path.join(repo_path, path) if path else repo_path
        target = os.path.normpath(target)

        # Security: make sure we are still inside repo_path
        if not target.startswith(os.path.normpath(repo_path)):
            return "Error: path escapes the repository root."

        if not os.path.isdir(target):
            return f"Error: '{path}' is not a directory inside the repository."

        EXCLUDED_DIRS = {
            ".git", "node_modules", "venv", ".venv", "__pycache__",
            "dist", "build", "target", ".idea", ".vscode"
        }
        MAX_DEPTH = 20

        lines = [f"Repository: {os.path.basename(repo_path)}"]
        if path:
            lines[0] += f" / {path}"
        lines.append("")

        def _walk(directory: str, prefix: str, depth: int):
            if depth > MAX_DEPTH:
                lines.append(prefix + "... (max depth reached)")
                return
            try:
                entries = sorted(os.listdir(directory))
            except PermissionError:
                lines.append(prefix + "[permission denied]")
                return

            # Separate dirs and files, skip excluded dirs
            dirs = [e for e in entries if os.path.isdir(os.path.join(directory, e)) and e not in EXCLUDED_DIRS]
            files = [e for e in entries if os.path.isfile(os.path.join(directory, e))]

            for i, d in enumerate(dirs):
                connector = "+-- " if (i < len(dirs) - 1 or files) else "+-- "
                lines.append(prefix + connector + d + "/")
                _walk(os.path.join(directory, d), prefix + "|   ", depth + 1)

            for i, f in enumerate(files):
                connector = "+-- " if i < len(files) - 1 else "\\-- "
                lines.append(prefix + connector + f)

        _walk(target, "", 1)
        return "\n".join(lines)

    return [code_search_tool, github_search_tool, graph_search_tool, list_directory_tool]
