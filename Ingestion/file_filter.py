import os

def get_parseable_files(repo_path: str) -> list[str]:
    """
    Walks the repository directory and returns a list of file paths worth parsing.
    """
    parseable_files = []
    
    ignored_dirs = {
        ".git", "node_modules", "venv", ".venv", "__pycache__", 
        "dist", "build", "target", ".idea", ".vscode"
    }
    
    allowed_extensions = {
        ".py", ".java", ".md", ".txt", ".json", ".xml", ".yaml", ".yml"
    }
    
    MAX_FILE_SIZE_BYTES = 500 * 1024  # 500 KB
    
    for root, dirs, files in os.walk(repo_path):
        # Modify dirs in-place to prevent walking into ignored directories
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        
        for file in files:
            extension = os.path.splitext(file)[1].lower()
            if extension in allowed_extensions:
                file_path = os.path.join(root, file)
                
                # Check file size
                try:
                    file_size = os.path.getsize(file_path)
                    if file_size <= MAX_FILE_SIZE_BYTES:
                        parseable_files.append(os.path.abspath(file_path))
                except OSError:
                    # Skip if we can't get the file size (e.g. broken symlink)
                    pass
                    
    return parseable_files