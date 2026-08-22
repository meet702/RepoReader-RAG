import os

def detect_language(file_path: str) -> str:
    """
    Detects the programming language based on the file extension.
    Returns the language name or 'unknown' if not recognized.
    """
    extension = os.path.splitext(file_path)[1].lower()
    
    if extension == ".py":
        return "python"
    elif extension == ".java":
        return "java"
    elif extension in {".md", ".txt", ".json", ".xml", ".yaml", ".yml"}:
        return "text"
        
    return "unknown"
