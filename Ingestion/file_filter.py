import os


SUPPORTED_EXTENSIONS = {
    ".java",
    ".py",
    ".md",
    ".txt",
    ".json",
    ".xml",
    ".yaml",
    ".yml"
}


IGNORED_FILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml"
}


def is_supported_file(file_path: str) -> bool:

    filename = os.path.basename(file_path)

    if filename in IGNORED_FILES:
        return False

    extension = os.path.splitext(file_path)[1].lower()

    return extension in SUPPORTED_EXTENSIONS