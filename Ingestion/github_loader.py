import os
import shutil
import tempfile

from git import Repo


class GitHubLoader:

    def __init__(self, repo_url: str):
        self.repo_url = repo_url

    def clone_repository(self):
        """
        Clone a GitHub repository into a temporary directory.
        """

        temp_dir = tempfile.mkdtemp(prefix="github_repo_")

        print(f"Cloning repository...")
        print(f"Repository: {self.repo_url}")

        try:
            Repo.clone_from(
                self.repo_url,
                temp_dir,
                depth=1
            )

            print(f"Repository cloned successfully.")
            print(f"Location: {temp_dir}")

            return temp_dir

        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(
                f"Failed to clone repository: {e}"
            )

    def get_repository_files(self, repo_path: str):
        """
        Return all files from the repository.
        """

        files = []

        ignored_directories = {
            ".git",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            "env",
            "target",
            "build",
            "dist",
            ".idea",
            ".vscode"
        }

        for root, dirs, filenames in os.walk(repo_path):

            # Prevent traversal into ignored directories
            dirs[:] = [
                d for d in dirs
                if d not in ignored_directories
            ]

            for filename in filenames:

                file_path = os.path.join(
                    root,
                    filename
                )

                files.append(file_path)

        return files

    def cleanup(self, repo_path: str):
        """
        Remove cloned repository after processing.
        """

        if os.path.exists(repo_path):
            shutil.rmtree(
                repo_path,
                ignore_errors=True
            )