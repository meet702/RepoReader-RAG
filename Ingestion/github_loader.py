import os
import git
from git.exc import GitCommandError

def clone_repo(repo_url: str, dest_dir: str = "cloned_repos") -> str:
    """
    Clones a public GitHub repository or pulls the latest changes if it already exists.
    Returns the local path to the repository.
    """
    if not repo_url.startswith("http"):
        raise ValueError(f"Invalid repository URL: {repo_url}")

    # Extract repo name from URL (e.g., 'myrepo' from '.../user/myrepo' or '.../myrepo.git')
    repo_name = repo_url.rstrip("/").split("/")[-1]
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    repo_path = os.path.join(dest_dir, repo_name)

    os.makedirs(dest_dir, exist_ok=True)

    if os.path.exists(repo_path) and os.path.isdir(os.path.join(repo_path, ".git")):
        print(f"Repository already exists at {repo_path}. Pulling latest changes...")
        try:
            repo = git.Repo(repo_path)
            origin = repo.remotes.origin
            origin.pull()
            print("Successfully pulled latest changes.")
            return os.path.abspath(repo_path)
        except GitCommandError as e:
            raise RuntimeError(f"Failed to pull repository at {repo_path}. Git error: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error while updating repository {repo_path}: {e}")
    else:
        print(f"Cloning repository {repo_url} into {repo_path}...")
        try:
            git.Repo.clone_from(repo_url, repo_path)
            print("Successfully cloned repository.")
            return os.path.abspath(repo_path)
        except GitCommandError as e:
            raise RuntimeError(f"Failed to clone repository from {repo_url}. Ensure the URL is correct and public. Git error: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error while cloning {repo_url}: {e}")