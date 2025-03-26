"""GitHub repository handling functionality."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import requests
from github import Github
from github.Repository import Repository
from tqdm import tqdm

from repo_search.config import config
from repo_search.models import RepositoryInfo


class GitHubRepositoryFetcher:
    """Fetches content from GitHub repositories."""

    def __init__(self, token: Optional[str] = None) -> None:
        """Initialize a GitHub repository fetcher.

        Args:
            token: GitHub token for authentication. If None, will use anonymous access
                which has stricter rate limits.
        """
        self.token = token or config.github_token
        self.github = Github(self.token) if self.token else Github()

    def get_repository_info(self, repo_name: str) -> RepositoryInfo:
        """Get information about a repository.

        Args:
            repo_name: Repository name in the format 'owner/name'.

        Returns:
            Information about the repository.

        Raises:
            ValueError: If the repository does not exist or is not accessible.
        """
        try:
            owner, name = repo_name.split("/", 1)
        except ValueError:
            raise ValueError(f"Invalid repository name: {repo_name}. Expected format: owner/name")

        try:
            repo = self.github.get_repo(repo_name)
            # Get the latest commit hash
            commit_hash = repo.get_commits()[0].sha
            return RepositoryInfo(
                owner=owner,
                name=name,
                url=repo.html_url,
                commit_hash=commit_hash,
            )
        except Exception as e:
            raise ValueError(f"Error accessing repository {repo_name}: {e}")

    def fetch_repository_contents(
        self, repo_name: str, output_dir: Optional[Path] = None
    ) -> Tuple[RepositoryInfo, Path]:
        """Fetch the contents of a repository.

        Args:
            repo_name: Repository name in the format 'owner/name'.
            output_dir: Directory where to store the cloned repository. If None, will
                create a temporary directory.

        Returns:
            Tuple of (repository info, directory containing the repository contents).

        Raises:
            ValueError: If the repository does not exist or is not accessible.
        """
        repo_info = self.get_repository_info(repo_name)

        if output_dir is None:
            # Create a temporary directory for the repository contents
            temp_dir = tempfile.mkdtemp(prefix=f"{repo_info.owner}_{repo_info.name}_")
            output_dir = Path(temp_dir)
        else:
            output_dir.mkdir(exist_ok=True, parents=True)

        self._download_repository(repo_info, output_dir)
        return repo_info, output_dir

    def _download_repository(self, repo_info: RepositoryInfo, output_dir: Path) -> None:
        """Download the contents of a repository.

        Args:
            repo_info: Repository information.
            output_dir: Directory where to store the repository contents.
        """
        repo = self.github.get_repo(repo_info.full_name)
        contents = self._get_all_files(repo)

        print(f"Downloading {len(contents)} files from {repo_info.full_name}...")
        successfully_downloaded = 0
        for content_file in tqdm(contents):
            # Skip directories
            if content_file.type == "dir":
                continue

            try:
                # Check if the file has a supported encoding
                if content_file.encoding != "base64":
                    print(f"Skipping file with unsupported encoding '{content_file.encoding}': {content_file.path}")
                    continue

                # Get the content
                file_content = content_file.decoded_content
                # Create the file path
                file_path = output_dir / content_file.path
                # Create the parent directory if it doesn't exist
                file_path.parent.mkdir(exist_ok=True, parents=True)
                # Write the content to the file
                file_path.write_bytes(file_content)
                successfully_downloaded += 1
            except AssertionError as e:
                # This handles the "unsupported encoding: none" error
                print(f"Skipping file due to encoding error: {content_file.path} - {str(e)}")
                continue
            except Exception as e:
                print(f"Error downloading file {content_file.path}: {str(e)}")
                continue

        repo_info.num_files = successfully_downloaded
        print(f"Successfully downloaded {successfully_downloaded} of {len(contents)} files")

    def _get_all_files(self, repo: Repository) -> List[Dict]:
        """Get all files in a repository.

        Args:
            repo: Repository object.

        Returns:
            List of file contents.
        """
        contents = []
        self._get_contents_recursive(repo, "", contents)
        return contents

    def _get_contents_recursive(
        self, repo: Repository, path: str, contents: List
    ) -> None:
        """Recursively get all contents of a repository.

        Args:
            repo: Repository object.
            path: Path within the repository.
            contents: List to append contents to.
        """
        items = repo.get_contents(path)
        for item in items:
            if item.type == "dir":
                self._get_contents_recursive(repo, item.path, contents)
            else:
                contents.append(item)

    def is_text_file(self, file_path: Path) -> bool:
        """Check if a file is a text file.

        Args:
            file_path: Path to the file.

        Returns:
            True if the file is a text file, False otherwise.
        """
        # Text extensions - be more specific to avoid processing non-text files
        text_extensions = {
            # Documentation
            ".txt", ".md", ".rst", ".adoc", ".asciidoc", 
            
            # Web
            ".html", ".htm", ".css", ".scss", ".sass", ".less",
            ".js", ".jsx", ".ts", ".tsx", ".vue", ".svelte",
            
            # Config
            ".json", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", 
            ".properties", ".env", ".gitignore", ".gitconfig", ".gitattributes",
            
            # Code
            ".py", ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".go",
            ".rs", ".swift", ".kt", ".scala", ".sh", ".bash", ".zsh", ".fish", 
            ".sql", ".graphql", ".proto", ".cmake", ".mk", ".Makefile",
            
            # Structured data that should be reasonably sized
            ".csv", ".tsv",
        }
        
        # Max file size for text files (5MB)
        MAX_TEXT_FILE_SIZE = 5 * 1024 * 1024
        
        # Check file size first - large files are likely binary
        if file_path.stat().st_size > MAX_TEXT_FILE_SIZE:
            print(f"Skipping large file (> 5MB): {file_path}")
            return False
        
        # Check extension - only process files with known text extensions
        ext = file_path.suffix.lower()
        if ext in text_extensions:
            # Still try to read a small sample to confirm it's text
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    sample = f.read(1024)
                    
                    # Check for binary characters in the sample
                    binary_chars = sum(1 for c in sample if ord(c) < 9 or (ord(c) > 13 and ord(c) < 32))
                    if binary_chars > len(sample) * 0.1:  # More than 10% binary chars
                        print(f"Skipping likely binary file despite text extension: {file_path}")
                        return False
                        
                return True
            except UnicodeDecodeError:
                try:
                    # Try one more encoding
                    with open(file_path, "r", encoding="latin-1") as f:
                        sample = f.read(1024)
                        binary_chars = sum(1 for c in sample if ord(c) < 9 or (ord(c) > 13 and ord(c) < 32))
                        if binary_chars > len(sample) * 0.1:
                            return False
                    return True
                except:
                    return False
            except Exception:
                return False
        
        # By default, consider unknown extensions as non-text
        return False

    def get_text_files(self, directory: Path) -> Iterator[Path]:
        """Get all text files in a directory.

        Args:
            directory: Directory to search.

        Returns:
            Iterator of paths to text files.
        """
        for file_path in directory.glob("**/*"):
            if file_path.is_file() and self.is_text_file(file_path):
                yield file_path
