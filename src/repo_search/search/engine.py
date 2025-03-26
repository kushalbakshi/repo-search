"""Search engine for RepoSearch."""

import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

from repo_search.config import config
from repo_search.database.base import VectorDatabase
from repo_search.database.chroma import ChromaVectorDatabase
from repo_search.embedding.openai import OpenAIEmbedder
from repo_search.github.repository import GitHubRepositoryFetcher
from repo_search.models import DocumentChunk, RepositoryInfo, SearchResult
from repo_search.processing.chunker import RepositoryChunker


class SearchEngine:
    """Search engine for GitHub repositories."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        api_key: Optional[str] = None,
        token: Optional[str] = None,
    ) -> None:
        """Initialize the search engine.

        Args:
            db_path: Path to the database directory. If None, will use the path from
                config.
            api_key: OpenAI API key. If None, will use the key from config.
            token: GitHub token for authentication. If None, will use anonymous access.
        """
        self.db_path = db_path or config.db_path
        self.api_key = api_key or config.openai_api_key
        self.token = token or config.github_token

        # Initialize components
        self.embedder = OpenAIEmbedder(api_key=self.api_key)
        self.db = ChromaVectorDatabase(db_path=self.db_path, embedder=self.embedder)
        self.repo_fetcher = GitHubRepositoryFetcher(token=self.token)
        self.chunker = RepositoryChunker()

    def index_repository(self, repository: str, force_refresh: bool = False) -> RepositoryInfo:
        """Index a GitHub repository.

        Args:
            repository: Repository name in the format 'owner/name'.
            force_refresh: If True, forces re-indexing even if commit hash is unchanged.

        Returns:
            Repository information.
        """
        # First, get current repository info (with the latest commit hash)
        print(f"Checking repository {repository}...")
        try:
            # This will fetch the latest repo info with commit hash
            current_repo_info = self.repo_fetcher.get_repository_info(repository)
        except Exception as e:
            print(f"Error getting repository info: {e}")
            raise

        # Check if the repository is already indexed
        existing_repo = self.db.get_repository(repository)
        
        # If already indexed with the same commit hash, we can skip re-indexing
        if (existing_repo and 
            existing_repo.commit_hash and 
            existing_repo.commit_hash == current_repo_info.commit_hash and 
            not force_refresh):
            print(f"Repository {repository} is already indexed with the same commit hash.")
            print(f"Latest commit: {existing_repo.commit_hash}")
            print(f"Last indexed: {existing_repo.last_indexed}")
            print(f"Number of chunks: {existing_repo.num_chunks}")
            return existing_repo
            
        # Repository needs to be indexed (new, changed, or forced refresh)
        if existing_repo:
            if force_refresh:
                print(f"Forcing refresh of repository {repository}...")
            else:
                print(f"Repository {repository} has changed (commit {current_repo_info.commit_hash}).")
                print(f"Previous commit: {existing_repo.commit_hash or 'unknown'}")
        else:
            print(f"Repository {repository} is not yet indexed.")

        # Create a temporary directory for the repository contents
        temp_dir = tempfile.mkdtemp(prefix=f"reposearch_")
        try:
            # Fetch the repository contents
            print(f"Fetching repository {repository}...")
            repo_info, repo_dir = self.repo_fetcher.fetch_repository_contents(
                repository, Path(temp_dir)
            )
            
            # Make sure we keep the commit hash
            if not repo_info.commit_hash:
                repo_info.commit_hash = current_repo_info.commit_hash

            # Store repository info
            self.db.add_repository(repo_info)

            # Chunk the repository contents
            print(f"Chunking repository contents...")
            try:
                chunks = []
                for chunk in self.chunker.chunk_repository(repository, repo_dir):
                    chunks.append(chunk)
                print(f"Generated {len(chunks)} chunks.")
            except UnicodeDecodeError as e:
                print(f"Warning: Skipping some files due to encoding issues: {e}")

            # Embed and store the chunks
            print(f"Embedding and storing chunks...")
            self.db.store_chunks(chunks)

            # Update repository info
            repo_info.num_chunks = len(chunks)
            self.db.add_repository(repo_info)

            return repo_info
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)

    def search(
        self,
        query: str,
        repository: Optional[str] = None,
        limit: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Search for documents similar to a query.

        Args:
            query: Query text.
            repository: Optional repository to search in (owner/name).
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score for results.

        Returns:
            List of search results.
        """
        limit = limit or config.max_results
        score_threshold = score_threshold or config.score_threshold
        return self.db.search(query, repository, limit, score_threshold)

    def get_repository(self, repository: str) -> Optional[RepositoryInfo]:
        """Get repository information.

        Args:
            repository: Repository name in the format 'owner/name'.

        Returns:
            Repository information, or None if not found.
        """
        return self.db.get_repository(repository)

    def get_repositories(self) -> List[RepositoryInfo]:
        """Get all indexed repositories.

        Returns:
            List of repository information.
        """
        return self.db.list_repositories()

    def delete_repository(self, repository: str) -> bool:
        """Delete a repository from the index.

        Args:
            repository: Repository name in the format 'owner/name'.

        Returns:
            True if the repository was deleted, False if it was not found.
        """
        return self.db.delete_repository(repository)

    def clear(self) -> None:
        """Clear all data from the index."""
        self.db.clear()
