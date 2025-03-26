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

    def index_repository(self, repository: str) -> RepositoryInfo:
        """Index a GitHub repository.

        Args:
            repository: Repository name in the format 'owner/name'.

        Returns:
            Repository information.
        """
        # Check if the repository is already indexed
        existing_repo = self.db.get_repository(repository)
        if existing_repo:
            print(f"Repository {repository} is already indexed.")
            print(f"Last indexed: {existing_repo.last_indexed}")
            print(f"Number of chunks: {existing_repo.num_chunks}")
            return existing_repo

        # Create a temporary directory for the repository contents
        temp_dir = tempfile.mkdtemp(prefix=f"reposearch_")
        try:
            # Fetch the repository contents
            print(f"Fetching repository {repository}...")
            repo_info, repo_dir = self.repo_fetcher.fetch_repository_contents(
                repository, Path(temp_dir)
            )

            # Store repository info
            self.db.add_repository(repo_info)

            # Chunk the repository contents
            print(f"Chunking repository contents...")
            chunks = list(self.chunker.chunk_repository(repository, repo_dir))
            print(f"Generated {len(chunks)} chunks.")

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
