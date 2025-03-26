"""Client API for RepoSearch."""

from pathlib import Path
from typing import Dict, List, Optional, Union

from repo_search.config import config
from repo_search.models import RepositoryInfo, SearchResult
from repo_search.search.engine import SearchEngine
from repo_search.embedding.openai import OpenAIEmbedder


class RepoSearchClient:
    """Client API for RepoSearch."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        api_key: Optional[str] = None,
        token: Optional[str] = None,
        max_tokens_per_chunk: int = 1000,
        max_tokens_per_batch: int = 4000,
    ) -> None:
        """Initialize the RepoSearch client.

        Args:
            db_path: Path to the database directory. If None, will use the path from
                config.
            api_key: OpenAI API key. If None, will use the key from config.
            token: GitHub token for authentication. If None, will use anonymous access.
            max_tokens_per_chunk: Maximum tokens allowed in a single chunk.
            max_tokens_per_batch: Maximum total tokens allowed in a single batch.
        """
        self.engine = SearchEngine(
            db_path=db_path,
            api_key=api_key,
            token=token,
        )

    def index_repository(
        self, 
        repository: str, 
        force_refresh: bool = False,
        force_redownload: bool = False,
        force_rechunk: bool = False,
        force_reembed: bool = False
    ) -> RepositoryInfo:
        """Index a GitHub repository.

        Args:
            repository: Repository name in the format 'owner/name'.
            force_refresh: If True, forces re-indexing of all steps even if commit hash is unchanged.
            force_redownload: If True, forces re-downloading the repository.
            force_rechunk: If True, forces re-chunking of the repository.
            force_reembed: If True, forces re-embedding of the repository chunks.

        Returns:
            Repository information.

        Raises:
            ValueError: If the repository does not exist or is not accessible.
        """
        return self.engine.index_repository(
            repository, 
            force_refresh=force_refresh,
            force_redownload=force_redownload,
            force_rechunk=force_rechunk,
            force_reembed=force_reembed
        )

    def semantic_search(
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
        return self.engine.search(query, repository, limit, score_threshold)

    def get_repository(self, repository: str) -> Optional[RepositoryInfo]:
        """Get repository information.

        Args:
            repository: Repository name in the format 'owner/name'.

        Returns:
            Repository information, or None if not found.
        """
        return self.engine.get_repository(repository)

    def list_repositories(self) -> List[RepositoryInfo]:
        """List all indexed repositories.

        Returns:
            List of repository information.
        """
        return self.engine.get_repositories()

    def delete_repository(self, repository: str) -> bool:
        """Delete a repository from the index.

        Args:
            repository: Repository name in the format 'owner/name'.

        Returns:
            True if the repository was deleted, False if it was not found.
        """
        return self.engine.delete_repository(repository)

    def clear(self) -> None:
        """Clear all data from the index."""
        self.engine.clear()
