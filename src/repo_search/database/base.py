"""Base vector database interface for RepoSearch."""

import abc
from pathlib import Path
from typing import Dict, List, Optional, Union

from repo_search.models import DocumentChunk, RepositoryInfo, SearchResult


class VectorDatabase(abc.ABC):
    """Abstract base class for vector databases.
    
    This interface defines the common operations that any vector database implementation
    should support.
    """

    @abc.abstractmethod
    def store_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Store document chunks in the database.

        Args:
            chunks: List of document chunks to store.
        """
        pass

    @abc.abstractmethod
    def search(
        self,
        query: str,
        repository: Optional[str] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
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
        pass

    @abc.abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get a document chunk by ID.

        Args:
            chunk_id: ID of the chunk to retrieve.

        Returns:
            Document chunk, or None if not found.
        """
        pass

    @abc.abstractmethod
    def list_repositories(self) -> List[RepositoryInfo]:
        """List all repositories in the database.

        Returns:
            List of repository information.
        """
        pass

    @abc.abstractmethod
    def add_repository(self, repository_info: RepositoryInfo) -> None:
        """Add a repository to the database.

        Args:
            repository_info: Repository information.
        """
        pass

    @abc.abstractmethod
    def get_repository(self, repository_name: str) -> Optional[RepositoryInfo]:
        """Get repository information.

        Args:
            repository_name: Repository name in the format 'owner/name'.

        Returns:
            Repository information, or None if not found.
        """
        pass

    @abc.abstractmethod
    def delete_repository(self, repository_name: str) -> bool:
        """Delete a repository and all its chunks from the database.

        Args:
            repository_name: Repository name in the format 'owner/name'.

        Returns:
            True if the repository was deleted, False if it was not found.
        """
        pass

    @abc.abstractmethod
    def clear(self) -> None:
        """Clear all data from the database."""
        pass
