"""Data models for RepoSearch."""

from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class RepositoryInfo(BaseModel):
    """Information about a GitHub repository."""

    owner: str
    name: str
    url: str
    last_indexed: Optional[datetime] = None
    num_files: int = 0
    num_chunks: int = 0
    commit_hash: Optional[str] = None
    download_successful: bool = False
    chunking_successful: bool = False
    embedding_successful: bool = False

    @property
    def full_name(self) -> str:
        """Get the full name of the repository (owner/name)."""
        return f"{self.owner}/{self.name}"


class DocumentChunk(BaseModel):
    """A chunk of text from a document."""

    id: str
    repository: str  # owner/name
    content: str
    metadata: Dict[str, Union[str, int, float, bool, None]] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

    @property
    def file_path(self) -> Optional[str]:
        """Get the file path from metadata."""
        return self.metadata.get("file_path")

    @property
    def chunk_type(self) -> str:
        """Get the chunk type from metadata."""
        return self.metadata.get("chunk_type", "text")

    @property
    def start_line(self) -> Optional[int]:
        """Get the start line from metadata."""
        start_line = self.metadata.get("start_line")
        return int(start_line) if start_line is not None else None

    @property
    def end_line(self) -> Optional[int]:
        """Get the end line from metadata."""
        end_line = self.metadata.get("end_line")
        return int(end_line) if end_line is not None else None


class SearchResult(BaseModel):
    """A search result from the vector database."""

    chunk: DocumentChunk
    score: float
    
    @property
    def content(self) -> str:
        """Get the content of the chunk."""
        return self.chunk.content
    
    @property
    def source(self) -> str:
        """Get a source description for the chunk."""
        if self.chunk.file_path:
            repo = self.chunk.repository
            path = self.chunk.file_path
            start = self.chunk.start_line
            end = self.chunk.end_line
            if start is not None and end is not None:
                return f"{repo} - {path}:{start}-{end}"
            return f"{repo} - {path}"
        return self.chunk.repository
