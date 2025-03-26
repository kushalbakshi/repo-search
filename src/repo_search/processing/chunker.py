"""Text chunking functionality for RepoSearch."""

import re
import uuid
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
)
from langchain.docstore.document import Document as LangChainDocument

from repo_search.config import config
from repo_search.models import DocumentChunk


class TextChunker:
    """Chunks text content into semantically meaningful segments using LangChain text splitters."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        max_tokens: int = 5000,  # Reduced from 7000 to avoid embedding API limits
    ) -> None:
        """Initialize the text chunker.

        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of characters to overlap between chunks.
            max_tokens: Maximum number of tokens allowed in a single chunk.
        """
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap
        self.max_tokens = max_tokens
        
        # Maximum file size to process (2MB) - larger files will be truncated
        self.max_file_size = 2 * 1024 * 1024  # 2MB
        
        # Initialize text splitters
        self.general_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        
        self.markdown_splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        
        # Initialize specific splitters
        self.python_splitter = PythonCodeTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        
    def _estimate_tokens(self, text: str) -> int:
        """Roughly estimate the number of tokens in a text string.
        
        This is a simple heuristic: ~4 chars per token for English text.
        
        Args:
            text: Text to estimate tokens for.
            
        Returns:
            Estimated token count.
        """
        return len(text) // 4

    def chunk_file(
        self, file_path: Path, repository: str, file_content: Optional[str] = None
    ) -> List[DocumentChunk]:
        """Chunk a file into semantically meaningful segments using LangChain text splitters.

        Args:
            file_path: Path to the file.
            repository: Repository name in the format 'owner/name'.
            file_content: File content. If None, will read from file_path.

        Returns:
            List of document chunks.
        """
        # Check file size first
        if file_path.stat().st_size > self.max_file_size:
            print(f"Warning: File too large, truncating to {self.max_file_size//1024}KB: {file_path}")
            
        if file_content is None:
            try:
                # Limit reading to max_file_size
                with open(file_path, "r", encoding="utf-8") as f:
                    if file_path.stat().st_size > self.max_file_size:
                        file_content = f.read(self.max_file_size)
                        print(f"Truncated {file_path} to {self.max_file_size//1024}KB")
                    else:
                        file_content = f.read()
            except UnicodeDecodeError:
                # Try with a different encoding
                try:
                    with open(file_path, "r", encoding="latin-1") as f:
                        if file_path.stat().st_size > self.max_file_size:
                            file_content = f.read(self.max_file_size)
                            print(f"Truncated {file_path} to {self.max_file_size//1024}KB")
                        else:
                            file_content = f.read()
                except UnicodeDecodeError:
                    print(f"Skipping file with unsupported encoding: {file_path}")
                    return []
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                    return []
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                return []

        # Get file extension and relative path
        ext = file_path.suffix.lower()
        relative_path = file_path.name
        
        # Create LangChain document with metadata
        lc_document = LangChainDocument(
            page_content=file_content,
            metadata={
                "file_path": str(relative_path),
                "repository": repository,
                "extension": ext,
            }
        )
        
        # Choose the appropriate splitter based on file type
        splitter = self.general_splitter
        chunk_type = "text"
        
        if self._is_code_file(ext):
            if ext == ".py":
                # Use specialized Python code splitter for Python files
                splitter = self.python_splitter
            else:
                # For other code files, use recursive character splitter with code-focused separators
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=[
                        # Common code separators in order of priority
                        "\nclass ", "\ndef ", "\nfunction ", "\npublic ", "\nprivate ", 
                        "\nprotected ", "\nif ", "\nfor ", "\nwhile ", "\nswitch ",
                        "\ncase ", "\nconst ", "\nlet ", "\nvar ", 
                        "\n\n", "\n", " ", ""
                    ]
                )
            chunk_type = "code"
        elif ext in [".md"]:
            splitter = self.markdown_splitter
            chunk_type = "markdown"
        
        # Split the document using LangChain
        lc_chunks = splitter.split_documents([lc_document])
        
        # Convert LangChain chunks to DocumentChunk objects
        result_chunks = []
        for i, lc_chunk in enumerate(lc_chunks):
            # Calculate approximate line numbers based on chunk position
            if i == 0:
                start_line = 0
            else:
                # Estimate start line based on position in document
                content_before = file_content.split(lc_chunk.page_content)[0]
                start_line = content_before.count('\n')
            
            end_line = start_line + lc_chunk.page_content.count('\n')
            
            # Create DocumentChunk
            result_chunks.append(self._create_chunk(
                content=lc_chunk.page_content,
                repository=repository,
                file_path=str(relative_path),
                start_line=start_line,
                end_line=end_line,
                chunk_type=chunk_type,
            ))
            
        return result_chunks

    def _is_code_file(self, extension: str) -> bool:
        """Check if a file is a code file.

        Args:
            extension: File extension.

        Returns:
            True if the file is a code file, False otherwise.
        """
        code_extensions = {
            ".py", ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".js", ".jsx",
            ".ts", ".tsx", ".php", ".rb", ".go", ".rs", ".swift", ".kt",
            ".scala", ".sh", ".bash", ".zsh", ".sql",
        }
        return extension in code_extensions

    def _create_chunk(
        self,
        content: str,
        repository: str,
        file_path: str,
        start_line: int = None,
        end_line: int = None,
        chunk_type: str = "text",
        **metadata,
    ) -> DocumentChunk:
        """Create a document chunk.

        Args:
            content: Chunk content.
            repository: Repository name in the format 'owner/name'.
            file_path: Path to the file.
            start_line: Start line number.
            end_line: End line number.
            chunk_type: Type of chunk (code, markdown, text).
            **metadata: Additional metadata.

        Returns:
            Document chunk.
        """
        # Final safety check to make sure content isn't too large
        token_estimate = self._estimate_tokens(content)
        if token_estimate > self.max_tokens:
            print(f"Warning: Truncating oversized chunk for {file_path} ({token_estimate} tokens > {self.max_tokens})")
            # Simple truncation - not ideal but prevents crashes
            lines = content.split('\n')
            truncated_lines = lines[:int(len(lines) * self.max_tokens / token_estimate)]
            content = '\n'.join(truncated_lines)
        
        # Generate a stable ID for the chunk
        chunk_id = str(uuid.uuid5(
            uuid.NAMESPACE_URL, 
            f"{repository}/{file_path}:{start_line}-{end_line}"
        ))
        
        # Create metadata
        chunk_metadata = {
            "file_path": file_path,
            "chunk_type": chunk_type,
            "start_line": start_line,
            "end_line": end_line,
            **metadata,
        }
        
        return DocumentChunk(
            id=chunk_id,
            repository=repository,
            content=content,
            metadata=chunk_metadata,
        )


class RepositoryChunker:
    """Chunks all text files in a repository."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ) -> None:
        """Initialize the repository chunker.

        Args:
            chunk_size: Maximum number of tokens per chunk.
            chunk_overlap: Number of tokens to overlap between chunks.
        """
        self.text_chunker = TextChunker(chunk_size, chunk_overlap)

    def chunk_repository(
        self, repository: str, directory: Path
    ) -> Iterator[DocumentChunk]:
        """Chunk all text files in a repository.

        Args:
            repository: Repository name in the format 'owner/name'.
            directory: Directory containing the repository contents.

        Yields:
            Document chunks.
        """
        from repo_search.github.repository import GitHubRepositoryFetcher

        # Get all text files in the repository
        repo_fetcher = GitHubRepositoryFetcher()
        for file_path in repo_fetcher.get_text_files(directory):
            try:
                # Get the relative path within the repository
                relative_path = file_path.relative_to(directory)
                
                # Try to chunk the file
                try:
                    chunks = self.text_chunker.chunk_file(file_path, repository)
                    
                    for chunk in chunks:
                        yield chunk
                except UnicodeDecodeError as e:
                    print(f"Skipping file with unsupported encoding: {file_path}")
                    continue
                except Exception as e:
                    print(f"Error chunking file {file_path}: {e}")
                    continue
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue
