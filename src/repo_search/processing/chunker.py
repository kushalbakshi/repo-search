"""Text chunking functionality for RepoSearch."""

import re
import uuid
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

from repo_search.config import config
from repo_search.models import DocumentChunk


class TextChunker:
    """Chunks text content into semantically meaningful segments."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        max_tokens: int = 7000,  # Set below OpenAI's 8192 limit to be safe
    ) -> None:
        """Initialize the text chunker.

        Args:
            chunk_size: Maximum number of tokens per chunk.
            chunk_overlap: Number of tokens to overlap between chunks.
            max_tokens: Maximum number of tokens allowed in a single chunk.
        """
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap
        self.max_tokens = max_tokens
        
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
        """Chunk a file into semantically meaningful segments.

        Args:
            file_path: Path to the file.
            repository: Repository name in the format 'owner/name'.
            file_content: File content. If None, will read from file_path.

        Returns:
            List of document chunks.
        """
        if file_content is None:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()
            except UnicodeDecodeError:
                # Try with a different encoding
                try:
                    with open(file_path, "r", encoding="latin-1") as f:
                        file_content = f.read()
                except UnicodeDecodeError:
                    print(f"Skipping file with unsupported encoding: {file_path}")
                    return []
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                    return []

        # Determine the chunking strategy based on the file type
        ext = file_path.suffix.lower()
        relative_path = file_path.name
        
        # For programming files, use semantic chunking
        if self._is_code_file(ext):
            return self._chunk_code(file_content, repository, str(relative_path))
        
        # For markdown and documentation files, chunk by headers
        elif ext in [".md", ".rst", ".txt", ".html", ".htm"]:
            return self._chunk_markdown(file_content, repository, str(relative_path))
        
        # For other text files, use fixed-size chunking
        else:
            return self._chunk_text(file_content, repository, str(relative_path))

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

    def _chunk_code(
        self, content: str, repository: str, file_path: str
    ) -> List[DocumentChunk]:
        """Chunk code into semantically meaningful segments.

        We'll try to chunk by functions, classes, methods, etc. For simplicity, we'll
        use a regex-based approach, which won't be perfect but should work reasonably
        well for most languages.

        Args:
            content: Code content.
            repository: Repository name in the format 'owner/name'.
            file_path: Path to the file.

        Returns:
            List of document chunks.
        """
        chunks = []
        lines = content.split("\n")
        
        # Try to identify function/class/method definitions
        # This is a simplified approach and won't work perfectly for all languages
        patterns = [
            # Python/JavaScript/Java/C# function/method
            r"^\s*(def|function|public|private|protected|async|static|class)\s+\w+.*$",
            # C/C++ function
            r"^\s*[\w\*]+\s+[\w\*]+\s*\(.*\).*$",
            # Variable declarations in various languages
            r"^\s*(var|let|const|public|private|protected|static)\s+\w+.*$",
        ]
        
        pattern = "|".join(f"({p})" for p in patterns)
        
        current_section = []
        current_section_start = 0
        
        for i, line in enumerate(lines):
            # Check if this line looks like a definition
            if re.match(pattern, line):
                # If we have accumulated lines, create a chunk
                if current_section:
                    chunk_content = "\n".join(current_section)
                    chunks.append(self._create_chunk(
                        chunk_content, 
                        repository, 
                        file_path,
                        start_line=current_section_start,
                        end_line=i-1,
                        chunk_type="code",
                    ))
                
                current_section = [line]
                current_section_start = i
            else:
                current_section.append(line)
                
                # Check if we need to create a chunk based on token count or line count
                chunk_content = "\n".join(current_section)
                token_estimate = self._estimate_tokens(chunk_content)
                
                if token_estimate > self.max_tokens or len(current_section) > self.chunk_size // 10:
                    chunks.append(self._create_chunk(
                        chunk_content, 
                        repository, 
                        file_path,
                        start_line=current_section_start,
                        end_line=current_section_start + len(current_section) - 1,
                        chunk_type="code",
                    ))
                    
                    # Keep some context for the next chunk
                    overlap = min(len(current_section), self.chunk_overlap // 10)
                    current_section = current_section[-overlap:]
                    current_section_start = current_section_start + len(current_section) - overlap
        
        # Add any remaining content
        if current_section:
            chunk_content = "\n".join(current_section)
            chunks.append(self._create_chunk(
                chunk_content, 
                repository, 
                file_path,
                start_line=current_section_start,
                end_line=current_section_start + len(current_section) - 1,
                chunk_type="code",
            ))
        
        return chunks

    def _chunk_markdown(
        self, content: str, repository: str, file_path: str
    ) -> List[DocumentChunk]:
        """Chunk markdown content by headers.

        Args:
            content: Markdown content.
            repository: Repository name in the format 'owner/name'.
            file_path: Path to the file.

        Returns:
            List of document chunks.
        """
        chunks = []
        lines = content.split("\n")
        
        # Pattern to match markdown headers
        header_pattern = r"^(#+)\s+(.*)$"
        
        current_section = []
        current_section_start = 0
        current_header = None
        
        for i, line in enumerate(lines):
            # Check if this line is a header
            header_match = re.match(header_pattern, line)
            
            if header_match:
                # If we have accumulated lines, create a chunk
                if current_section:
                    chunk_content = "\n".join(current_section)
                    chunks.append(self._create_chunk(
                        chunk_content, 
                        repository, 
                        file_path,
                        start_line=current_section_start,
                        end_line=i-1,
                        chunk_type="markdown",
                        header=current_header,
                    ))
                
                # Start a new section with this header
                current_header = header_match.group(2)
                current_section = [line]
                current_section_start = i
            else:
                current_section.append(line)
                
                # Check token count or line count
                chunk_content = "\n".join(current_section)
                token_estimate = self._estimate_tokens(chunk_content)
                
                if token_estimate > self.max_tokens or len(current_section) > self.chunk_size // 10:
                    chunks.append(self._create_chunk(
                        chunk_content, 
                        repository, 
                        file_path,
                        start_line=current_section_start,
                        end_line=current_section_start + len(current_section) - 1,
                        chunk_type="markdown",
                        header=current_header,
                    ))
                    
                    # Keep some context for the next chunk
                    overlap = min(len(current_section), self.chunk_overlap // 10)
                    current_section = current_section[-overlap:]
                    current_section_start = current_section_start + len(current_section) - overlap
        
        # Add any remaining content
        if current_section:
            chunk_content = "\n".join(current_section)
            chunks.append(self._create_chunk(
                chunk_content, 
                repository, 
                file_path,
                start_line=current_section_start,
                end_line=current_section_start + len(current_section) - 1,
                chunk_type="markdown",
                header=current_header,
            ))
        
        return chunks

    def _chunk_text(
        self, content: str, repository: str, file_path: str
    ) -> List[DocumentChunk]:
        """Chunk text into fixed-size segments with overlap.

        Args:
            content: Text content.
            repository: Repository name in the format 'owner/name'.
            file_path: Path to the file.

        Returns:
            List of document chunks.
        """
        chunks = []
        lines = content.split("\n")
        
        current_section = []
        current_section_start = 0
        
        for i, line in enumerate(lines):
            current_section.append(line)
            
            # Check if we need to create a chunk based on token count or line count
            chunk_content = "\n".join(current_section)
            token_estimate = self._estimate_tokens(chunk_content)
            
            if token_estimate > self.max_tokens or len(current_section) >= self.chunk_size // 10:
                chunks.append(self._create_chunk(
                    chunk_content, 
                    repository, 
                    file_path,
                    start_line=current_section_start,
                    end_line=current_section_start + len(current_section) - 1,
                    chunk_type="text",
                ))
                
                # Keep some context for the next chunk
                overlap = min(len(current_section), self.chunk_overlap // 10)
                current_section = current_section[-overlap:]
                current_section_start = current_section_start + len(current_section) - overlap
        
        # Add any remaining content
        if current_section:
            chunk_content = "\n".join(current_section)
            chunks.append(self._create_chunk(
                chunk_content, 
                repository, 
                file_path,
                start_line=current_section_start,
                end_line=current_section_start + len(current_section) - 1,
                chunk_type="text",
            ))
        
        return chunks

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
