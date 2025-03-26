"""ChromaDB vector database implementation for RepoSearch."""

import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from repo_search.config import config
from repo_search.database.base import VectorDatabase
from repo_search.embedding.openai import OpenAIEmbedder
from repo_search.models import DocumentChunk, RepositoryInfo, SearchResult


class ChromaVectorDatabase(VectorDatabase):
    """ChromaDB implementation of the vector database interface."""

    def __init__(
        self,
        db_path: Path,
        embedder: Optional[OpenAIEmbedder] = None,
    ) -> None:
        """Initialize the ChromaDB vector database.

        Args:
            db_path: Path to the database directory.
            embedder: Embedder to use for generating embeddings. If None, embeddings
                must be pre-computed.
        """
        self.db_path = db_path or config.db_path
        self.embedder = embedder

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )

        # Create collections if they don't exist
        self.chunks_collection = self._get_or_create_collection("chunks")
        self.repositories_collection = self._get_or_create_collection("repositories")

    def _get_or_create_collection(self, name: str) -> chromadb.Collection:
        """Get or create a collection.

        Args:
            name: Name of the collection.

        Returns:
            ChromaDB collection.
        """
        # HNSW index parameters
        hnsw_params = {
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 200,  # Default is 100
            "hnsw:search_ef": 100,        # Default is 10
            "hnsw:M": 16                  # Default is 8
        }
        
        try:
            return self.client.get_or_create_collection(
                name=name,
                metadata=hnsw_params
            )
        except AttributeError:
            # Fallback if get_or_create_collection is not available
            try:
                return self.client.get_collection(name=name)
            except Exception as e:
                # Collection doesn't exist, create it
                if "does not exist" in str(e):
                    return self.client.create_collection(
                        name=name,
                        metadata=hnsw_params
                    )
                raise e

    def store_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Store document chunks in the database.

        Args:
            chunks: List of document chunks to store.
        """
        if not chunks:
            return

        # Ensure chunks have embeddings
        if any(chunk.embedding is None for chunk in chunks):
            if self.embedder is None:
                raise ValueError(
                    "Cannot store chunks without embeddings. "
                    "Either pre-compute embeddings or provide an embedder."
                )
            chunks = self.embedder.embed_chunks(chunks)

        # Check for duplicate IDs and make them unique
        seen_ids = set()
        unique_chunks = []
        
        for chunk in chunks:
            if chunk.id in seen_ids:
                print(f"Warning: Duplicate chunk ID detected: {chunk.id}. Skipping chunk.")
                continue
            seen_ids.add(chunk.id)
            unique_chunks.append(chunk)
        
        if len(unique_chunks) < len(chunks):
            print(f"Removed {len(chunks) - len(unique_chunks)} duplicate chunks")
            chunks = unique_chunks
        
        if not chunks:
            print("No unique chunks to store after deduplication.")
            return
        
        # Prepare data for ChromaDB
        ids = [chunk.id for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]
        metadatas = [
            {
                "repository": chunk.repository,
                "file_path": chunk.metadata.get("file_path", ""),
                "chunk_type": chunk.metadata.get("chunk_type", "text"),
                "start_line": str(chunk.metadata.get("start_line", "")),
                "end_line": str(chunk.metadata.get("end_line", "")),
                # Convert all metadata to strings for ChromaDB compatibility
                **{k: str(v) if v is not None else "" 
                   for k, v in chunk.metadata.items() 
                   if k not in ["file_path", "chunk_type", "start_line", "end_line"]},
            }
            for chunk in chunks
        ]
        documents = [chunk.content for chunk in chunks]

        # Add to ChromaDB in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            try:
                self.chunks_collection.add(
                    ids=ids[i:i+batch_size],
                    embeddings=embeddings[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size],
                    documents=documents[i:i+batch_size],
                )
            except Exception as e:
                if "duplicate" in str(e).lower():
                    print(f"Warning: Duplicate IDs detected in batch. Attempting to add individually.")
                    # Process one by one to skip only the problematic documents
                    for j in range(i, min(i + batch_size, len(chunks))):
                        try:
                            self.chunks_collection.add(
                                ids=[ids[j]],
                                embeddings=[embeddings[j]],
                                metadatas=[metadatas[j]],
                                documents=[documents[j]],
                            )
                        except Exception as inner_e:
                            if "duplicate" in str(inner_e).lower():
                                print(f"Skipping duplicate ID: {ids[j]}")
                            else:
                                raise inner_e
                else:
                    raise e

        # Update repository chunk count
        repositories = set(chunk.repository for chunk in chunks)
        for repo in repositories:
            repo_info = self.get_repository(repo)
            if repo_info:
                repo_info.num_chunks += sum(1 for chunk in chunks if chunk.repository == repo)
                repo_info.last_indexed = datetime.datetime.now()
                self.add_repository(repo_info)

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
        # Generate embedding for the query
        if self.embedder is None:
            raise ValueError("Cannot search without an embedder.")

        query_embedding = self.embedder.embed_text(query)
        
        # Prepare filter if repository is specified
        where_filter = None
        if repository:
            where_filter = {"repository": repository}
        
        # Search in ChromaDB
        results = self.chunks_collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_filter,
        )
        
        # Process results
        search_results = []
        if results and len(results["ids"]) > 0:
            for i, chunk_id in enumerate(results["ids"][0]):
                # Skip results below the score threshold
                if score_threshold > 0:
                    score = results["distances"][0][i]
                    # ChromaDB returns distances, not similarities, so we convert
                    # (assuming cosine distance where 0 is most similar)
                    similarity = 1.0 - score
                    if similarity < score_threshold:
                        continue
                
                # Get the document chunk
                chunk = self.get_chunk(chunk_id)
                if chunk:
                    # Add to results
                    search_results.append(
                        SearchResult(
                            chunk=chunk,
                            score=1.0 - results["distances"][0][i],  # Convert to similarity
                        )
                    )
        
        return search_results

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get a document chunk by ID.

        Args:
            chunk_id: ID of the chunk to retrieve.

        Returns:
            Document chunk, or None if not found.
        """
        result = self.chunks_collection.get(ids=[chunk_id])
        
        if not result["ids"]:
            return None
        
        # Build the document chunk from the result
        chunk_index = 0
        chunk_id = result["ids"][chunk_index]
        content = result["documents"][chunk_index]
        metadata = result["metadatas"][chunk_index]
        
        # Convert string values to appropriate types
        processed_metadata = {}
        for k, v in metadata.items():
            if k in ["start_line", "end_line"]:
                if v.isdigit():
                    processed_metadata[k] = int(v)
                else:
                    processed_metadata[k] = None
            else:
                processed_metadata[k] = v
        
        # Create the document chunk
        return DocumentChunk(
            id=chunk_id,
            repository=metadata["repository"],
            content=content,
            metadata=processed_metadata,
        )

    def list_repositories(self) -> List[RepositoryInfo]:
        """List all repositories in the database.

        Returns:
            List of repository information.
        """
        result = self.repositories_collection.get()
        
        if not result["ids"]:
            return []
        
        # Build repository info objects
        repositories = []
        for i, repo_id in enumerate(result["ids"]):
            metadata = result["metadatas"][i]
            
            # Parse the serialized data
            try:
                repo_data = json.loads(result["documents"][i])
                last_indexed = None
                if "last_indexed" in repo_data and repo_data["last_indexed"]:
                    last_indexed = datetime.datetime.fromisoformat(repo_data["last_indexed"])
                
                repositories.append(
                    RepositoryInfo(
                        owner=metadata["owner"],
                        name=metadata["name"],
                        url=repo_data["url"],
                        last_indexed=last_indexed,
                        num_files=repo_data.get("num_files", 0),
                        num_chunks=repo_data.get("num_chunks", 0),
                        commit_hash=repo_data.get("commit_hash"),
                        download_successful=repo_data.get("download_successful", False),
                        chunking_successful=repo_data.get("chunking_successful", False),
                        embedding_successful=repo_data.get("embedding_successful", False),
                    )
                )
            except Exception as e:
                print(f"Error parsing repository data: {e}")
        
        return repositories

    def add_repository(self, repository_info: RepositoryInfo) -> None:
        """Add a repository to the database.

        Args:
            repository_info: Repository information.
        """
        repo_id = repository_info.full_name
        
        # Prepare serialized data
        repository_data = {
            "url": repository_info.url,
            "num_files": repository_info.num_files,
            "num_chunks": repository_info.num_chunks,
            "last_indexed": repository_info.last_indexed.isoformat() if repository_info.last_indexed else None,
            "commit_hash": repository_info.commit_hash,
            "download_successful": repository_info.download_successful,
            "chunking_successful": repository_info.chunking_successful,
            "embedding_successful": repository_info.embedding_successful,
        }
        
        # Check if the repository already exists
        existing = self.repositories_collection.get(
            ids=[repo_id],
            include=["metadatas"],
        )
        
        if existing["ids"]:
            # Update existing repository
            self.repositories_collection.update(
                ids=[repo_id],
                metadatas=[{
                    "owner": repository_info.owner,
                    "name": repository_info.name,
                    "full_name": repository_info.full_name,
                }],
                documents=[json.dumps(repository_data)],
            )
        else:
            # Add new repository
            self.repositories_collection.add(
                ids=[repo_id],
                metadatas=[{
                    "owner": repository_info.owner,
                    "name": repository_info.name,
                    "full_name": repository_info.full_name,
                }],
                documents=[json.dumps(repository_data)],
            )

    def get_repository(self, repository_name: str) -> Optional[RepositoryInfo]:
        """Get repository information.

        Args:
            repository_name: Repository name in the format 'owner/name'.

        Returns:
            Repository information, or None if not found.
        """
        result = self.repositories_collection.get(ids=[repository_name])
        
        if not result["ids"]:
            return None
        
        # Build the repository info from the result
        metadata = result["metadatas"][0]
        
        # Parse the serialized data
        try:
            repo_data = json.loads(result["documents"][0])
            last_indexed = None
            if "last_indexed" in repo_data and repo_data["last_indexed"]:
                last_indexed = datetime.datetime.fromisoformat(repo_data["last_indexed"])
            
            return RepositoryInfo(
                owner=metadata["owner"],
                name=metadata["name"],
                url=repo_data["url"],
                last_indexed=last_indexed,
                num_files=repo_data.get("num_files", 0),
                num_chunks=repo_data.get("num_chunks", 0),
                commit_hash=repo_data.get("commit_hash"),
                download_successful=repo_data.get("download_successful", False),
                chunking_successful=repo_data.get("chunking_successful", False),
                embedding_successful=repo_data.get("embedding_successful", False),
            )
        except Exception as e:
            print(f"Error parsing repository data: {e}")
            return None

    def delete_repository(self, repository_name: str) -> bool:
        """Delete a repository and all its chunks from the database.

        Args:
            repository_name: Repository name in the format 'owner/name'.

        Returns:
            True if the repository was deleted, False if it was not found.
        """
        # Check if the repository exists
        repo_info = self.get_repository(repository_name)
        if not repo_info:
            return False
        
        # Delete all chunks for the repository
        self.chunks_collection.delete(where={"repository": repository_name})
        
        # Delete the repository info
        self.repositories_collection.delete(ids=[repository_name])
        
        return True

    def clear(self) -> None:
        """Clear all data from the database."""
        # Delete and recreate collections
        self.client.delete_collection("chunks")
        self.client.delete_collection("repositories")
        self.chunks_collection = self.client.create_collection("chunks")
        self.repositories_collection = self.client.create_collection("repositories")
