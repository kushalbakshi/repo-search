"""OpenAI embedding functionality for RepoSearch."""

from typing import Dict, List, Optional, Union

import openai
from tqdm import tqdm

from repo_search.config import config
from repo_search.models import DocumentChunk


class OpenAIEmbedder:
    """Generates embeddings using OpenAI."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """Initialize the OpenAI embedder.

        Args:
            api_key: OpenAI API key. If None, will use the key from config.
            model_name: Name of the OpenAI embedding model to use. If None, will use
                the model from config.
            batch_size: Number of texts to embed in a single batch. If None, will use
                the batch size from config.
        """
        self.api_key = api_key or config.openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        # Initialize the OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        self.model_name = model_name or config.embedding_model
        self.batch_size = batch_size or config.embedding_batch_size

    def embed_text(self, text: str) -> List[float]:
        """Generate an embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        return response.data[0].embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        embeddings = []
        
        # Process texts in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            
            # Embed the batch
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            
            # Extract embeddings from the response
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        
        return embeddings

    def embed_chunks(
        self, chunks: List[DocumentChunk], show_progress: bool = True
    ) -> List[DocumentChunk]:
        """Generate embeddings for document chunks.

        Args:
            chunks: List of document chunks to embed.
            show_progress: Whether to show a progress bar.

        Returns:
            List of document chunks with embeddings.
        """
        # Get the texts to embed
        texts = [chunk.content for chunk in chunks]
        
        # Create batches for processing
        batches = [texts[i:i+self.batch_size] for i in range(0, len(texts), self.batch_size)]
        batch_chunks = [chunks[i:i+self.batch_size] for i in range(0, len(chunks), self.batch_size)]
        
        # Create a progress iterator if requested
        if show_progress:
            progress_iter = tqdm(range(len(batches)), desc="Embedding chunks")
        else:
            progress_iter = range(len(batches))
        
        # Process chunks in batches
        embedded_chunks = []
        for i in progress_iter:
            # Embed the batch
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batches[i],
            )
            
            # Extract embeddings and assign to chunks
            for j, embedding_data in enumerate(response.data):
                chunk = batch_chunks[i][j].model_copy()
                chunk.embedding = embedding_data.embedding
                embedded_chunks.append(chunk)
        
        return embedded_chunks
