"""OpenAI embedding functionality for RepoSearch."""

import time
import tiktoken
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
        max_tokens_per_batch: int = 4000,
        max_tokens_per_chunk: int = 1000,
    ) -> None:
        """Initialize the OpenAI embedder.

        Args:
            api_key: OpenAI API key. If None, will use the key from config.
            model_name: Name of the OpenAI embedding model to use. If None, will use
                the model from config.
            batch_size: Number of texts to embed in a single batch. If None, will use
                the batch size from config.
            max_tokens_per_batch: Maximum total tokens allowed in a single batch.
            max_tokens_per_chunk: Maximum tokens allowed in a single chunk.
        """
        self.api_key = api_key or config.openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        # Initialize the OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        self.model_name = model_name or config.embedding_model
        self.batch_size = batch_size or config.embedding_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.max_tokens_per_chunk = max_tokens_per_chunk
        
        # Initialize tokenizer for counting tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI embedding models use this
    
    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string using tiktoken.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        return len(self.tokenizer.encode(text))
    
    def _truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """Truncate text to stay within token limit.
        
        Args:
            text: The text to truncate
            max_tokens: Maximum number of tokens allowed
            
        Returns:
            Truncated text
        """
        if not text:
            return ""
            
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
            
        # Truncate to max_tokens
        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens)

    def embed_text(self, text: str) -> List[float]:
        """Generate an embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        # Check for empty text
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
            
        # Truncate text if it exceeds token limit
        token_count = self._count_tokens(text)
        if token_count > self.max_tokens_per_chunk:
            text = self._truncate_to_token_limit(text, self.max_tokens_per_chunk)
        
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
        
        # Filter out empty texts
        filtered_texts = []
        for text in texts:
            if text and text.strip():
                # Truncate text if it exceeds token limit
                token_count = self._count_tokens(text)
                if token_count > self.max_tokens_per_chunk:
                    text = self._truncate_to_token_limit(text, self.max_tokens_per_chunk)
                filtered_texts.append(text)
            else:
                print("Warning: Skipping empty text in embed_texts")
        
        if not filtered_texts:
            return []
        
        # Create token-aware batches
        batches = []
        current_batch = []
        current_batch_tokens = 0
        
        for text in filtered_texts:
            text_tokens = self._count_tokens(text)
            
            # If adding this text would exceed the batch token limit or batch size limit,
            # finalize the current batch and start a new one
            if (current_batch_tokens + text_tokens > self.max_tokens_per_batch or 
                len(current_batch) >= self.batch_size):
                if current_batch:  # Don't add empty batches
                    batches.append(current_batch)
                current_batch = [text]
                current_batch_tokens = text_tokens
            else:
                current_batch.append(text)
                current_batch_tokens += text_tokens
        
        # Add the last batch if it's not empty
        if current_batch:
            batches.append(current_batch)
        
        # Process batches
        for batch in batches:
            # Embed the batch
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            
            # Extract embeddings from the response
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            
            # Add a small delay to avoid rate limits
            time.sleep(0.1)
        
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
        # Filter out chunks with empty content
        valid_chunks = []
        filtered_chunks = []
        
        print("Processing chunks for embedding...")
        
        # Process each chunk - filter empty chunks and truncate large ones
        for chunk in chunks:
            if not chunk.content or chunk.content.strip() == "":
                print(f"Warning: Filtering out chunk with empty content (ID: {chunk.id})")
                continue
                
            # Count tokens and truncate if needed
            token_count = self._count_tokens(chunk.content)
            if token_count > self.max_tokens_per_chunk:
                print(f"Warning: Truncating chunk with {token_count} tokens to {self.max_tokens_per_chunk} tokens (ID: {chunk.id})")
                modified_chunk = chunk.model_copy()
                modified_chunk.content = self._truncate_to_token_limit(chunk.content, self.max_tokens_per_chunk)
                valid_chunks.append(modified_chunk)
            else:
                valid_chunks.append(chunk)
        
        if not valid_chunks:
            print("Warning: No valid chunks with content to embed!")
            return []
        
        # Create token-aware batches
        batches = []
        current_batch = []
        current_batch_tokens = 0
        
        for chunk in valid_chunks:
            chunk_tokens = self._count_tokens(chunk.content)
            
            # If adding this chunk would exceed the batch token limit or batch size limit,
            # finalize the current batch and start a new one
            if (current_batch_tokens + chunk_tokens > self.max_tokens_per_batch or 
                len(current_batch) >= self.batch_size):
                if current_batch:  # Don't add empty batches
                    batches.append(current_batch)
                current_batch = [chunk]
                current_batch_tokens = chunk_tokens
            else:
                current_batch.append(chunk)
                current_batch_tokens += chunk_tokens
        
        # Add the last batch if it's not empty
        if current_batch:
            batches.append(current_batch)
        
        print(f"Created {len(batches)} batches from {len(valid_chunks)} chunks")
        
        # Process each batch
        chunks_with_embeddings = []
        
        if show_progress:
            pbar = tqdm(total=len(valid_chunks), desc="Embedding chunks")
        
        for batch in batches:
            try:
                # Create input list for this batch
                batch_texts = [chunk.content for chunk in batch]
                
                # Embed the batch
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch_texts,
                )
                
                # Extract embeddings and assign to chunks
                for j, embedding_data in enumerate(response.data):
                    chunk = batch[j].model_copy()
                    chunk.embedding = embedding_data.embedding
                    chunks_with_embeddings.append(chunk)
                
                if show_progress:
                    pbar.update(len(batch))
                
                # Add a small delay to avoid rate limits
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error embedding batch: {e}")
                print(f"Batch size: {len(batch)}")
                print(f"First chunk content preview: {batch[0].content[:100]}...")
                
                # Try embedding one at a time to isolate problematic chunks
                print("Trying to embed chunks individually...")
                for chunk in batch:
                    try:
                        response = self.client.embeddings.create(
                            model=self.model_name,
                            input=chunk.content,
                        )
                        chunk_copy = chunk.model_copy()
                        chunk_copy.embedding = response.data[0].embedding
                        chunks_with_embeddings.append(chunk_copy)
                        
                        if show_progress:
                            pbar.update(1)
                        
                        # Add a small delay to avoid rate limits
                        time.sleep(0.2)
                    except Exception as inner_e:
                        print(f"Error embedding individual chunk: {inner_e}")
                        print(f"Content length: {len(chunk.content)}")
                        print(f"Content preview: {chunk.content[:100]}...")
        
        if show_progress:
            pbar.close()
            
        print(f"Successfully embedded {len(chunks_with_embeddings)} chunks")
        return chunks_with_embeddings
