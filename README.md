# RepoSearch

A tool for indexing GitHub repositories for semantic search via OpenAI embeddings.

## Features

- Index GitHub repositories with semantic search capabilities
- Filter out binary files to focus on text-only content
- Token-aware chunking and batching to prevent OpenAI API errors
- Automatic handling of oversized content with proper truncation
- Smart batching to optimize API usage

## Installation

```bash
pip install -e .
```

## Environment Setup

Create a `.env` file with:

```
OPENAI_API_KEY=your-openai-api-key
GITHUB_TOKEN=your-github-token  # Optional, allows higher rate limits
```

## Usage

### Command-line Interface

```bash
# Basic usage
python -m improved_repo_search_test.py owner/repository

# With search queries
python -m improved_repo_search_test.py owner/repository --query "What is the purpose of this class?" "How does authentication work?"

# With customized token limits
python -m improved_repo_search_test.py owner/repository --max-tokens-per-chunk 1500 --max-tokens-per-batch 6000

# With customized chunking parameters
python -m improved_repo_search_test.py owner/repository --chunk-size 300 --chunk-overlap 50 --batch-size 8
```

### API Usage

```python
from repo_search.api.client import RepoSearchClient

# Create a client with default token limits (1000 per chunk, 4000 per batch)
client = RepoSearchClient()

# Or customize token limits
client = RepoSearchClient(
    max_tokens_per_chunk=1500,  # Maximum tokens per text chunk
    max_tokens_per_batch=6000,  # Maximum tokens per API batch
)

# Index a repository
repo_info = client.index_repository("owner/repository")

# Search the repository
results = client.semantic_search(
    "How does authentication work?",
    repository="owner/repository",
    limit=3
)

# Display results
for i, result in enumerate(results, 1):
    print(f"Result {i} (Score: {result.score:.4f}):")
    print(f"Source: {result.source}")
    print(f"Content: {result.content[:200]}...")
```

## Token Limits Explained

The RepoSearch tool now intelligently handles OpenAI's token limits:

- `max_tokens_per_chunk` (default: 1000): Maximum tokens for any single chunk of text. Chunks exceeding this limit are truncated. This prevents "token limit exceeded" errors from OpenAI's API.

- `max_tokens_per_batch` (default: 4000): Maximum combined tokens for a batch of chunks sent to the API. This optimizes API calls while staying within OpenAI's limits.

These defaults provide a good balance of context preservation and API efficiency, but can be adjusted for your specific needs.
