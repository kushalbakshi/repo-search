from repo_search.api.client import RepoSearchClient
from repo_search.config import config

# Print configuration
print("Using OpenAI embedding model:", config.embedding_model)
print("Score threshold:", config.score_threshold)
print("Max results:", config.max_results)

client = RepoSearchClient()

# List all repositories
repos = client.list_repositories()
print(f"Found {len(repos)} repositories:")
for repo in repos:
    print(f"  - {repo.full_name} ({repo.num_chunks} chunks)")

# Perform search
query = "scatter plot examples"
print(f"\nSearching for: '{query}'")
results = client.semantic_search(query)
print(f"Found {len(results)} results")

# Print top results
for i, result in enumerate(results[:5]):
    print(f"\nResult {i+1}, Score: {result.score:.4f}")
    print(f"Repository: {result.chunk.repository}")
    print(f"File: {result.chunk.metadata.get('file_path', 'N/A')}")
    print(f"Lines: {result.chunk.metadata.get('start_line', 'N/A')}-{result.chunk.metadata.get('end_line', 'N/A')}")
    print(f"Content: {result.chunk.content[:200]}...")
