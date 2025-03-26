from repo_search.api.client import RepoSearchClient

client = RepoSearchClient()
client.delete_repository("bendichter/brokenaxes")
client.index_repository("bendichter/brokenaxes")
results = client.semantic_search("scatter")
