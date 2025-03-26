#!/usr/bin/env python3
"""MCP server for RepoSearch using FastMCP."""

import asyncio
import json
import sys
from typing import Optional, Any

from mcp.server.fastmcp import FastMCP

from repo_search.api.client import RepoSearchClient
from repo_search.config import config
from repo_search.models import RepositoryInfo, SearchResult


# Create an MCP server
mcp = FastMCP("RepoSearch")

# Initialize RepoSearch client
client = RepoSearchClient()


# Tool definitions using decorators
@mcp.tool()
def index_repository(repository: str, force_refresh: bool = False) -> str:
    """Index a GitHub repository for semantic search.
    
    Args:
        repository: Repository name in the format 'owner/name'
        force_refresh: If True, forces re-indexing even if commit hash is unchanged
    
    Returns:
        Success message with repository information
    """
    try:
        repo_info = client.index_repository(repository, force_refresh)
        result = [
            f"Successfully indexed repository {repository}.",
            f"- URL: {repo_info.url}",
            f"- Files: {repo_info.num_files}",
            f"- Chunks: {repo_info.num_chunks}",
            f"- Last indexed: {repo_info.last_indexed}"
        ]
        
        if repo_info.commit_hash:
            result.append(f"- Commit hash: {repo_info.commit_hash}")
            
        return "\n".join(result)
    except Exception as e:
        raise Exception(f"Error indexing repository: {e}")


@mcp.tool()
def semantic_search(query: str, repository: Optional[str] = None, 
                   limit: Optional[int] = None, 
                   score_threshold: Optional[float] = None) -> str:
    """Perform semantic search over indexed repositories.
    
    Args:
        query: Query text to search for
        repository: Optional repository to search in (owner/name)
        limit: Maximum number of results to return
        score_threshold: Minimum similarity score for results
    
    Returns:
        Search results
    """
    try:
        results = client.semantic_search(query, repository, limit, score_threshold)
        
        if not results:
            return "No results found for the query."
            
        result_texts = []
        for i, result in enumerate(results, 1):
            result_texts.append(
                f"Result {i} (Score: {result.score:.2f}):\n"
                f"Source: {result.source}\n"
                f"Content:\n{result.content}\n"
            )
            
        return f"Query: {query}\n\n" + "\n".join(result_texts)
    except Exception as e:
        raise Exception(f"Error searching repositories: {e}")


@mcp.tool()
def get_document(chunk_id: str) -> str:
    """Retrieve a document chunk by ID.
    
    Args:
        chunk_id: ID of the document chunk to retrieve
    
    Returns:
        Document content and metadata
    """
    try:
        # We don't have a direct API for getting a chunk by ID, so we need to implement it
        chunk = client.engine.db.get_chunk(chunk_id)
        if not chunk:
            raise Exception(f"Document chunk with ID {chunk_id} not found")
            
        metadata_str = "\n".join(
            f"- {k}: {v}" for k, v in chunk.metadata.items() if v is not None
        )
        
        return (f"Document Chunk {chunk_id}:\n"
                f"Repository: {chunk.repository}\n"
                f"Metadata:\n{metadata_str}\n\n"
                f"Content:\n{chunk.content}")
    except Exception as e:
        raise Exception(f"Error retrieving document: {e}")


@mcp.tool()
def list_indexed_repositories() -> str:
    """List all indexed repositories.
    
    Returns:
        List of all indexed repositories with details
    """
    try:
        repositories = client.list_repositories()
        if not repositories:
            return "No repositories are currently indexed."
            
        repo_texts = []
        for repo in repositories:
            repo_texts.append(
                f"- {repo.full_name}\n"
                f"  URL: {repo.url}\n"
                f"  Files: {repo.num_files}\n"
                f"  Chunks: {repo.num_chunks}\n"
                f"  Last indexed: {repo.last_indexed}"
            )
            
        return "Indexed Repositories:\n\n" + "\n\n".join(repo_texts)
    except Exception as e:
        raise Exception(f"Error listing repositories: {e}")


@mcp.tool()
def delete_repository(repository: str) -> str:
    """Delete a repository from the index.
    
    Args:
        repository: Repository name in the format 'owner/name'
    
    Returns:
        Success message or error
    """
    try:
        success = client.delete_repository(repository)
        if success:
            return f"Successfully deleted repository {repository} from the index."
        else:
            raise Exception(f"Repository {repository} not found in the index.")
    except Exception as e:
        raise Exception(f"Error deleting repository: {e}")


@mcp.tool()
def search_repository(repository: str, query: str) -> str:
    """Search for text in a GitHub repository without using semantic search.
    
    Args:
        repository: Repository name in the format 'owner/name'
        query: Text to search for in the repository
    
    Returns:
        Search results
    """
    try:
        # First check if the repository is already indexed, if not index it
        try:
            repo_info = client.get_repository_info(repository)
        except:
            repo_info = client.index_repository(repository)
            
        import os
        import tempfile
        import subprocess
        
        # Create a temporary directory for cloning the repository
        with tempfile.TemporaryDirectory() as temp_dir:
            # Clone the repository
            subprocess.run(
                ["git", "clone", f"https://github.com/{repository}.git", temp_dir],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            # Use grep to search for the query text in the repository
            result = subprocess.run(
                ["grep", "-r", "--include=*.py", "--include=*.ipynb", "--include=*.md", 
                 "--include=*.txt", "--include=*.js", "--include=*.html", "--include=*.css",
                 "--include=*.json", "--include=*.c", "--include=*.cpp", "--include=*.h",
                 "-n", "-A", "3", "-B", "3", query, temp_dir],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            
            if result.returncode != 0 and result.returncode != 1:  # grep returns 1 if no matches
                raise Exception(f"Error searching repository: {result.stderr}")
            
            if not result.stdout.strip():
                return f"No results found for '{query}' in {repository}."
            
            # Format the results
            result_texts = []
            lines = result.stdout.strip().split("\n")
            for line in lines:
                if line.startswith(temp_dir):
                    # Extract just the file path relative to the repo
                    file_path = line[len(temp_dir)+1:]
                    result_texts.append(f"File: {file_path}")
                else:
                    # This is a match context line
                    result_texts.append(line)
            
            return f"Results for '{query}' in {repository}:\n\n" + "\n".join(result_texts)
                
    except Exception as e:
        raise Exception(f"Error searching repository: {e}")


# For testing compatibility with JSON-RPC, implement a stdio handler
async def handle_jsonrpc_request(request_str: str) -> str:
    """Handle a JSON-RPC request for testing purposes."""
    try:
        # Parse the request
        request = json.loads(request_str)
        method = request.get("method")
        request_id = request.get("id")
        
        if not method:
            return create_error_response(request_id, 32700, "Invalid request: missing method")
            
        # Handle initialization
        if method == "initialize":
            return json.dumps({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": "repo-search",
                        "version": "0.1.0",
                    },
                    "capabilities": {
                        "tools": {"list": True, "call": True},
                        "resources": {"list": True, "read": True},
                        "resourceTemplates": {"list": True},
                    },
                },
            })
            
        # Handle tool listing
        elif method == "tools/list":
            # For testing, return hardcoded tool definitions
            return json.dumps({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "index_repository",
                            "description": "Index a GitHub repository for semantic search",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "repository": {
                                        "type": "string",
                                        "description": "Repository name in the format 'owner/name'",
                                    },
                                    "force_refresh": {
                                        "type": "boolean",
                                        "description": "If True, forces re-indexing even if commit hash is unchanged",
                                        "default": "false"
                                    }
                                },
                                "required": ["repository"],
                            },
                        },
                        {
                            "name": "semantic_search",
                            "description": "Perform semantic search over indexed repositories",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "Query text to search for",
                                    },
                                    "repository": {
                                        "type": "string",
                                        "description": "Optional repository to search in (owner/name)",
                                    },
                                    "limit": {
                                        "type": "number",
                                        "description": "Maximum number of results to return",
                                    },
                                    "score_threshold": {
                                        "type": "number",
                                        "description": "Minimum similarity score for results",
                                    },
                                },
                                "required": ["query"],
                            },
                        },
                        {
                            "name": "get_document",
                            "description": "Retrieve a document chunk by ID",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "chunk_id": {
                                        "type": "string",
                                        "description": "ID of the document chunk to retrieve",
                                    }
                                },
                                "required": ["chunk_id"],
                            },
                        },
                        {
                            "name": "list_indexed_repositories",
                            "description": "List all indexed repositories",
                            "inputSchema": {
                                "type": "object",
                                "properties": {},
                            },
                        },
                        {
                            "name": "delete_repository",
                            "description": "Delete a repository from the index",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "repository": {
                                        "type": "string",
                                        "description": "Repository name in the format 'owner/name'",
                                    }
                                },
                                "required": ["repository"],
                            },
                        },
                    ]
                },
            })
            
        # Handle tool calls
        elif method == "tools/call":
            params = request.get("params", {})
            tool_name = params.get("name")
            args = params.get("arguments", {})
            
            result = None
            
            # Call the tool function based on name
            if tool_name == "index_repository":
                repository = args.get("repository")
                if not repository:
                    return create_error_response(request_id, 32602, "Repository name is required.")
                force_refresh = args.get("force_refresh", False)
                result = index_repository(repository, force_refresh)
                
            elif tool_name == "semantic_search":
                query = args.get("query")
                if not query:
                    return create_error_response(request_id, 32602, "Query is required.")
                repository = args.get("repository")
                limit = args.get("limit")
                score_threshold = args.get("score_threshold")
                result = semantic_search(query, repository, limit, score_threshold)
                
            elif tool_name == "get_document":
                chunk_id = args.get("chunk_id")
                if not chunk_id:
                    return create_error_response(request_id, 32602, "Chunk ID is required.")
                result = get_document(chunk_id)
                
            elif tool_name == "list_indexed_repositories":
                result = list_indexed_repositories()
                
            elif tool_name == "delete_repository":
                repository = args.get("repository")
                if not repository:
                    return create_error_response(request_id, 32602, "Repository name is required.")
                result = delete_repository(repository)
                
            else:
                return create_error_response(request_id, 32601, f"Unknown tool: {tool_name}")
                
            # Return successful result
            return json.dumps({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": result,
                        }
                    ]
                },
            })
            
        # Handle resource listing (empty for now)
        elif method == "resources/list":
            return json.dumps({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "resources": []
                },
            })
            
        # Unknown method
        else:
            return create_error_response(request_id, 32601, f"Method not found: {method}")
            
    except json.JSONDecodeError:
        return create_error_response(None, 32700, "Parse error")
    except Exception as e:
        return create_error_response(None, 32603, f"Internal error: {str(e)}")


def create_error_response(id_value, code: int, message: str) -> str:
    """Create a JSON-RPC error response."""
    response = {
        "jsonrpc": "2.0",
        "id": id_value,
        "error": {
            "code": code,
            "message": message,
        },
    }
    return json.dumps(response)


async def run_jsonrpc_server() -> None:
    """Run the server in JSON-RPC mode for testing."""
    print("RepoSearch MCP server starting with JSON-RPC over stdio...", file=sys.stderr)
    
    while True:
        try:
            # Read request from stdin
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            if not line:
                break
                
            # Process request
            request_str = line.strip()
            response_str = await handle_jsonrpc_request(request_str)
            
            # Write response to stdout
            sys.stdout.write(response_str + "\n")
            sys.stdout.flush()
            
        except Exception as e:
            error_response = create_error_response(None, 32603, f"Internal error: {str(e)}")
            sys.stdout.write(error_response + "\n")
            sys.stdout.flush()
            print(f"Error: {e}", file=sys.stderr)


def main() -> None:
    """Run the RepoSearch MCP server."""
    import asyncio

    # Use FastMCP to handle MCP requests
    asyncio.run(mcp.run())


if __name__ == "__main__":
    main()
