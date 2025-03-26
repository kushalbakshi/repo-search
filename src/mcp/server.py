#!/usr/bin/env python3
"""MCP server for DocSearch."""

import os
import sys
from typing import Dict, List, Optional, Union

from mcp import (
    CallToolRequestSchema,
    Content,
    ErrorCode,
    ListResourcesRequestSchema,
    ListToolsRequestSchema,
    McpError,
    Server,
    StdioServerTransport,
)

from doc_search.api.client import DocSearchClient
from doc_search.config import config
from doc_search.models import RepositoryInfo, SearchResult


class DocSearchMcpServer:
    """MCP server for DocSearch."""

    def __init__(self) -> None:
        """Initialize the DocSearch MCP server."""
        # Initialize DocSearch client
        self.client = DocSearchClient()

        # Initialize MCP server
        self.server = Server(
            {
                "name": "doc-search",
                "version": "0.1.0",
            },
            {
                "capabilities": {
                    "resources": {},
                    "tools": {},
                },
            },
        )

        # Set up request handlers
        self._setup_handlers()

        # Error handling
        self.server.onerror = lambda error: print(f"[MCP Error] {error}", file=sys.stderr)

    def _setup_handlers(self) -> None:
        """Set up request handlers for the MCP server."""
        # Tool handlers
        self.server.setRequestHandler(ListToolsRequestSchema, self._handle_list_tools)
        self.server.setRequestHandler(CallToolRequestSchema, self._handle_call_tool)

        # Resource handlers
        self.server.setRequestHandler(ListResourcesRequestSchema, self._handle_list_resources)

    async def _handle_list_tools(self, request) -> Dict:
        """Handle ListToolsRequest.

        Args:
            request: Request object.

        Returns:
            Response object.
        """
        return {
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
        }

    async def _handle_call_tool(self, request) -> Dict:
        """Handle CallToolRequest.

        Args:
            request: Request object.

        Returns:
            Response object.

        Raises:
            McpError: If the tool name is unknown or if there's an error.
        """
        tool_name = request.params.name
        args = request.params.arguments

        try:
            if tool_name == "index_repository":
                return await self._handle_index_repository(args)
            elif tool_name == "semantic_search":
                return await self._handle_semantic_search(args)
            elif tool_name == "get_document":
                return await self._handle_get_document(args)
            elif tool_name == "list_indexed_repositories":
                return await self._handle_list_indexed_repositories(args)
            elif tool_name == "delete_repository":
                return await self._handle_delete_repository(args)
            else:
                raise McpError(ErrorCode.MethodNotFound, f"Unknown tool: {tool_name}")
        except Exception as e:
            if isinstance(e, McpError):
                raise e
            else:
                raise McpError(ErrorCode.InternalError, str(e))

    async def _handle_list_resources(self, request) -> Dict:
        """Handle ListResourcesRequest.

        Args:
            request: Request object.

        Returns:
            Response object.
        """
        # For this server, we don't expose any resources
        return {"resources": []}

    async def _handle_index_repository(self, args: Dict) -> Dict:
        """Handle index_repository tool.

        Args:
            args: Tool arguments.

        Returns:
            Tool response.
        """
        try:
            repository = args.get("repository")
            if not repository:
                return {
                    "content": [
                        {"type": "text", "text": "Repository name is required."}
                    ],
                    "isError": True,
                }

            repo_info = self.client.index_repository(repository)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Successfully indexed repository {repository}.\n"
                        f"- URL: {repo_info.url}\n"
                        f"- Files: {repo_info.num_files}\n"
                        f"- Chunks: {repo_info.num_chunks}\n"
                        f"- Last indexed: {repo_info.last_indexed}",
                    }
                ]
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error indexing repository: {e}"}],
                "isError": True,
            }

    async def _handle_semantic_search(self, args: Dict) -> Dict:
        """Handle semantic_search tool.

        Args:
            args: Tool arguments.

        Returns:
            Tool response.
        """
        try:
            query = args.get("query")
            if not query:
                return {
                    "content": [{"type": "text", "text": "Query is required."}],
                    "isError": True,
                }

            repository = args.get("repository")
            limit = args.get("limit")
            score_threshold = args.get("score_threshold")

            results = self.client.semantic_search(
                query, repository, limit, score_threshold
            )

            if not results:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": "No results found for the query.",
                        }
                    ]
                }

            result_texts = []
            for i, result in enumerate(results, 1):
                result_texts.append(
                    f"Result {i} (Score: {result.score:.2f}):\n"
                    f"Source: {result.source}\n"
                    f"Content:\n{result.content}\n"
                )

            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Query: {query}\n\n" + "\n".join(result_texts),
                    }
                ]
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error searching repositories: {e}"}],
                "isError": True,
            }

    async def _handle_get_document(self, args: Dict) -> Dict:
        """Handle get_document tool.

        Args:
            args: Tool arguments.

        Returns:
            Tool response.
        """
        try:
            chunk_id = args.get("chunk_id")
            if not chunk_id:
                return {
                    "content": [{"type": "text", "text": "Chunk ID is required."}],
                    "isError": True,
                }

            # We don't have a direct API for getting a chunk by ID, so we need to implement it
            chunk = self.client.engine.db.get_chunk(chunk_id)
            if not chunk:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Document chunk with ID {chunk_id} not found.",
                        }
                    ],
                    "isError": True,
                }

            metadata_str = "\n".join(
                f"- {k}: {v}" for k, v in chunk.metadata.items() if v is not None
            )

            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Document Chunk {chunk_id}:\n"
                        f"Repository: {chunk.repository}\n"
                        f"Metadata:\n{metadata_str}\n\n"
                        f"Content:\n{chunk.content}",
                    }
                ]
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error retrieving document: {e}"}],
                "isError": True,
            }

    async def _handle_list_indexed_repositories(self, args: Dict) -> Dict:
        """Handle list_indexed_repositories tool.

        Args:
            args: Tool arguments.

        Returns:
            Tool response.
        """
        try:
            repositories = self.client.list_repositories()
            if not repositories:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": "No repositories are currently indexed.",
                        }
                    ]
                }

            repo_texts = []
            for repo in repositories:
                repo_texts.append(
                    f"- {repo.full_name}\n"
                    f"  URL: {repo.url}\n"
                    f"  Files: {repo.num_files}\n"
                    f"  Chunks: {repo.num_chunks}\n"
                    f"  Last indexed: {repo.last_indexed}"
                )

            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Indexed Repositories:\n\n" + "\n\n".join(repo_texts),
                    }
                ]
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error listing repositories: {e}"}],
                "isError": True,
            }

    async def _handle_delete_repository(self, args: Dict) -> Dict:
        """Handle delete_repository tool.

        Args:
            args: Tool arguments.

        Returns:
            Tool response.
        """
        try:
            repository = args.get("repository")
            if not repository:
                return {
                    "content": [{"type": "text", "text": "Repository name is required."}],
                    "isError": True,
                }

            success = self.client.delete_repository(repository)
            if success:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Successfully deleted repository {repository} from the index.",
                        }
                    ]
                }
            else:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Repository {repository} not found in the index.",
                        }
                    ],
                    "isError": True,
                }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error deleting repository: {e}"}],
                "isError": True,
            }

    async def run(self) -> None:
        """Run the MCP server."""
        transport = StdioServerTransport()
        await self.server.connect(transport)
        print("DocSearch MCP server running on stdio", file=sys.stderr)


def main() -> None:
    """Run the DocSearch MCP server."""
    import asyncio

    server = DocSearchMcpServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
