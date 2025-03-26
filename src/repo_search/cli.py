"""Command-line interface for RepoSearch."""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from repo_search.api.client import RepoSearchClient
from repo_search.config import config
from repo_search.models import RepositoryInfo, SearchResult


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="RepoSearch - Semantic search for GitHub repositories"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Index command
    index_parser = subparsers.add_parser(
        "index", help="Index a GitHub repository for semantic search"
    )
    index_parser.add_argument(
        "repository", help="Repository name in the format 'owner/name'"
    )
    index_parser.add_argument(
        "--api-key",
        help="OpenAI API key (or set OPENAI_API_KEY environment variable)",
    )
    index_parser.add_argument(
        "--github-token",
        help="GitHub token (or set GITHUB_TOKEN environment variable)",
    )
    index_parser.add_argument(
        "--data-dir", help="Data directory (or set DATA_DIR environment variable)"
    )

    # Search command
    search_parser = subparsers.add_parser(
        "search", help="Search indexed repositories"
    )
    search_parser.add_argument("query", help="Query text to search for")
    search_parser.add_argument(
        "--repository",
        help="Optional repository to search in (owner/name)",
    )
    search_parser.add_argument(
        "--limit", type=int, help="Maximum number of results to return"
    )
    search_parser.add_argument(
        "--score-threshold",
        type=float,
        help="Minimum similarity score for results",
    )
    search_parser.add_argument(
        "--api-key",
        help="OpenAI API key (or set OPENAI_API_KEY environment variable)",
    )
    search_parser.add_argument(
        "--data-dir", help="Data directory (or set DATA_DIR environment variable)"
    )

    # List repositories command
    list_parser = subparsers.add_parser(
        "list", help="List indexed repositories"
    )
    list_parser.add_argument(
        "--data-dir", help="Data directory (or set DATA_DIR environment variable)"
    )

    # Delete repository command
    delete_parser = subparsers.add_parser(
        "delete", help="Delete a repository from the index"
    )
    delete_parser.add_argument(
        "repository", help="Repository name in the format 'owner/name'"
    )
    delete_parser.add_argument(
        "--data-dir", help="Data directory (or set DATA_DIR environment variable)"
    )

    return parser.parse_args()


def index_repository(args: argparse.Namespace) -> None:
    """Index a GitHub repository.

    Args:
        args: Command-line arguments.
    """
    # Update environment variables
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.github_token:
        os.environ["GITHUB_TOKEN"] = args.github_token
    if args.data_dir:
        os.environ["DATA_DIR"] = args.data_dir

    # Create client
    client = RepoSearchClient()

    # Index repository
    try:
        repo_info = client.index_repository(args.repository)
        print(f"Successfully indexed repository {args.repository}.")
        print(f"URL: {repo_info.url}")
        print(f"Files: {repo_info.num_files}")
        print(f"Chunks: {repo_info.num_chunks}")
        print(f"Last indexed: {repo_info.last_indexed}")
    except Exception as e:
        print(f"Error indexing repository: {e}", file=sys.stderr)
        sys.exit(1)


def search_repositories(args: argparse.Namespace) -> None:
    """Search indexed repositories.

    Args:
        args: Command-line arguments.
    """
    # Update environment variables
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.data_dir:
        os.environ["DATA_DIR"] = args.data_dir

    # Create client
    client = RepoSearchClient()

    # Search repositories
    try:
        results = client.semantic_search(
            args.query, args.repository, args.limit, args.score_threshold
        )
        if not results:
            print("No results found for the query.")
            return

        print(f"Query: {args.query}")
        print()
        for i, result in enumerate(results, 1):
            print(f"Result {i} (Score: {result.score:.2f}):")
            print(f"Source: {result.source}")
            print("Content:")
            print(result.content)
            print()
    except Exception as e:
        print(f"Error searching repositories: {e}", file=sys.stderr)
        sys.exit(1)


def list_repositories(args: argparse.Namespace) -> None:
    """List indexed repositories.

    Args:
        args: Command-line arguments.
    """
    # Update environment variables
    if args.data_dir:
        os.environ["DATA_DIR"] = args.data_dir

    # Create client
    client = RepoSearchClient()

    # List repositories
    try:
        repositories = client.list_repositories()
        if not repositories:
            print("No repositories are currently indexed.")
            return

        print("Indexed Repositories:")
        print()
        for repo in repositories:
            print(f"- {repo.full_name}")
            print(f"  URL: {repo.url}")
            print(f"  Files: {repo.num_files}")
            print(f"  Chunks: {repo.num_chunks}")
            print(f"  Last indexed: {repo.last_indexed}")
            print()
    except Exception as e:
        print(f"Error listing repositories: {e}", file=sys.stderr)
        sys.exit(1)


def delete_repository(args: argparse.Namespace) -> None:
    """Delete a repository from the index.

    Args:
        args: Command-line arguments.
    """
    # Update environment variables
    if args.data_dir:
        os.environ["DATA_DIR"] = args.data_dir

    # Create client
    client = RepoSearchClient()

    # Delete repository
    try:
        success = client.delete_repository(args.repository)
        if success:
            print(f"Successfully deleted repository {args.repository} from the index.")
        else:
            print(f"Repository {args.repository} not found in the index.")
            sys.exit(1)
    except Exception as e:
        print(f"Error deleting repository: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    args = parse_args()

    if args.command == "index":
        index_repository(args)
    elif args.command == "search":
        search_repositories(args)
    elif args.command == "list":
        list_repositories(args)
    elif args.command == "delete":
        delete_repository(args)
    else:
        print("Please specify a command. Use --help for usage information.")
        sys.exit(1)


if __name__ == "__main__":
    main()
