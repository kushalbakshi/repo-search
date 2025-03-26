"""Utility functions for RepoSearch."""

import textwrap
from typing import List
import colorama
from colorama import Fore, Style
from repo_search.models import SearchResult


def pretty_print_results(results: List[SearchResult], max_content_length: int = 300) -> None:
    """Print search results in a nicely formatted way.
    
    Args:
        results: List of search results to print.
        max_content_length: Maximum length of content to show (will be truncated if longer).
    """
    # Initialize colorama
    colorama.init()
    
    if not results:
        print(f"{Fore.YELLOW}No results found.{Style.RESET_ALL}")
        return
    
    print(f"\n{Fore.CYAN}Found {len(results)} results:{Style.RESET_ALL}\n")
    
    for i, result in enumerate(results):
        # Format header with source and score
        header = f"Result #{i+1} - {result.source}"
        score_text = f"Score: {result.score:.4f}"
        
        # Print header
        print(f"{Fore.GREEN}{header} {Fore.YELLOW}[{score_text}]{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'-' * 80}{Style.RESET_ALL}")
        
        # Get content with truncation if needed
        content = result.content
        if len(content) > max_content_length:
            content = content[:max_content_length] + f"{Fore.RED}... (truncated){Style.RESET_ALL}"
        
        # Format content with proper indentation for readability
        formatted_content = textwrap.indent(content, '    ')
        print(formatted_content)
        print()  # Empty line between results
    
    # Reset colors at the end
    print(Style.RESET_ALL)
