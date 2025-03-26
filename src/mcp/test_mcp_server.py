#!/usr/bin/env python3
"""Test script for the MCP server."""

import asyncio
import json
import subprocess
import sys
import time

async def test_server():
    """Test the MCP server implementation."""
    # Start the server process
    server_process = subprocess.Popen(
        ["python", "src/mcp/server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    
    # Give the server some time to start
    await asyncio.sleep(1)
    
    try:
        print("Testing MCP server...")
        
        # Test initialize request
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        }
        server_process.stdin.write(json.dumps(request) + "\n")
        server_process.stdin.flush()
        
        # Read response
        response_line = server_process.stdout.readline().strip()
        response = json.loads(response_line)
        print("\nInitialize Response:")
        print(json.dumps(response, indent=2))
        
        # Test tools/list request
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        server_process.stdin.write(json.dumps(request) + "\n")
        server_process.stdin.flush()
        
        # Read response
        response_line = server_process.stdout.readline().strip()
        response = json.loads(response_line)
        print("\nList Tools Response:")
        print(json.dumps(response, indent=2))
        
        # Test resources/list request
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/list",
            "params": {}
        }
        server_process.stdin.write(json.dumps(request) + "\n")
        server_process.stdin.flush()
        
        # Read response
        response_line = server_process.stdout.readline().strip()
        response = json.loads(response_line)
        print("\nList Resources Response:")
        print(json.dumps(response, indent=2))
        
        # Test tools/call request - list_indexed_repositories
        request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "list_indexed_repositories",
                "arguments": {}
            }
        }
        server_process.stdin.write(json.dumps(request) + "\n")
        server_process.stdin.flush()
        
        # Read response
        response_line = server_process.stdout.readline().strip()
        response = json.loads(response_line)
        print("\nList Indexed Repositories Response:")
        print(json.dumps(response, indent=2))
        
        print("\nAll tests completed successfully!")
    
    finally:
        # Terminate the server process
        server_process.terminate()
        await asyncio.sleep(0.5)
        
        # Check if process is still running, kill if necessary
        if server_process.poll() is None:
            server_process.kill()

if __name__ == "__main__":
    asyncio.run(test_server())
