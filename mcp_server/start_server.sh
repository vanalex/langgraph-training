#!/bin/bash
# Start MCP server with HTTP transport

cd "$(dirname "$0")"
echo "Starting MCP server on http://localhost:8000"
uv run mcp_server.py
