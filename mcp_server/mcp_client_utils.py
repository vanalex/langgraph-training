from contextlib import asynccontextmanager
import os
from fastmcp import MultiServerMCPClient

# MCP server HTTP URL
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")

_CURRENT_CLIENT: MultiServerMCPClient | None = None

@asynccontextmanager
async def mcp_client_context():
    """Context manager that maintains a single MCP client for the duration of the context."""
    global _CURRENT_CLIENT

    client = MultiServerMCPClient(
        {
            "prompts": {
                "transport": "http",
                "url": MCP_SERVER_URL,
            }
        }
    )

    async with client:
        _CURRENT_CLIENT = client
        try:
            yield client
        finally:
            _CURRENT_CLIENT = None

def get_current_session() -> MultiServerMCPClient:
    """Get the currently active MCP client."""
    if _CURRENT_CLIENT is None:
        raise RuntimeError("No active MCP client. Ensure code is running within an 'mcp_client_context' block.")
    return _CURRENT_CLIENT

# Backwards compatibility for now, though we'll remove usage of this in refactor
@asynccontextmanager
async def get_mcp_client():
    """Deprecated: Use mcp_client_context for long-lived sessions."""
    async with mcp_client_context() as client:
        yield client
