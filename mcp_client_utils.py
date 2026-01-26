from contextlib import asynccontextmanager
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Get the absolute path to the server file
SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")

_CURRENT_SESSION: ClientSession | None = None

@asynccontextmanager
async def mcp_client_context():
    """Context manager that maintains a single MCP session for the duration of the context."""
    global _CURRENT_SESSION
    server_params = StdioServerParameters(
        command="uv",
        args=["run", SERVER_PATH],
        env=os.environ.copy()
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            _CURRENT_SESSION = session
            try:
                yield session
            finally:
                _CURRENT_SESSION = None

def get_current_session() -> ClientSession:
    """Get the currently active MCP session."""
    if _CURRENT_SESSION is None:
        raise RuntimeError("No active MCP session. Ensure code is running within an 'mcp_client_context' block.")
    return _CURRENT_SESSION

# Backwards compatibility for now, though we'll remove usage of this in refactor
@asynccontextmanager
async def get_mcp_client():
    """Deprecated: Use mcp_client_context for long-lived sessions."""
    async with mcp_client_context() as session:
        yield session
