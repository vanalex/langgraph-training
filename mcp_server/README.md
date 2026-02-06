# MCP Server - HTTP Transport

This MCP server provides prompts for the LangGraph agents using streamable HTTP transport.

## Starting the Server

```bash
cd mcp_server
./start_server.sh
```

Or manually:

```bash
cd mcp_server
uv run mcp_server.py
```

The server will start on `http://localhost:8000`

## Configuration

The MCP server URL can be configured via environment variable:

```bash
export MCP_SERVER_URL="http://localhost:8000/mcp"
```

Default: `http://localhost:8000/mcp`

The client uses `MultiServerMCPClient` from FastMCP for HTTP transport.

## Available Prompts

- `arithmetic-agent-system-prompt` - System prompt for arithmetic agents
- `conversation-summary-system-prompt` - System prompt with conversation summary
- `update-summary-prompt` - Prompt to update conversation summary
- `create-summary-prompt` - Prompt to create new summary
- `analyst-instructions` - Instructions for creating AI analyst personas
- `question-instructions` - Instructions for analyst interview questions
- `search-instructions` - Instructions for generating search queries
- `answer-instructions` - Instructions for expert answers
- `section-writer-instructions` - Instructions for writing report sections
- `report-writer-instructions` - Instructions for final report writing
- `intro-conclusion-instructions` - Instructions for intro/conclusion writing

## Client Usage

All agents automatically connect to the MCP server via the HTTP transport when running within the `mcp_client_context()`:

```python
from mcp_server.mcp_client_utils import mcp_client_context, get_current_session

async def main():
    async with mcp_client_context():
        client = get_current_session()
        result = await client.get_prompt("arithmetic-agent-system-prompt")
        print(result.messages[0].content.text)
```
