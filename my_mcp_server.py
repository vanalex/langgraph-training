from fastmcp import FastMCP

mcp = FastMCP("My Agent Prompts")

@mcp.prompt("arithmetic-agent-system-prompt")
def arithmetic_prompt() -> str:
    """System prompt for the arithmetic agent."""
    return "You are a helpful assistant tasked with performing arithmetic on a set of inputs."

@mcp.prompt("conversation-summary-system-prompt")
def conversation_summary(summary: str) -> str:
    """System prompt that includes the conversation summary."""
    return f"Summary of conversation earlier: {summary}"

@mcp.prompt("update-summary-prompt")
def update_summary(summary: str) -> str:
    """Prompt to update the conversation summary."""
    return (
        f"This is summary of the conversation to date: {summary}\n\n"
        "Extend the summary by taking into account the new messages above:"
    )

@mcp.prompt("create-summary-prompt")
def create_summary() -> str:
    """Prompt to create a new conversation summary."""
    return "Create a summary of the conversation above:"

if __name__ == "__main__":
    mcp.run()
