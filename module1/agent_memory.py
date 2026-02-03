"""LangGraph agent with arithmetic tools.

This module demonstrates a ReAct-style agent that can perform sequential
arithmetic operations using tool calls.
"""

from typing import List

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
import asyncio
from mcp_server.mcp_client_utils import mcp_client_context, get_current_session

load_dotenv()


# Tool definitions
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer

    Returns:
        Sum of a and b
    """
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer

    Returns:
        Product of a and b
    """
    return a * b


def divide(a: int, b: int) -> float:
    """Divide two integers.

    Args:
        a: Dividend
        b: Divisor

    Returns:
        Quotient of a divided by b

    Raises:
        ZeroDivisionError: If b is zero
    """
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b


# Constants
TOOLS = [add, multiply, divide]
SYSTEM_MESSAGE = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)


def create_assistant_node(llm_with_tools):
    """Create an assistant node function.

    Args:
        llm_with_tools: LLM instance bound with tools

    Returns:
        Assistant node function
    """
    async def assistant(state: MessagesState) -> dict:
        """Process messages and invoke LLM with tools.

        Args:
            state: Current message state

        Returns:
            Dictionary with updated messages
        """
        # Use the shared session
        client = get_current_session()
        result = await client.get_prompt("arithmetic-agent-system-prompt")
        prompt_content = result.messages[0].content.text

        system_message = SystemMessage(content=prompt_content)
        messages = [system_message] + state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    return assistant


def build_agent_graph() -> StateGraph:
    """Build and compile the agent graph.

    Returns:
        Compiled StateGraph
    """
    # Initialize LLM with tools
    # Note: parallel_tool_calls=False ensures sequential execution for arithmetic
    # See: https://python.langchain.com/docs/how_to/tool_calling_parallel/
    llm = ChatOpenAI(model="gpt-5-nano")
    llm_with_tools = llm.bind_tools(TOOLS, parallel_tool_calls=False)

    # Create graph builder
    builder = StateGraph(MessagesState)

    # Add nodes
    builder.add_node("assistant", create_assistant_node(llm_with_tools))
    builder.add_node("tools", ToolNode(TOOLS))

    # Add edges
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


async def run_agent(graph: StateGraph, user_input: str) -> List[BaseMessage]:
    """Run the agent with a user input.

    Args:
        graph: Compiled agent graph
        user_input: User's query string

    Returns:
        List of messages from the conversation
    """
    # Specify a thread
    config = {"configurable": {"thread_id": "1"}}

    # Specify an input
    messages = [HumanMessage(content=user_input)]
    result = await graph.ainvoke({"messages": messages}, config=config)
    return result["messages"]


async def main():
    """Main execution function."""
    async with mcp_client_context():
        # Build the agent graph
        agent_graph = build_agent_graph()

        # Example query
        query = "Add 3 and 4. Multiply the output by 2. Divide the output by 5"
        messages = await run_agent(agent_graph, query)

        # Print results
        for message in messages:
            message.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())