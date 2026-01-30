
import asyncio
from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
from dotenv import load_dotenv
load_dotenv()


# Define tools
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


def divide(a: int, b: int) -> float:
    """Divides a by b.

    Args:
        a: first int
        b: second int
    """
    return a / b


# System message
sys_msg = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)

# Create tools and LLM
tools = [add, multiply, divide]
llm = ChatOpenAI(model="gpt-5-mini")
llm_with_tools = llm.bind_tools(tools)


# Node
def assistant(state: MessagesState):
    """Assistant node that calls the LLM with tools."""
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


def build_graph(interrupt_before_tools: bool = True):
    """
    Build the agent graph with optional breakpoint before tools.

    Args:
        interrupt_before_tools: If True, interrupts before executing tools

    Returns:
        Compiled graph with memory
    """
    # Graph
    builder = StateGraph(MessagesState)

    # Define nodes: these do the work
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    # Define edges: these determine the control flow
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # Compile with memory and optional breakpoint
    memory = MemorySaver()
    if interrupt_before_tools:
        graph = builder.compile(interrupt_before=["tools"], checkpointer=memory)
    else:
        graph = builder.compile(checkpointer=memory)

    return graph


def basic_breakpoint_example():
    """
    Example: Basic breakpoint usage.

    The graph is interrupted before the tools node, allowing inspection
    of the state before tool execution.
    """
    print("=" * 50)
    print("BASIC BREAKPOINT EXAMPLE")
    print("=" * 50)

    graph = build_graph(interrupt_before_tools=True)

    # Input
    initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}

    # Thread
    thread = {"configurable": {"thread_id": "1"}}

    # Run the graph until the first interruption
    print("\nRunning until breakpoint...")
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

    # Check the state - next node should be 'tools'
    state = graph.get_state(thread)
    print(f"\nNext node to execute: {state.next}")
    print("\nGraph is interrupted! Ready for human approval.")


def continue_from_breakpoint():
    """
    Example: Continuing execution from a breakpoint.

    When we invoke the graph with None, it continues from the last checkpoint.
    """
    print("\n" + "=" * 50)
    print("CONTINUING FROM BREAKPOINT")
    print("=" * 50)

    graph = build_graph(interrupt_before_tools=True)

    # Input
    initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}

    # Thread
    thread = {"configurable": {"thread_id": "2"}}

    # Run the graph until the first interruption
    print("\nRunning until breakpoint...")
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

    # Continue from the breakpoint by passing None as input
    print("\nContinuing execution from breakpoint...")
    for event in graph.stream(None, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()


def user_approval_workflow():
    """
    Example: Human approval workflow.

    The graph interrupts before tool execution, asks for user approval,
    and continues only if approved.
    """
    print("\n" + "=" * 50)
    print("USER APPROVAL WORKFLOW")
    print("=" * 50)

    graph = build_graph(interrupt_before_tools=True)

    # Input
    initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}

    # Thread
    thread = {"configurable": {"thread_id": "3"}}

    # Run the graph until the first interruption
    print("\nRunning until breakpoint...")
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

    # Get user feedback
    user_approval = input("\nDo you want to call the tool? (yes/no): ")

    # Check approval
    if user_approval.lower() == "yes":
        # If approved, continue the graph execution
        print("\nApproved! Continuing execution...")
        for event in graph.stream(None, thread, stream_mode="values"):
            event["messages"][-1].pretty_print()
    else:
        print("Operation cancelled by user.")


async def api_breakpoint_example():
    """
    Example: Using breakpoints with LangGraph API.

    Note: This requires LangGraph Studio running locally.
    Only works on Mac and not in Google Colab.
    """
    try:
        from langgraph_sdk import get_client

        print("\n" + "=" * 50)
        print("API BREAKPOINT EXAMPLE")
        print("=" * 50)

        # Replace with your deployed graph URL
        URL = "http://localhost:56091"
        client = get_client(url=URL)

        # Create initial input and thread
        initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}
        thread = await client.threads.create()

        print("\nRunning with breakpoint before tools...")
        print("-" * 50)

        # Stream with interrupt_before passed to the stream method
        async for chunk in client.runs.stream(
            thread["thread_id"],
            assistant_id="agent",
            input=initial_input,
            stream_mode="values",
            interrupt_before=["tools"],
        ):
            print(f"Event type: {chunk.event}")
            messages = chunk.data.get("messages", [])
            if messages:
                print(f"Last message: {messages[-1]}")
            print("-" * 50)

        print("\nContinuing from breakpoint...")
        print("-" * 50)

        # Continue from breakpoint by passing None as input
        async for chunk in client.runs.stream(
            thread["thread_id"],
            "agent",
            input=None,
            stream_mode="values",
            interrupt_before=["tools"],
        ):
            print(f"Event type: {chunk.event}")
            messages = chunk.data.get("messages", [])
            if messages:
                print(f"Last message: {messages[-1]}")
            print("-" * 50)

    except ImportError:
        print("LangGraph SDK not installed. Install with: pip install langgraph_sdk")
    except Exception as e:
        print(f"API example requires LangGraph Studio running locally: {e}")


def main():
    """Run all breakpoint examples."""
    # Basic breakpoint example
    basic_breakpoint_example()

    # Continue from breakpoint
    continue_from_breakpoint()

    # User approval workflow
    user_approval_workflow()

    # API breakpoint example (requires Studio)
    print("\n")
    asyncio.run(api_breakpoint_example())


if __name__ == "__main__":
    main()
