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
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)


# Assistant node
def assistant(state: MessagesState):
    """Assistant node that calls the LLM with tools."""
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


def build_graph_with_edit(interrupt_before_assistant: bool = True):
    """
    Build the agent graph with breakpoint before assistant node.

    This allows editing the state before the assistant processes messages.

    Args:
        interrupt_before_assistant: If True, interrupts before assistant node

    Returns:
        Compiled graph with memory
    """
    builder = StateGraph(MessagesState)

    # Define nodes
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    # Define edges
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    # Compile with memory and optional breakpoint
    memory = MemorySaver()
    if interrupt_before_assistant:
        graph = builder.compile(interrupt_before=["assistant"], checkpointer=memory)
    else:
        graph = builder.compile(checkpointer=memory)

    return graph


def edit_state_example():
    """
    Example: Editing graph state at a breakpoint.

    The graph interrupts before the assistant node, allowing us to
    modify the messages in the state before the assistant processes them.
    """
    print("=" * 50)
    print("EDITING STATE EXAMPLE")
    print("=" * 50)

    graph = build_graph_with_edit(interrupt_before_assistant=True)

    # Input
    initial_input = {"messages": "Multiply 2 and 3"}

    # Thread
    thread = {"configurable": {"thread_id": "1"}}

    # Run the graph until the first interruption
    print("\nRunning until breakpoint...")
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

    # Check current state
    state = graph.get_state(thread)
    print(f"\nNext node to execute: {state.next}")
    print("\nCurrent messages in state:")
    for m in state.values["messages"]:
        m.pretty_print()

    # Update the state by adding a new message
    print("\n" + "=" * 50)
    print("UPDATING STATE")
    print("=" * 50)
    graph.update_state(
        thread,
        {"messages": [HumanMessage(content="No, actually multiply 3 and 3!")]},
    )

    # Check the updated state
    new_state = graph.get_state(thread).values
    print("\nUpdated messages in state:")
    for m in new_state["messages"]:
        m.pretty_print()

    # Continue execution from the breakpoint
    print("\n" + "=" * 50)
    print("CONTINUING EXECUTION")
    print("=" * 50)
    for event in graph.stream(None, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

    # Continue again to complete the conversation
    print("\n" + "=" * 50)
    print("FINAL RESPONSE")
    print("=" * 50)
    for event in graph.stream(None, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()


async def api_edit_state_example():
    """
    Example: Editing graph state using LangGraph API.

    This demonstrates:
    1. Running graph with interrupt_before via API
    2. Getting current state
    3. Modifying message content by ID
    4. Updating state via API
    5. Continuing execution
    """
    try:
        from langgraph_sdk import get_client

        print("=" * 50)
        print("API EDIT STATE EXAMPLE")
        print("=" * 50)

        # Replace with your deployed graph URL
        URL = "http://localhost:56091"
        client = get_client(url=URL)

        # Create initial input and thread
        initial_input = {"messages": "Multiply 2 and 3"}
        thread = await client.threads.create()

        print("\nRunning with breakpoint before assistant...")
        print("-" * 50)

        # Stream with interrupt_before
        async for chunk in client.runs.stream(
            thread["thread_id"],
            "agent",
            input=initial_input,
            stream_mode="values",
            interrupt_before=["assistant"],
        ):
            messages = chunk.data.get("messages", [])
            if messages:
                print(f"Last message: {messages[-1]}")

        # Get current state
        print("\n" + "=" * 50)
        print("GETTING AND MODIFYING STATE")
        print("=" * 50)

        current_state = await client.threads.get_state(thread["thread_id"])
        print(f"Next node: {current_state['next']}")

        # Get last message
        last_message = current_state["values"]["messages"][-1]
        print(f"\nOriginal message: {last_message['content']}")

        # Edit the message content (keeping the same ID to overwrite)
        last_message["content"] = "No, actually multiply 3 and 3!"
        print(f"Modified message: {last_message['content']}")

        # Update state
        await client.threads.update_state(
            thread["thread_id"], {"messages": last_message}
        )

        # Continue execution
        print("\n" + "=" * 50)
        print("CONTINUING EXECUTION")
        print("=" * 50)

        async for chunk in client.runs.stream(
            thread["thread_id"],
            assistant_id="agent",
            input=None,
            stream_mode="values",
            interrupt_before=["assistant"],
        ):
            messages = chunk.data.get("messages", [])
            if messages:
                print(f"Last message: {messages[-1]}")
            print("-" * 50)

        # Continue one more time to get final response
        print("\n" + "=" * 50)
        print("FINAL RESPONSE")
        print("=" * 50)

        async for chunk in client.runs.stream(
            thread["thread_id"],
            assistant_id="agent",
            input=None,
            stream_mode="values",
            interrupt_before=["assistant"],
        ):
            messages = chunk.data.get("messages", [])
            if messages:
                print(f"Last message: {messages[-1]}")
            print("-" * 50)

    except ImportError:
        print("LangGraph SDK not installed. Install with: pip install langgraph_sdk")
    except Exception as e:
        print(f"API example requires LangGraph Studio running locally: {e}")


# Human feedback node
def human_feedback(state: MessagesState):
    """No-op node that should be interrupted on for human feedback."""
    pass


def build_graph_with_human_feedback():
    """
    Build a graph with a dedicated human feedback node.

    The human_feedback node acts as a placeholder where the graph
    interrupts to await user input. This is useful for implementing
    human-in-the-loop workflows.

    Returns:
        Compiled graph with memory
    """
    builder = StateGraph(MessagesState)

    # Define nodes
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("human_feedback", human_feedback)

    # Define edges
    builder.add_edge(START, "human_feedback")
    builder.add_edge("human_feedback", "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "human_feedback")

    # Compile with breakpoint before human_feedback
    memory = MemorySaver()
    graph = builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)

    return graph


def human_feedback_example():
    """
    Example: Using a human feedback node for user input.

    This demonstrates:
    1. Graph with dedicated human_feedback node
    2. Interrupting before human_feedback
    3. Using update_state with as_node parameter
    4. Applying state updates as if they came from a specific node
    """
    print("=" * 50)
    print("HUMAN FEEDBACK NODE EXAMPLE")
    print("=" * 50)

    graph = build_graph_with_human_feedback()

    # Input
    initial_input = {"messages": "Multiply 2 and 3"}

    # Thread
    thread = {"configurable": {"thread_id": "5"}}

    # Run the graph until the first interruption
    print("\nRunning until human_feedback node...")
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

    # Get user input
    user_input = input("\nTell me how you want to update the state: ")

    # Update state as if we are the human_feedback node
    # The as_node parameter applies this update as the specified node
    print("\nUpdating state as human_feedback node...")
    graph.update_state(thread, {"messages": user_input}, as_node="human_feedback")

    # Continue the graph execution
    print("\nContinuing execution...")
    for event in graph.stream(None, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

    # Continue one more time to get final response
    print("\nGetting final response...")
    for event in graph.stream(None, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()


def main():
    """Run all edit state and human feedback examples."""
    # Edit state example
    edit_state_example()

    # Human feedback example (interactive)
    print("\n")
    human_feedback_example()

    # API edit state example (requires Studio)
    print("\n")
    asyncio.run(api_edit_state_example())


if __name__ == "__main__":
    main()
