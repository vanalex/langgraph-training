import asyncio
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


def build_graph():
    """
    Build the agent graph with memory for time travel.

    Returns:
        Compiled graph with checkpointer
    """
    builder = StateGraph(MessagesState)

    # Define nodes
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    # Define edges
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    # Compile with memory (required for time travel)
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    return graph


def browsing_history_example():
    """
    Example: Browsing state history.

    Shows how to access all past states of a graph execution
    using get_state_history().
    """
    print("=" * 50)
    print("BROWSING STATE HISTORY")
    print("=" * 50)

    graph = build_graph()

    # Run the graph
    initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}
    thread = {"configurable": {"thread_id": "1"}}

    print("\nRunning graph...")
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

    # Get current state
    print("\n" + "=" * 50)
    print("CURRENT STATE")
    print("=" * 50)

    current_state = graph.get_state(thread)
    print(f"Checkpoint ID: {current_state.config['configurable']['checkpoint_id']}")
    print(f"Next nodes: {current_state.next}")
    print(f"Number of messages: {len(current_state.values['messages'])}")

    # Browse history
    print("\n" + "=" * 50)
    print("STATE HISTORY")
    print("=" * 50)

    all_states = [s for s in graph.get_state_history(thread)]
    print(f"Total states in history: {len(all_states)}")

    # Show each state
    for i, state in enumerate(reversed(all_states)):
        print(f"\nState {i}:")
        print(f"  Checkpoint ID: {state.config['configurable']['checkpoint_id']}")
        print(f"  Next nodes: {state.next}")
        print(f"  Messages: {len(state.values['messages'])}")
        if state.values["messages"]:
            print(f"  Last message type: {type(state.values['messages'][-1]).__name__}")


def replaying_example():
    """
    Example: Replaying from a previous checkpoint.

    Shows how to re-execute the graph from a specific point in history
    by passing the checkpoint_id to stream().
    """
    print("\n" + "=" * 50)
    print("REPLAYING FROM CHECKPOINT")
    print("=" * 50)

    graph = build_graph()

    # Run the graph
    initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}
    thread = {"configurable": {"thread_id": "2"}}

    print("\nInitial run...")
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

    # Get history and select a checkpoint to replay
    all_states = [s for s in graph.get_state_history(thread)]
    to_replay = all_states[-2]  # State with human input, before first assistant call

    print("\n" + "=" * 50)
    print("STATE TO REPLAY")
    print("=" * 50)
    print(f"Checkpoint ID: {to_replay.config['configurable']['checkpoint_id']}")
    print(f"Next nodes: {to_replay.next}")
    print(f"State values: {to_replay.values}")

    # Replay from this checkpoint
    print("\n" + "=" * 50)
    print("REPLAYING...")
    print("=" * 50)

    for event in graph.stream(None, to_replay.config, stream_mode="values"):
        event["messages"][-1].pretty_print()

    print("\nReplay complete! The graph re-executed from the checkpoint.")


def forking_example():
    """
    Example: Forking by modifying a past state.

    Shows how to create a new execution branch by updating state
    at a checkpoint with a different input (using message ID to overwrite).
    """
    print("\n" + "=" * 50)
    print("FORKING EXECUTION")
    print("=" * 50)

    graph = build_graph()

    # Run the graph
    initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}
    thread = {"configurable": {"thread_id": "3"}}

    print("\nInitial run...")
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

    # Get history and select a checkpoint to fork
    all_states = [s for s in graph.get_state_history(thread)]
    to_fork = all_states[-2]  # State with human input

    print("\n" + "=" * 50)
    print("STATE TO FORK")
    print("=" * 50)
    print(f"Original message: {to_fork.values['messages'][0].content}")
    print(f"Message ID: {to_fork.values['messages'][0].id}")
    print(f"Checkpoint ID: {to_fork.config['configurable']['checkpoint_id']}")

    # Update state with new input (using same message ID to overwrite)
    print("\n" + "=" * 50)
    print("CREATING FORK")
    print("=" * 50)

    fork_config = graph.update_state(
        to_fork.config,
        {
            "messages": [
                HumanMessage(
                    content="Multiply 5 and 3", id=to_fork.values["messages"][0].id
                )
            ]
        },
    )

    print(f"New checkpoint ID: {fork_config['configurable']['checkpoint_id']}")

    # Check updated state
    current_state = graph.get_state(thread)
    print(f"Updated message: {current_state.values['messages'][0].content}")

    # Run from the fork
    print("\n" + "=" * 50)
    print("RUNNING FROM FORK")
    print("=" * 50)

    for event in graph.stream(None, fork_config, stream_mode="values"):
        event["messages"][-1].pretty_print()

    print("\nFork complete! Created a new execution branch with different input.")


async def api_time_travel_example():
    """
    Example: Time travel with LangGraph API.

    Demonstrates replaying and forking using the LangGraph SDK.
    Note: This requires LangGraph Studio running locally.
    """
    try:
        from langgraph_sdk import get_client

        print("\n" + "=" * 50)
        print("API TIME TRAVEL EXAMPLE")
        print("=" * 50)

        # Replace with your deployed graph URL
        URL = "http://localhost:62780"
        client = get_client(url=URL)

        # Initial run
        print("\nInitial run...")
        initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}
        thread = await client.threads.create()

        async for chunk in client.runs.stream(
            thread["thread_id"],
            assistant_id="agent",
            input=initial_input,
            stream_mode="updates",
        ):
            if chunk.data:
                assistant_node = chunk.data.get("assistant", {}).get("messages", [])
                tool_node = chunk.data.get("tools", {}).get("messages", [])
                if assistant_node:
                    print(f"Assistant: {assistant_node[-1].get('content', 'Tool call')}")
                elif tool_node:
                    print(f"Tool result: {tool_node[-1].get('content')}")

        # Get history
        print("\n" + "=" * 50)
        print("REPLAYING FROM CHECKPOINT")
        print("=" * 50)

        states = await client.threads.get_history(thread["thread_id"])
        to_replay = states[-2]  # State with human input
        print(f"Replaying from checkpoint: {to_replay['checkpoint_id']}")

        async for chunk in client.runs.stream(
            thread["thread_id"],
            assistant_id="agent",
            input=None,
            stream_mode="updates",
            checkpoint_id=to_replay["checkpoint_id"],
        ):
            if chunk.data:
                assistant_node = chunk.data.get("assistant", {}).get("messages", [])
                tool_node = chunk.data.get("tools", {}).get("messages", [])
                if assistant_node:
                    print(f"Assistant: {assistant_node[-1].get('content', 'Tool call')}")
                elif tool_node:
                    print(f"Tool result: {tool_node[-1].get('content')}")

        # Forking
        print("\n" + "=" * 50)
        print("FORKING EXECUTION")
        print("=" * 50)

        # Create new thread for forking example
        thread2 = await client.threads.create()

        async for chunk in client.runs.stream(
            thread2["thread_id"],
            assistant_id="agent",
            input=initial_input,
            stream_mode="updates",
        ):
            pass  # Run to completion

        states2 = await client.threads.get_history(thread2["thread_id"])
        to_fork = states2[-2]

        # Update state to fork
        forked_input = {
            "messages": HumanMessage(
                content="Multiply 3 and 3", id=to_fork["values"]["messages"][0]["id"]
            )
        }

        forked_config = await client.threads.update_state(
            thread2["thread_id"], forked_input, checkpoint_id=to_fork["checkpoint_id"]
        )

        print(f"Created fork with checkpoint: {forked_config['checkpoint_id']}")

        # Run from fork
        async for chunk in client.runs.stream(
            thread2["thread_id"],
            assistant_id="agent",
            input=None,
            stream_mode="updates",
            checkpoint_id=forked_config["checkpoint_id"],
        ):
            if chunk.data:
                assistant_node = chunk.data.get("assistant", {}).get("messages", [])
                tool_node = chunk.data.get("tools", {}).get("messages", [])
                if assistant_node:
                    print(f"Assistant: {assistant_node[-1].get('content', 'Tool call')}")
                elif tool_node:
                    print(f"Tool result: {tool_node[-1].get('content')}")

    except ImportError:
        print("LangGraph SDK not installed. Install with: pip install langgraph_sdk")
    except Exception as e:
        print(f"API example requires LangGraph Studio running locally: {e}")


def main():
    """Run all time travel examples."""
    # Browse history
    browsing_history_example()

    # Replay from checkpoint
    replaying_example()

    # Fork execution
    forking_example()

    # API examples (requires Studio)
    print("\n")
    asyncio.run(api_time_travel_example())


if __name__ == "__main__":
    main()
