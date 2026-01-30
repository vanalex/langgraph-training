import asyncio
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import NodeInterrupt
from langgraph.graph import START, END, StateGraph
from dotenv import load_dotenv
load_dotenv()


class State(TypedDict):
    """Simple state with input field."""
    input: str


def step_1(state: State) -> State:
    """First step in the graph."""
    print("---Step 1---")
    return state


def step_2(state: State) -> State:
    """
    Second step with conditional dynamic breakpoint.

    Raises NodeInterrupt if input length exceeds 5 characters.
    """
    # Dynamic breakpoint: interrupt if input is too long
    if len(state["input"]) > 5:
        raise NodeInterrupt(
            f"Received input that is longer than 5 characters: {state['input']}"
        )

    print("---Step 2---")
    return state


def step_3(state: State) -> State:
    """Third step in the graph."""
    print("---Step 3---")
    return state


def build_graph():
    """
    Build a graph with dynamic breakpoint in step_2.

    The breakpoint is triggered internally based on input length,
    not set at compile time.

    Returns:
        Compiled graph with memory
    """
    builder = StateGraph(State)
    builder.add_node("step_1", step_1)
    builder.add_node("step_2", step_2)
    builder.add_node("step_3", step_3)
    builder.add_edge(START, "step_1")
    builder.add_edge("step_1", "step_2")
    builder.add_edge("step_2", "step_3")
    builder.add_edge("step_3", END)

    # Set up memory
    memory = MemorySaver()

    # Compile the graph with memory (no static breakpoints)
    graph = builder.compile(checkpointer=memory)

    return graph


def dynamic_breakpoint_long_input():
    """
    Example: Dynamic breakpoint triggered by long input.

    The graph interrupts during step_2 because the input exceeds 5 characters.
    """
    print("=" * 50)
    print("DYNAMIC BREAKPOINT - LONG INPUT")
    print("=" * 50)

    graph = build_graph()

    # Run with long input (> 5 characters)
    initial_input = {"input": "hello world"}
    thread_config = {"configurable": {"thread_id": "1"}}

    print("\nRunning with long input...")
    for event in graph.stream(initial_input, thread_config, stream_mode="values"):
        print(event)

    # Inspect state to see the interruption
    print("\n" + "=" * 50)
    print("INSPECTING STATE")
    print("=" * 50)

    state = graph.get_state(thread_config)
    print(f"Next node to execute: {state.next}")
    print(f"\nTasks: {state.tasks}")

    # Check if there's an interrupt
    if state.tasks and state.tasks[0].interrupts:
        interrupt = state.tasks[0].interrupts[0]
        print(f"\nInterrupt value: {interrupt.value}")
        print(f"Interrupt when: {interrupt.when}")

    # Try to resume without changing state (will interrupt again)
    print("\n" + "=" * 50)
    print("ATTEMPTING TO RESUME (will interrupt again)")
    print("=" * 50)

    for event in graph.stream(None, thread_config, stream_mode="values"):
        print(event)

    state = graph.get_state(thread_config)
    print(f"Next node: {state.next} (still stuck!)")


def dynamic_breakpoint_with_fix():
    """
    Example: Fixing the condition and resuming execution.

    After the dynamic breakpoint, we update the state to fix the condition,
    then successfully resume execution.
    """
    print("\n" + "=" * 50)
    print("DYNAMIC BREAKPOINT - WITH FIX")
    print("=" * 50)

    graph = build_graph()

    # Run with long input
    initial_input = {"input": "hello world"}
    thread_config = {"configurable": {"thread_id": "2"}}

    print("\nRunning with long input...")
    for event in graph.stream(initial_input, thread_config, stream_mode="values"):
        print(event)

    # Update state to fix the condition
    print("\n" + "=" * 50)
    print("UPDATING STATE TO FIX CONDITION")
    print("=" * 50)

    graph.update_state(thread_config, {"input": "hi"})

    state = graph.get_state(thread_config)
    print(f"Updated state: {state.values}")

    # Resume execution
    print("\n" + "=" * 50)
    print("RESUMING EXECUTION")
    print("=" * 50)

    for event in graph.stream(None, thread_config, stream_mode="values"):
        print(event)

    print("\nExecution completed successfully!")


def dynamic_breakpoint_short_input():
    """
    Example: No breakpoint with short input.

    When input is 5 characters or less, the graph executes without interruption.
    """
    print("\n" + "=" * 50)
    print("NO BREAKPOINT - SHORT INPUT")
    print("=" * 50)

    graph = build_graph()

    # Run with short input (<= 5 characters)
    initial_input = {"input": "hi"}
    thread_config = {"configurable": {"thread_id": "3"}}

    print("\nRunning with short input...")
    for event in graph.stream(initial_input, thread_config, stream_mode="values"):
        print(event)

    print("\nExecution completed without interruption!")


async def api_dynamic_breakpoint_example():
    """
    Example: Dynamic breakpoints with LangGraph API.

    Note: This requires LangGraph Studio running locally.
    Only works on Mac and not in Google Colab.
    """
    try:
        from langgraph_sdk import get_client

        print("\n" + "=" * 50)
        print("API DYNAMIC BREAKPOINT EXAMPLE")
        print("=" * 50)

        # Replace with your deployed graph URL
        URL = "http://localhost:62575"
        client = get_client(url=URL)

        # Create thread and input
        thread = await client.threads.create()
        input_dict = {"input": "hello world"}

        print("\nRunning with long input via API...")
        print("-" * 50)

        # Stream execution
        async for chunk in client.runs.stream(
            thread["thread_id"],
            assistant_id="dynamic_breakpoints",
            input=input_dict,
            stream_mode="values",
        ):
            print(f"Event type: {chunk.event}")
            print(chunk.data)
            print()

        # Check state
        print("=" * 50)
        print("CHECKING STATE")
        print("=" * 50)

        current_state = await client.threads.get_state(thread["thread_id"])
        print(f"Next node: {current_state['next']}")
        print(f"Current input: {current_state['values']['input']}")

        # Update state
        print("\n" + "=" * 50)
        print("UPDATING STATE")
        print("=" * 50)

        await client.threads.update_state(thread["thread_id"], {"input": "hi!"})
        print("State updated to: {'input': 'hi!'}")

        # Resume execution
        print("\n" + "=" * 50)
        print("RESUMING EXECUTION")
        print("=" * 50)

        async for chunk in client.runs.stream(
            thread["thread_id"],
            assistant_id="dynamic_breakpoints",
            input=None,
            stream_mode="values",
        ):
            print(f"Event type: {chunk.event}")
            print(chunk.data)
            print()

        # Check final state
        current_state = await client.threads.get_state(thread["thread_id"])
        print(f"Final state: {current_state['values']}")

    except ImportError:
        print("LangGraph SDK not installed. Install with: pip install langgraph_sdk")
    except Exception as e:
        print(f"API example requires LangGraph Studio running locally: {e}")


def main():
    """Run all dynamic breakpoint examples."""
    # Example 1: Long input triggers breakpoint
    dynamic_breakpoint_long_input()

    # Example 2: Fix condition and resume
    dynamic_breakpoint_with_fix()

    # Example 3: Short input, no breakpoint
    dynamic_breakpoint_short_input()

    # API example (requires Studio)
    print("\n")
    asyncio.run(api_dynamic_breakpoint_example())


if __name__ == "__main__":
    main()
