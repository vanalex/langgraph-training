import asyncio
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from dotenv import load_dotenv
load_dotenv()


# LLM
model = ChatOpenAI(model="gpt-5-mini", temperature=0)


# State
class State(MessagesState):
    summary: str


# Define the logic to call the model
def call_model(state: State, config: RunnableConfig):
    """Call the chat model with conversation history and summary."""
    # Get summary if it exists
    summary = state.get("summary", "")

    # If there is summary, then we add it
    if summary:
        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"

        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]

    response = model.invoke(messages, config)
    return {"messages": response}


def summarize_conversation(state: State):
    """Summarize the conversation and remove old messages."""
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt
    if summary:
        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


# Determine whether to end or summarize the conversation
def should_continue(state: State) -> Literal["summarize_conversation", END]:
    """Return the next node to execute."""
    messages = state["messages"]

    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"

    # Otherwise we can just end
    return END


# Build the graph
def build_graph():
    """Build and compile the conversation graph."""
    # Define a new graph
    workflow = StateGraph(State)
    workflow.add_node("conversation", call_model)
    workflow.add_node(summarize_conversation)

    # Set the entrypoint as conversation
    workflow.add_edge(START, "conversation")
    workflow.add_conditional_edges("conversation", should_continue)
    workflow.add_edge("summarize_conversation", END)

    # Compile
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    return graph


# Streaming examples
def stream_updates_example():
    """Example: Stream updates mode - only shows state updates after each node."""
    print("=" * 50)
    print("STREAMING UPDATES MODE")
    print("=" * 50)

    graph = build_graph()
    config = {"configurable": {"thread_id": "1"}}

    # Start conversation
    for chunk in graph.stream(
        {"messages": [HumanMessage(content="hi! I'm Lance")]},
        config,
        stream_mode="updates"
    ):
        print(f"Node: {list(chunk.keys())[0]}")
        chunk[list(chunk.keys())[0]]["messages"].pretty_print()
        print("-" * 50)


def stream_values_example():
    """Example: Stream values mode - shows full state after each node."""
    print("=" * 50)
    print("STREAMING VALUES MODE")
    print("=" * 50)

    graph = build_graph()
    config = {"configurable": {"thread_id": "2"}}

    # Start conversation
    input_message = HumanMessage(content="hi! I'm Lance")
    for event in graph.stream(
        {"messages": [input_message]},
        config,
        stream_mode="values"
    ):
        print("Current state:")
        for m in event["messages"]:
            m.pretty_print()
        print("-" * 75)


async def stream_tokens_example():
    """Example: Stream tokens from chat model as they are generated."""
    print("=" * 50)
    print("STREAMING TOKENS")
    print("=" * 50)

    graph = build_graph()
    node_to_stream = "conversation"
    config = {"configurable": {"thread_id": "4"}}
    input_message = HumanMessage(content="Tell me about the 49ers NFL team")

    async for event in graph.astream_events(
        {"messages": [input_message]},
        config,
        version="v2"
    ):
        # Get chat model tokens from a particular node
        if (
            event["event"] == "on_chat_model_stream"
            and event["metadata"].get("langgraph_node", "") == node_to_stream
        ):
            data = event["data"]
            print(data["chunk"].content, end="|", flush=True)

    print("\n" + "=" * 50)


async def stream_api_example():
    """
    Example: Stream using LangGraph API with SDK.

    Note: This requires LangGraph Studio running locally.
    Only works on Mac and not in Google Colab.
    """
    try:
        from langgraph_sdk import get_client

        print("=" * 50)
        print("STREAMING WITH LANGGRAPH API")
        print("=" * 50)

        # Replace with your deployed graph URL
        URL = "http://localhost:56091"
        client = get_client(url=URL)

        # Create a new thread
        thread = await client.threads.create()
        input_message = HumanMessage(content="Multiply 2 and 3")

        print("Streaming values mode:")
        print("-" * 50)

        async for event in client.runs.stream(
            thread["thread_id"],
            assistant_id="agent",
            input={"messages": [input_message]},
            stream_mode="values"
        ):
            messages = event.data.get("messages", None)
            if messages:
                from langchain_core.messages import convert_to_messages
                print(convert_to_messages(messages)[-1])
            print("=" * 25)

        # Messages mode streaming
        thread2 = await client.threads.create()

        print("\n" + "=" * 50)
        print("Streaming messages mode:")
        print("-" * 50)

        async for event in client.runs.stream(
            thread2["thread_id"],
            assistant_id="agent",
            input={"messages": [input_message]},
            stream_mode="messages"
        ):
            if event.event == "metadata":
                print(f"Metadata: Run ID - {event.data['run_id']}")
                print("-" * 50)
            elif event.event == "messages/partial":
                for data_item in event.data:
                    content = data_item.get("content", "")
                    if content:
                        print(f"AI: {content}")

    except ImportError:
        print("LangGraph SDK not installed. Install with: pip install langgraph_sdk")
    except Exception as e:
        print(f"API streaming example requires LangGraph Studio running locally: {e}")


def main():
    """Run all streaming examples."""
    # Synchronous streaming examples
    stream_updates_example()
    print("\n")
    stream_values_example()

    # Asynchronous streaming examples
    print("\n")
    asyncio.run(stream_tokens_example())

    # API streaming (requires Studio)
    print("\n")
    asyncio.run(stream_api_example())


if __name__ == "__main__":
    main()
