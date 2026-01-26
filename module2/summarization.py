from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
import asyncio
from mcp_client_utils import mcp_client_context, get_current_session

load_dotenv()

model = ChatOpenAI(model="gpt-5-mini",temperature=0)

class State(MessagesState):
    summary: str

async def call_model(state: State):
    summary = state.get("summary", "")

    # If there is summary, then we add it
    if summary:
        client = get_current_session()
        result = await client.get_prompt("conversation-summary-system-prompt", arguments={"summary": summary})
        system_message_content = result.messages[0].content.text
             
        # Add summary to system message
        system_message = system_message_content

        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state["messages"]

    else:
        messages = state["messages"]

    response = await model.ainvoke(messages)
    return {"messages": response}


async def summarize_conversation(state: State):
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt
    if summary:
        client = get_current_session()
        result = await client.get_prompt("update-summary-prompt", arguments={"summary": summary})
        summary_message = result.messages[0].content.text

    else:
        client = get_current_session()
        result = await client.get_prompt("create-summary-prompt")
        summary_message = result.messages[0].content.text

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = await model.ainvoke(messages)

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


def should_continue(state: State):
    """Return the next node to execute."""

    messages = state["messages"]

    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"

    # Otherwise we can just end
    return END

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

config = {"configurable": {"thread_id": "1"}}

async def main():
    async with mcp_client_context():
        # Start conversation
        input_message = HumanMessage(content="hi! I'm Lance")
        output = await graph.ainvoke({"messages": [input_message]}, config)
        for m in output['messages'][-1:]:
            m.pretty_print()

        input_message = HumanMessage(content="what's my name?")
        output = await graph.ainvoke({"messages": [input_message]}, config)
        for m in output['messages'][-1:]:
            m.pretty_print()

        input_message = HumanMessage(content="i like the 49ers!")
        output = await graph.ainvoke({"messages": [input_message]}, config)
        for m in output['messages'][-1:]:
            m.pretty_print()

        summary = (await graph.aget_state(config)).values.get("summary","")

        input_message = HumanMessage(content="i like Nick Bosa, isn't he the highest paid defensive player?")
        output = await graph.ainvoke({"messages": [input_message]}, config)
        for m in output['messages'][-1:]:
            m.pretty_print()

if __name__ == "__main__":
    asyncio.run(main())