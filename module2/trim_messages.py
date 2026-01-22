from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage, trim_messages
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph

load_dotenv()

messages = [AIMessage("Hi.", name="Bot", id="1")]
messages.append(HumanMessage("Hi.", name="Lance", id="2"))
messages.append(AIMessage("So you said you were researching ocean mammals?", name="Bot", id="3"))
messages.append(HumanMessage("Yes, I know about whales. But what others should I learn about?", name="Lance", id="4"))

for m in messages:
    m.pretty_print()

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-5-mini")
#result = llm.invoke(messages)
#pprint(result)

def chat_model_node(state: MessagesState):
    messages = trim_messages(
        state["messages"],
        max_tokens=100,
        strategy="last",
        token_counter=ChatOpenAI(model="gpt-4o"),
        allow_partial=False,
    )
    return {"messages": [llm.invoke(messages)]}

builder = StateGraph(MessagesState)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()

output = graph.invoke({'messages': messages})
for m in output['messages']:
    m.pretty_print()

messages.append(output['messages'][-1])
messages.append(HumanMessage(f"Tell me more about Orcas Live!", name="Lance"))

# Example of trimming messages
trim_messages(
            messages,
            max_tokens=100,
            strategy="last",
            token_counter=ChatOpenAI(model="gpt-4o"),
            allow_partial=False
        )

output = graph.invoke({'messages': messages})
for m in output['messages']:
    m.pretty_print()