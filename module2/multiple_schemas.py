from typing import TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph


class OverallState(TypedDict):
    foo: int


class PrivateState(TypedDict):
    baz: str

def node_1(state: OverallState):
    print("---Node 1---")
    return {"baz": state["foo"] + 1}

def node_2(state: PrivateState):
    print("---Node 2---")
    return {"foo": state["baz"] + 1}

buildder = StateGraph(OverallState)
buildder.add_node("node_1", node_1)
buildder.add_node("node_2", node_2)

buildder.add_edge(START, "node_1")
buildder.add_edge("node_1", "node_2")
buildder.add_edge("node_2", END)

graph = buildder.compile()

result = graph.invoke({"foo": 1})
print(result)