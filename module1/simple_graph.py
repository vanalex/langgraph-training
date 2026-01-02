import random
from typing import Literal

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel


class State(BaseModel):
    """State schema for the mood graph."""
    graph_state: str


class MoodGraph:
    """
    Encapsulates a simple conditional graph that randomly routes between happy and sad moods.
    Demonstrates basic graph construction with nodes and conditional edges.
    """

    def __init__(self, random_seed: int = None):
        """
        Initialize the mood graph.

        Args:
            random_seed: Optional seed for reproducible random mood selection
        """
        if random_seed is not None:
            random.seed(random_seed)
        self.graph = self._build_graph()

    def _node_1(self, state: State):
        """First node that adds 'I am' to the state."""
        print("---Node 1---")
        return {"graph_state": state.graph_state + " I am"}

    def _node_2(self, state: State):
        """Happy mood node."""
        print("---Node 2---")
        return {"graph_state": state.graph_state + " happy!"}

    def _node_3(self, state: State):
        """Sad mood node."""
        print("---Node 3---")
        return {"graph_state": state.graph_state + " sad!"}

    def _decide_mood(self, state: State) -> Literal["node_2", "node_3"]:
        """
        Conditional edge function that randomly decides between happy and sad.

        Args:
            state: Current graph state

        Returns:
            Either "node_2" (happy) or "node_3" (sad)
        """
        # Often, we will use state to decide on the next node to visit
        # Here, let's just do a 50 / 50 split between nodes 2, 3
        if random.random() < 0.5:
            # 50% of the time, we return Node 2
            return "node_2"

        # 50% of the time, we return Node 3
        return "node_3"

    def _build_graph(self):
        """Constructs the mood selection graph."""
        builder = StateGraph(State)
        builder.add_node("node_1", self._node_1)
        builder.add_node("node_2", self._node_2)
        builder.add_node("node_3", self._node_3)

        # Logic
        builder.add_edge(START, "node_1")
        builder.add_conditional_edges("node_1", self._decide_mood)
        builder.add_edge("node_2", END)
        builder.add_edge("node_3", END)

        return builder.compile()

    def run(self, initial_message: str):
        """
        Execute the graph with an initial message.

        Args:
            initial_message: Starting text for the conversation

        Returns:
            Final state after graph execution
        """
        result = self.graph.invoke({"graph_state": initial_message})
        print(result)
        return result


def main():
    # Initialize the encapsulated mood graph
    mood_graph = MoodGraph()

    # Test the graph
    print("--- Testing mood graph ---")
    mood_graph.run("Hi, this is Lance.")


if __name__ == "__main__":
    main()



