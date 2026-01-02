from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph

load_dotenv()


def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int

    Returns:
        Product of a and b
    """
    return a * b


class MathAssistant:
    """
    Encapsulates the LLM configuration and LangGraph logic for a math-capable assistant.
    """
    def __init__(self, model: str = "gpt-5-nano"):
        self.llm = ChatOpenAI(model=model)
        self.llm_with_tools = self.llm.bind_tools([multiply])
        self.graph = self._build_graph()

    def _tool_calling_node(self, state: MessagesState):
        """Node that invokes the LLM with tool access."""
        response = self.llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def _build_graph(self):
        """Constructs the LangGraph state machine."""
        builder = StateGraph(MessagesState)
        builder.add_node("assistant", self._tool_calling_node)
        builder.add_edge(START, "assistant")
        builder.add_edge("assistant", END)
        return builder.compile()

    def query(self, text: str):
        """Helper to invoke the graph with a single user message and print output."""
        result = self.graph.invoke({"messages": [HumanMessage(content=text)]})
        for m in result['messages']:
            m.pretty_print()
        return result

def main():
    # Initialize the encapsulated assistant
    assistant = MathAssistant()

    # Test basic conversation
    print("--- Testing standard greeting ---")
    assistant.query("Hello!")

    # Test tool invocation
    print("\n--- Testing tool usage ---")
    assistant.query("Multiply 2 and 3")

if __name__ == "__main__":
    main()


