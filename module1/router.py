from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

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


class MathAssistantWithRouter:
    """
    Encapsulates a math assistant with conditional routing between LLM and tools.
    Uses tools_condition to route between tool execution and completion.
    """

    def __init__(self, model: str = "gpt-4o", tools: list = None):
        self.tools = tools or [multiply]
        self.llm = ChatOpenAI(model=model)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph = self._build_graph()

    def _tool_calling_llm(self, state: MessagesState):
        """Node that invokes the LLM with tool access."""
        return {"messages": [self.llm_with_tools.invoke(state["messages"])]}

    def _build_graph(self):
        """Constructs the LangGraph state machine with conditional routing."""
        builder = StateGraph(MessagesState)
        builder.add_node("tool_calling_llm", self._tool_calling_llm)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_edge(START, "tool_calling_llm")
        builder.add_conditional_edges(
            "tool_calling_llm",
            # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
            # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
            tools_condition,
        )
        builder.add_edge("tools", END)
        return builder.compile()

    def query(self, text: str):
        """Helper to invoke the graph with a single user message and print output."""
        messages = [HumanMessage(content=text)]
        result = self.graph.invoke({"messages": messages})
        for m in result['messages']:
            m.pretty_print()
        return result


def main():
    # Initialize the encapsulated assistant
    assistant = MathAssistantWithRouter()

    # Test basic conversation
    print("--- Testing standard greeting ---")
    assistant.query("Hello world.")

    # Test tool invocation
    print("\n--- Testing tool usage ---")
    assistant.query("Multiply 2 and 3")


if __name__ == "__main__":
    main()