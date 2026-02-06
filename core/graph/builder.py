"""Graph builder for creating LangGraph StateGraphs."""

from typing import Dict, List, Callable, Any, Optional, Union
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


class GraphBuilder:
    """Fluent interface for building StateGraphs.

    Example:
        builder = GraphBuilder(MyState)
        builder.add_node("process", process_func)
               .add_edge(START, "process")
               .add_edge("process", END)
               .build()
    """

    def __init__(self, state_schema: type):
        """Initialize the graph builder.

        Args:
            state_schema: State schema class (TypedDict or MessagesState)
        """
        self.state_schema = state_schema
        self.graph = StateGraph(state_schema)
        self._nodes: Dict[str, Callable] = {}
        self._edges: List[tuple] = []
        self._conditional_edges: List[tuple] = []

    def add_node(self, name: str, func: Callable, **kwargs) -> "GraphBuilder":
        """Add a node to the graph.

        Args:
            name: Node name
            func: Node function
            **kwargs: Additional node parameters

        Returns:
            Self for chaining
        """
        self.graph.add_node(name, func, **kwargs)
        self._nodes[name] = func
        return self

    def add_edge(
        self,
        source: Union[str, type],
        target: Union[str, type]
    ) -> "GraphBuilder":
        """Add an edge between nodes.

        Args:
            source: Source node name or START
            target: Target node name or END

        Returns:
            Self for chaining
        """
        self.graph.add_edge(source, target)
        self._edges.append((source, target))
        return self

    def add_conditional_edge(
        self,
        source: str,
        condition: Callable,
        targets: Union[List[str], Dict[str, str]],
        **kwargs
    ) -> "GraphBuilder":
        """Add a conditional edge.

        Args:
            source: Source node name
            condition: Condition function
            targets: List of target names or mapping
            **kwargs: Additional edge parameters

        Returns:
            Self for chaining
        """
        self.graph.add_conditional_edges(source, condition, targets, **kwargs)
        self._conditional_edges.append((source, condition, targets))
        return self

    def add_fan_out(
        self,
        source: str,
        target: str,
        fan_function: Callable
    ) -> "GraphBuilder":
        """Add a fan-out pattern (map).

        Args:
            source: Source node name
            target: Target node name
            fan_function: Function that returns list of Send objects

        Returns:
            Self for chaining
        """
        self.add_conditional_edge(source, fan_function, [target])
        return self

    def set_entry_point(self, node: str) -> "GraphBuilder":
        """Set the entry point for the graph.

        Args:
            node: Entry node name

        Returns:
            Self for chaining
        """
        self.add_edge(START, node)
        return self

    def set_finish_point(self, node: str) -> "GraphBuilder":
        """Set the finish point for the graph.

        Args:
            node: Final node name

        Returns:
            Self for chaining
        """
        self.add_edge(node, END)
        return self

    def build(self) -> StateGraph:
        """Build and return the graph.

        Returns:
            Constructed StateGraph
        """
        return self.graph

    def compile(self, **kwargs) -> Any:
        """Build and compile the graph.

        Args:
            **kwargs: Compilation arguments (checkpointer, interrupt_before, etc.)

        Returns:
            Compiled graph
        """
        return self.graph.compile(**kwargs)

    def get_nodes(self) -> Dict[str, Callable]:
        """Get all registered nodes.

        Returns:
            Dictionary of node names to functions
        """
        return self._nodes.copy()

    def get_edges(self) -> List[tuple]:
        """Get all registered edges.

        Returns:
            List of edge tuples
        """
        return self._edges.copy()

    def get_conditional_edges(self) -> List[tuple]:
        """Get all conditional edges.

        Returns:
            List of conditional edge tuples
        """
        return self._conditional_edges.copy()

    def validate(self) -> List[str]:
        """Validate the graph structure.

        Returns:
            List of validation warnings
        """
        warnings = []

        # Check if entry point is defined
        has_entry = any(source == START for source, _ in self._edges)
        if not has_entry:
            warnings.append("No entry point defined (START edge)")

        # Check if exit point is defined
        has_exit = any(target == END for _, target in self._edges)
        if not has_exit:
            warnings.append("No exit point defined (END edge)")

        # Check for orphaned nodes
        connected_nodes = set()
        for source, target in self._edges:
            if source != START and isinstance(source, str):
                connected_nodes.add(source)
            if target != END and isinstance(target, str):
                connected_nodes.add(target)

        orphaned = set(self._nodes.keys()) - connected_nodes
        if orphaned:
            warnings.append(f"Orphaned nodes (not connected): {orphaned}")

        return warnings

    def __repr__(self) -> str:
        """String representation."""
        return f"GraphBuilder(nodes={len(self._nodes)}, edges={len(self._edges)})"
