"""Unit tests for graph building utilities."""

import pytest
from unittest.mock import Mock
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from core.graph.builder import GraphBuilder
from core.graph.nodes import NodeRegistry, register_node, get_node, list_nodes, _global_registry
from core.graph.edges import EdgeBuilder


class TestStateSchema(TypedDict):
    """Test state schema."""
    count: int
    message: str


def sample_node_1(state: TestStateSchema):
    """Test node function 1."""
    return {"count": state["count"] + 1}


def sample_node_2(state: TestStateSchema):
    """Test node function 2."""
    return {"message": "processed"}


class TestGraphBuilder:
    """Tests for GraphBuilder class."""

    def test_init(self):
        """Test GraphBuilder initialization."""
        builder = GraphBuilder(TestStateSchema)

        assert builder.state_schema == TestStateSchema
        assert isinstance(builder.graph, StateGraph)

    def test_add_node(self):
        """Test adding a node."""
        builder = GraphBuilder(TestStateSchema)

        result = builder.add_node("test_node", sample_node_1)

        assert result is builder  # Test chaining
        # Verify node was added to graph
        assert "test_node" in builder.graph.nodes

    def test_add_multiple_nodes(self):
        """Test adding multiple nodes."""
        builder = GraphBuilder(TestStateSchema)

        builder.add_node("node1", sample_node_1)
        builder.add_node("node2", sample_node_2)

        assert "node1" in builder.graph.nodes
        assert "node2" in builder.graph.nodes

    def test_add_edge(self):
        """Test adding an edge."""
        builder = GraphBuilder(TestStateSchema)
        builder.add_node("node1", sample_node_1)
        builder.add_node("node2", sample_node_2)

        result = builder.add_edge("node1", "node2")

        assert result is builder  # Test chaining

    def test_add_edge_from_start(self):
        """Test adding edge from START."""
        builder = GraphBuilder(TestStateSchema)
        builder.add_node("node1", sample_node_1)

        result = builder.add_edge(START, "node1")

        assert result is builder

    def test_add_edge_to_end(self):
        """Test adding edge to END."""
        builder = GraphBuilder(TestStateSchema)
        builder.add_node("node1", sample_node_1)

        result = builder.add_edge("node1", END)

        assert result is builder

    def test_add_conditional_edge(self):
        """Test adding conditional edge."""
        builder = GraphBuilder(TestStateSchema)
        builder.add_node("router", sample_node_1)
        builder.add_node("target1", sample_node_2)

        def router_func(state):
            return "target1"

        result = builder.add_conditional_edge(
            "router",
            router_func,
            ["target1", END]
        )

        assert result is builder

    def test_add_conditional_edge_with_dict(self):
        """Test adding conditional edge with path mapping."""
        builder = GraphBuilder(TestStateSchema)
        builder.add_node("router", sample_node_1)
        builder.add_node("target1", sample_node_2)

        def router_func(state):
            return "continue"

        result = builder.add_conditional_edge(
            "router",
            router_func,
            {"continue": "target1", "end": END}
        )

        assert result is builder

    def test_build(self):
        """Test building the graph."""
        builder = GraphBuilder(TestStateSchema)
        builder.add_node("node1", sample_node_1)
        builder.add_edge(START, "node1")
        builder.add_edge("node1", END)

        graph = builder.build()

        assert graph is builder.graph

    def test_chaining(self):
        """Test method chaining."""
        builder = GraphBuilder(TestStateSchema)

        graph = (builder
                 .add_node("node1", sample_node_1)
                 .add_node("node2", sample_node_2)
                 .add_edge(START, "node1")
                 .add_edge("node1", "node2")
                 .add_edge("node2", END)
                 .build())

        assert isinstance(graph, StateGraph)
        assert "node1" in builder.graph.nodes
        assert "node2" in builder.graph.nodes

    def test_build_empty_graph(self):
        """Test building empty graph."""
        builder = GraphBuilder(TestStateSchema)

        graph = builder.build()

        assert graph is not None

    def test_multiple_builds(self):
        """Test multiple builds return same graph."""
        builder = GraphBuilder(TestStateSchema)
        builder.add_node("node1", sample_node_1)

        graph1 = builder.build()
        graph2 = builder.build()

        assert graph1 is graph2


class TestNodeRegistry:
    """Tests for NodeRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        _global_registry._nodes.clear()

    def test_register_node_basic(self):
        """Test basic node registration."""
        @register_node("test_node")
        def my_node(state):
            return state

        node_info = get_node("test_node")
        assert node_info is not None
        assert node_info["name"] == "test_node"
        assert node_info["func"] == my_node

    def test_register_node_with_metadata(self):
        """Test node registration with metadata."""
        @register_node("test_node", description="Test node", tags=["test", "demo"])
        def my_node(state):
            return state

        node_info = get_node("test_node")
        assert node_info["description"] == "Test node"
        assert node_info["tags"] == ["test", "demo"]

    def test_register_multiple_nodes(self):
        """Test registering multiple nodes."""
        @register_node("node1")
        def node_1(state):
            return state

        @register_node("node2")
        def node_2(state):
            return state

        assert get_node("node1") is not None
        assert get_node("node2") is not None

    def test_get_nonexistent_node(self):
        """Test getting nonexistent node returns None."""
        result = get_node("nonexistent")
        assert result is None

    def test_list_nodes_empty(self):
        """Test listing nodes when registry is empty."""
        nodes = list_nodes()
        assert nodes == []

    def test_list_nodes(self):
        """Test listing registered nodes."""
        @register_node("node1", tags=["tag1"])
        def node_1(state):
            return state

        @register_node("node2", tags=["tag2"])
        def node_2(state):
            return state

        nodes = list_nodes()
        assert len(nodes) == 2
        names = [n["name"] for n in nodes]
        assert "node1" in names
        assert "node2" in names

    def test_list_nodes_by_tag(self):
        """Test listing nodes filtered by tag."""
        @register_node("node1", tags=["processing"])
        def node_1(state):
            return state

        @register_node("node2", tags=["routing"])
        def node_2(state):
            return state

        @register_node("node3", tags=["processing", "routing"])
        def node_3(state):
            return state

        processing_nodes = list_nodes(tag="processing")
        assert len(processing_nodes) == 2
        names = [n["name"] for n in processing_nodes]
        assert "node1" in names
        assert "node3" in names

    def test_register_duplicate_name(self):
        """Test registering duplicate name overwrites."""
        @register_node("test_node")
        def node_1(state):
            return {"version": 1}

        @register_node("test_node")
        def node_2(state):
            return {"version": 2}

        node_info = get_node("test_node")
        assert node_info["func"] == node_2


class TestEdgeBuilder:
    """Tests for EdgeBuilder utilities."""

    def test_create_router_with_conditions(self):
        """Test creating router with condition functions."""
        conditions = {
            "continue": lambda s: s["count"] < 10,
            END: lambda s: s["count"] >= 10
        }

        router = EdgeBuilder.create_router(conditions)

        assert router({"count": 5}) == "continue"
        assert router({"count": 15}) == END

    def test_create_router_first_match(self):
        """Test router returns first matching condition."""
        conditions = {
            "path1": lambda s: s["count"] > 0,
            "path2": lambda s: s["count"] > 5,
            END: lambda s: True
        }

        router = EdgeBuilder.create_router(conditions)

        # Should match path1, not path2, even though both conditions are true
        assert router({"count": 7}) == "path1"

    def test_create_router_default(self):
        """Test router with default path."""
        conditions = {
            "path1": lambda s: s["count"] > 100
        }

        router = EdgeBuilder.create_router(conditions, default=END)

        assert router({"count": 5}) == END

    def test_create_router_no_match_no_default(self):
        """Test router raises error when no match and no default."""
        conditions = {
            "path1": lambda s: s["count"] > 100
        }

        router = EdgeBuilder.create_router(conditions)

        with pytest.raises(ValueError, match="No matching condition"):
            router({"count": 5})

    def test_create_field_router(self):
        """Test creating router based on field value."""
        router = EdgeBuilder.create_field_router(
            "status",
            {"active": "process", "inactive": END}
        )

        assert router({"status": "active"}) == "process"
        assert router({"status": "inactive"}) == END

    def test_create_field_router_missing_field(self):
        """Test field router with missing field uses default."""
        router = EdgeBuilder.create_field_router(
            "status",
            {"active": "process"},
            default=END
        )

        assert router({}) == END

    def test_create_field_router_missing_mapping(self):
        """Test field router with unmapped value uses default."""
        router = EdgeBuilder.create_field_router(
            "status",
            {"active": "process"},
            default=END
        )

        assert router({"status": "unknown"}) == END

    def test_create_field_router_no_default_raises(self):
        """Test field router raises error without default for unmapped value."""
        router = EdgeBuilder.create_field_router(
            "status",
            {"active": "process"}
        )

        with pytest.raises(ValueError, match="No path found"):
            router({"status": "unknown"})

    def test_create_count_router(self):
        """Test creating router based on count threshold."""
        router = EdgeBuilder.create_count_router(
            "iterations",
            max_count=5,
            continue_target="process",
            end_target=END
        )

        assert router({"iterations": 3}) == "process"
        assert router({"iterations": 5}) == END
        assert router({"iterations": 7}) == END

    def test_create_boolean_router(self):
        """Test creating boolean router."""
        router = EdgeBuilder.create_boolean_router(
            "should_continue",
            true_path="continue",
            false_path=END
        )

        assert router({"should_continue": True}) == "continue"
        assert router({"should_continue": False}) == END

    def test_create_boolean_router_missing_field(self):
        """Test boolean router with missing field defaults to False."""
        router = EdgeBuilder.create_boolean_router(
            "should_continue",
            true_path="continue",
            false_path=END
        )

        assert router({}) == END

    def test_create_multi_field_router(self):
        """Test router based on multiple fields."""
        def condition(state):
            return state.get("count", 0) < 10 and state.get("status") == "active"

        conditions = {
            "continue": condition,
            END: lambda s: True
        }

        router = EdgeBuilder.create_router(conditions)

        assert router({"count": 5, "status": "active"}) == "continue"
        assert router({"count": 15, "status": "active"}) == END
        assert router({"count": 5, "status": "inactive"}) == END


class TestGraphBuilderIntegration:
    """Integration tests for graph building."""

    def test_build_complete_graph(self):
        """Test building a complete working graph."""
        def start_node(state):
            return {"count": 0, "message": "started"}

        def increment_node(state):
            return {"count": state["count"] + 1}

        def router(state):
            return "continue" if state["count"] < 3 else "end"

        builder = GraphBuilder(TestStateSchema)
        graph = (builder
                 .add_node("start", start_node)
                 .add_node("increment", increment_node)
                 .add_edge(START, "start")
                 .add_conditional_edge("start", router, {"continue": "increment", "end": END})
                 .add_edge("increment", "start")
                 .build())

        assert graph is not None
        assert "start" in builder.graph.nodes
        assert "increment" in builder.graph.nodes

    def test_build_linear_graph(self):
        """Test building linear graph."""
        builder = GraphBuilder(TestStateSchema)
        graph = (builder
                 .add_node("node1", sample_node_1)
                 .add_node("node2", sample_node_2)
                 .add_node("node3", sample_node_1)
                 .add_edge(START, "node1")
                 .add_edge("node1", "node2")
                 .add_edge("node2", "node3")
                 .add_edge("node3", END)
                 .build())

        assert "node1" in builder.graph.nodes
        assert "node2" in builder.graph.nodes
        assert "node3" in builder.graph.nodes

    def test_build_branching_graph(self):
        """Test building branching graph."""
        def router(state):
            return "path1" if state.get("count", 0) < 5 else "path2"

        builder = GraphBuilder(TestStateSchema)
        graph = (builder
                 .add_node("start", sample_node_1)
                 .add_node("path1", sample_node_2)
                 .add_node("path2", sample_node_1)
                 .add_edge(START, "start")
                 .add_conditional_edge("start", router, ["path1", "path2"])
                 .add_edge("path1", END)
                 .add_edge("path2", END)
                 .build())

        assert "start" in builder.graph.nodes
        assert "path1" in builder.graph.nodes
        assert "path2" in builder.graph.nodes
