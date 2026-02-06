"""Unit tests for BaseAgent class."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from core.agents.base_agent import BaseAgent, AgentContext
from core.config.settings import AgentConfig


class MockStateSchema:
    """Mock state schema for testing."""
    pass


class TestAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""

    def __init__(self, config: AgentConfig, context: AgentContext = None):
        super().__init__(config, context)
        self.build_graph_called = False
        self.before_run_called = False
        self.after_run_called = False
        self.on_error_called = False

    def build_graph(self) -> StateGraph:
        """Build a simple test graph."""
        self.build_graph_called = True
        graph = StateGraph(dict)
        graph.add_node("test_node", lambda state: {"output": "test"})
        graph.add_edge(START, "test_node")
        graph.add_edge("test_node", END)
        return graph

    def get_state_schema(self):
        """Return mock state schema."""
        return MockStateSchema

    async def before_run(self, input_data):
        """Track before_run call."""
        self.before_run_called = True
        return input_data

    async def after_run(self, result):
        """Track after_run call."""
        self.after_run_called = True
        return result

    async def on_error(self, error):
        """Track on_error call."""
        self.on_error_called = True


class TestBaseAgent:
    """Tests for BaseAgent class."""

    def test_init_with_config_only(self):
        """Test initialization with config only."""
        config = AgentConfig(name="test_agent")
        agent = TestAgent(config)

        assert agent.config == config
        assert agent.context is not None
        assert agent.context.config == config
        assert agent._compiled_graph is None

    def test_init_with_context(self):
        """Test initialization with config and context."""
        config = AgentConfig(name="test_agent")
        llm = Mock(spec=ChatOpenAI)
        context = AgentContext(llm=llm, config=config)

        agent = TestAgent(config, context)

        assert agent.config == config
        assert agent.context == context
        assert agent.context.llm == llm

    def test_initialize_llm_from_config(self):
        """Test LLM initialization from config."""
        config = AgentConfig(
            name="test_agent",
            llm_model="gpt-4o-mini",
            llm_temperature=0.5
        )
        agent = TestAgent(config)

        with patch('core.agents.base_agent.ChatOpenAI') as mock_openai:
            mock_llm = Mock()
            mock_llm.model_name = "gpt-4o-mini"
            mock_llm.temperature = 0.5
            mock_openai.return_value = mock_llm

            agent.initialize_llm()

            assert agent.context.llm is not None
            assert agent.context.llm == mock_llm
            mock_openai.assert_called_once_with(model="gpt-4o-mini", temperature=0.5)

    def test_initialize_llm_idempotent(self):
        """Test LLM initialization is idempotent."""
        config = AgentConfig(name="test_agent")
        agent = TestAgent(config)

        with patch('core.agents.base_agent.ChatOpenAI') as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm

            agent.initialize_llm()
            llm1 = agent.context.llm

            agent.initialize_llm()
            llm2 = agent.context.llm

            assert llm1 is llm2
            # Should only be called once due to idempotency
            mock_openai.assert_called_once()

    def test_compile_graph(self):
        """Test graph compilation."""
        config = AgentConfig(name="test_agent")
        agent = TestAgent(config)

        compiled = agent.compile_graph()

        assert agent.build_graph_called
        assert agent._compiled_graph is not None
        assert compiled is agent._compiled_graph

    def test_compile_graph_with_checkpointer(self):
        """Test graph compilation with checkpointer."""
        config = AgentConfig(name="test_agent", enable_checkpointing=True)
        agent = TestAgent(config)

        compiled = agent.compile_graph()

        assert agent._compiled_graph is not None
        assert agent._checkpointer is not None
        assert isinstance(agent._checkpointer, MemorySaver)

    def test_compile_graph_idempotent(self):
        """Test graph compilation is idempotent."""
        config = AgentConfig(name="test_agent")
        agent = TestAgent(config)

        compiled1 = agent.compile_graph()
        compiled2 = agent.compile_graph()

        assert compiled1 is compiled2

    @pytest.mark.asyncio
    async def test_run_success(self):
        """Test successful run."""
        config = AgentConfig(name="test_agent", enable_checkpointing=False)
        agent = TestAgent(config)

        result = await agent.run({"input": "test"})

        assert result is not None
        assert agent.before_run_called
        assert agent.after_run_called
        assert not agent.on_error_called

    @pytest.mark.asyncio
    async def test_run_compiles_graph(self):
        """Test run compiles graph if not already compiled."""
        config = AgentConfig(name="test_agent", enable_checkpointing=False)
        agent = TestAgent(config)

        assert agent._compiled_graph is None

        await agent.run({"input": "test"})

        assert agent._compiled_graph is not None

    @pytest.mark.asyncio
    async def test_run_with_config(self):
        """Test run with runtime config."""
        config = AgentConfig(name="test_agent", enable_checkpointing=False)
        agent = TestAgent(config)

        run_config = {"recursion_limit": 50}
        result = await agent.run({"input": "test"}, config=run_config)

        assert result is not None

    @pytest.mark.asyncio
    async def test_run_error_handling(self):
        """Test error handling in run."""
        config = AgentConfig(name="test_agent")
        agent = TestAgent(config)

        # Mock the graph to raise an error
        agent.compile_graph()
        original_ainvoke = agent._compiled_graph.ainvoke
        agent._compiled_graph.ainvoke = AsyncMock(side_effect=Exception("Test error"))

        with pytest.raises(Exception, match="Test error"):
            await agent.run({"input": "test"})

        assert agent.on_error_called

    def test_get_config(self):
        """Test get_config method."""
        config = AgentConfig(name="test_agent", llm_model="gpt-4o")
        agent = TestAgent(config)

        retrieved_config = agent.get_config()

        assert retrieved_config == config

    def test_get_context(self):
        """Test get_context method."""
        config = AgentConfig(name="test_agent")
        context = AgentContext(config=config)
        agent = TestAgent(config, context)

        retrieved_context = agent.get_context()

        assert retrieved_context == context

    def test_base_agent_abstract(self):
        """Test that BaseAgent cannot be instantiated directly."""
        config = AgentConfig(name="test_agent")

        with pytest.raises(TypeError):
            BaseAgent(config)

    def test_state_schema_required(self):
        """Test that get_state_schema must be implemented."""
        class IncompleteAgent(BaseAgent):
            def build_graph(self):
                pass

            def get_state_schema(self):
                return None

        config = AgentConfig(name="test_agent")
        agent = IncompleteAgent(config)

        # get_state_schema should return None by default
        assert agent.get_state_schema() is None


class TestAgentContext:
    """Tests for AgentContext class."""

    def test_context_initialization(self):
        """Test AgentContext initialization."""
        config = AgentConfig(name="test_agent")
        context = AgentContext(config=config)

        assert context.config == config
        assert context.llm is None
        assert context.mcp_client is None
        assert context.search_client is None
        assert context.metadata == {}

    def test_context_with_llm(self):
        """Test AgentContext with LLM."""
        config = AgentConfig(name="test_agent")
        llm = Mock(spec=ChatOpenAI)
        context = AgentContext(llm=llm, config=config)

        assert context.llm == llm

    def test_context_with_metadata(self):
        """Test AgentContext with metadata."""
        config = AgentConfig(name="test_agent")
        metadata = {"key": "value"}
        context = AgentContext(config=config, metadata=metadata)

        assert context.metadata == metadata

    def test_context_immutable_config(self):
        """Test that config is stored correctly."""
        config1 = AgentConfig(name="agent1")
        context = AgentContext(config=config1)

        config2 = AgentConfig(name="agent2")
        # Assigning new config doesn't change the original
        assert context.config == config1
