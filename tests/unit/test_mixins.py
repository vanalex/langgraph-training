"""Unit tests for agent mixins."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from core.agents.mixins import MCPMixin, LLMMixin, SearchMixin, StateMixin
from core.agents.base_agent import AgentContext
from core.config.settings import AgentConfig


class MockMCPResult:
    """Mock MCP result."""

    def __init__(self, text: str):
        self.messages = [Mock(content=Mock(text=text))]


class TestMCPAgent(MCPMixin):
    """Test agent with MCP mixin."""

    def __init__(self, context: AgentContext):
        self.context = context


class TestLLMAgent(LLMMixin):
    """Test agent with LLM mixin."""

    def __init__(self, context: AgentContext):
        self.context = context

    def initialize_llm(self):
        """Initialize LLM for testing."""
        if self.context.llm is None:
            config = self.context.config
            self.context.llm = ChatOpenAI(
                model=config.llm_model,
                temperature=config.llm_temperature
            )


class TestSearchAgent(SearchMixin):
    """Test agent with search mixin."""

    def __init__(self, context: AgentContext):
        self.context = context
        self.config = context.config


class TestStateMixinAgent(StateMixin):
    """Test agent with state mixin."""

    pass


class TestMCPMixin:
    """Tests for MCPMixin."""

    @pytest.mark.asyncio
    async def test_initialize_mcp_client(self):
        """Test MCP client initialization."""
        config = AgentConfig(mcp_enabled=True)
        context = AgentContext(config=config)
        agent = TestMCPAgent(context)

        with patch("mcp_server.mcp_client_utils.get_current_session") as mock_get_session:
            mock_client = AsyncMock()
            mock_get_session.return_value = mock_client

            client = agent.initialize_mcp_client()

            assert client == mock_client
            assert context.mcp_client == mock_client

    @pytest.mark.asyncio
    async def test_initialize_mcp_client_cached(self):
        """Test MCP client initialization is cached."""
        config = AgentConfig(mcp_enabled=True)
        mock_client = AsyncMock()
        context = AgentContext(config=config, mcp_client=mock_client)
        agent = TestMCPAgent(context)

        client = agent.initialize_mcp_client()

        assert client == mock_client

    @pytest.mark.asyncio
    async def test_get_prompt(self):
        """Test getting prompt from MCP."""
        config = AgentConfig(mcp_enabled=True)
        mock_client = AsyncMock()
        mock_client.get_prompt = AsyncMock(
            return_value=MockMCPResult("Test prompt text")
        )
        context = AgentContext(config=config, mcp_client=mock_client)
        agent = TestMCPAgent(context)

        result = await agent.get_prompt("test-prompt", {"arg": "value"})

        assert result == "Test prompt text"
        mock_client.get_prompt.assert_called_once_with(
            "test-prompt", arguments={"arg": "value"}
        )

    @pytest.mark.asyncio
    async def test_get_prompt_no_arguments(self):
        """Test getting prompt without arguments."""
        config = AgentConfig(mcp_enabled=True)
        mock_client = AsyncMock()
        mock_client.get_prompt = AsyncMock(
            return_value=MockMCPResult("Test prompt")
        )
        context = AgentContext(config=config, mcp_client=mock_client)
        agent = TestMCPAgent(context)

        result = await agent.get_prompt("test-prompt")

        assert result == "Test prompt"
        mock_client.get_prompt.assert_called_once_with(
            "test-prompt", arguments=None
        )

    def test_mcp_mixin_requires_context(self):
        """Test MCPMixin requires context attribute."""
        agent = type("BadAgent", (MCPMixin,), {})()

        with pytest.raises(AttributeError, match="MCPMixin requires 'context' attribute"):
            agent.initialize_mcp_client()


class TestLLMMixin:
    """Tests for LLMMixin."""

    def test_get_llm_existing(self):
        """Test getting existing LLM."""
        config = AgentConfig(llm_model="gpt-4o")
        llm = Mock(spec=ChatOpenAI)
        context = AgentContext(config=config, llm=llm)
        agent = TestLLMAgent(context)

        result = agent.get_llm()

        assert result == llm

    def test_get_llm_initialize(self):
        """Test LLM initialization when not present."""
        config = AgentConfig(llm_model="gpt-4o")
        context = AgentContext(config=config)
        agent = TestLLMAgent(context)

        result = agent.get_llm()

        assert result is not None
        assert isinstance(result, ChatOpenAI)
        assert result.model_name == "gpt-4o"

    @pytest.mark.asyncio
    async def test_generate(self):
        """Test generating response from LLM."""
        config = AgentConfig(llm_model="gpt-4o")
        llm = Mock(spec=ChatOpenAI)
        llm.ainvoke = AsyncMock(return_value="Test response")
        context = AgentContext(config=config, llm=llm)
        agent = TestLLMAgent(context)

        messages = [{"role": "user", "content": "test"}]
        result = await agent.generate(messages)

        assert result == "Test response"
        llm.ainvoke.assert_called_once_with(messages)

    @pytest.mark.asyncio
    async def test_generate_with_kwargs(self):
        """Test generating with additional kwargs."""
        config = AgentConfig(llm_model="gpt-4o")
        llm = Mock(spec=ChatOpenAI)
        llm.ainvoke = AsyncMock(return_value="Test response")
        context = AgentContext(config=config, llm=llm)
        agent = TestLLMAgent(context)

        messages = [{"role": "user", "content": "test"}]
        result = await agent.generate(messages, temperature=0.5)

        assert result == "Test response"
        llm.ainvoke.assert_called_once_with(messages, temperature=0.5)

    @pytest.mark.asyncio
    async def test_generate_structured(self):
        """Test generating structured output."""
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            name: str
            value: int

        config = AgentConfig(llm_model="gpt-4o")
        llm = Mock(spec=ChatOpenAI)
        structured_llm = Mock()
        structured_llm.ainvoke = AsyncMock(
            return_value=TestSchema(name="test", value=42)
        )
        llm.with_structured_output = Mock(return_value=structured_llm)
        context = AgentContext(config=config, llm=llm)
        agent = TestLLMAgent(context)

        messages = [{"role": "user", "content": "test"}]
        result = await agent.generate_structured(messages, schema=TestSchema)

        assert isinstance(result, TestSchema)
        assert result.name == "test"
        assert result.value == 42
        llm.with_structured_output.assert_called_once_with(TestSchema)
        structured_llm.ainvoke.assert_called_once_with(messages)

    def test_llm_mixin_requires_context(self):
        """Test LLMMixin requires context attribute."""
        agent = type("BadAgent", (LLMMixin,), {})()

        with pytest.raises(AttributeError, match="LLMMixin requires 'context' attribute"):
            agent.get_llm()


class TestSearchMixin:
    """Tests for SearchMixin."""

    def test_initialize_search(self):
        """Test search client initialization."""
        config = AgentConfig(search_enabled=True, search_max_results=5)
        context = AgentContext(config=config)
        agent = TestSearchAgent(context)

        with patch("core.agents.mixins.TavilySearch") as mock_tavily:
            mock_search = Mock()
            mock_tavily.return_value = mock_search

            result = agent.initialize_search()

            assert result == mock_search
            assert context.search_client == mock_search
            mock_tavily.assert_called_once_with(max_results=5)

    def test_initialize_search_default_max_results(self):
        """Test search initialization with default max_results."""
        config = AgentConfig(search_enabled=True)
        context = AgentContext(config=config)
        agent = TestSearchAgent(context)

        with patch("core.agents.mixins.TavilySearch") as mock_tavily:
            mock_search = Mock()
            mock_tavily.return_value = mock_search

            agent.initialize_search()

            mock_tavily.assert_called_once_with(max_results=3)

    def test_initialize_search_cached(self):
        """Test search client initialization is cached."""
        config = AgentConfig(search_enabled=True)
        mock_search = Mock()
        context = AgentContext(config=config, search_client=mock_search)
        agent = TestSearchAgent(context)

        result = agent.initialize_search()

        assert result == mock_search

    @pytest.mark.asyncio
    async def test_search(self):
        """Test performing search."""
        config = AgentConfig(search_enabled=True)
        mock_search = Mock()
        mock_search.invoke = Mock(return_value={"results": [{"title": "Test"}]})
        context = AgentContext(config=config, search_client=mock_search)
        agent = TestSearchAgent(context)

        results = await agent.search("test query")

        assert results == [{"title": "Test"}]
        mock_search.invoke.assert_called_once_with({"query": "test query"})

    @pytest.mark.asyncio
    async def test_search_no_results_key(self):
        """Test search when results not in nested dict."""
        config = AgentConfig(search_enabled=True)
        mock_search = Mock()
        mock_results = [{"title": "Test"}]
        mock_search.invoke = Mock(return_value=mock_results)
        context = AgentContext(config=config, search_client=mock_search)
        agent = TestSearchAgent(context)

        results = await agent.search("test query")

        assert results == mock_results

    def test_search_mixin_requires_context(self):
        """Test SearchMixin requires context attribute."""
        agent = type("BadAgent", (SearchMixin,), {})()

        with pytest.raises(AttributeError, match="SearchMixin requires 'context' attribute"):
            agent.initialize_search()


class TestStateMixin:
    """Tests for StateMixin."""

    def test_get_state_value(self):
        """Test getting state value."""
        agent = TestStateMixinAgent()
        state = {"key1": "value1", "key2": 42}

        result = agent.get_state_value(state, "key1")

        assert result == "value1"

    def test_get_state_value_default(self):
        """Test getting state value with default."""
        agent = TestStateMixinAgent()
        state = {"key1": "value1"}

        result = agent.get_state_value(state, "missing", default="default_value")

        assert result == "default_value"

    def test_get_state_value_missing_no_default(self):
        """Test getting missing state value without default."""
        agent = TestStateMixinAgent()
        state = {"key1": "value1"}

        result = agent.get_state_value(state, "missing")

        assert result is None

    def test_update_state_value(self):
        """Test updating state value."""
        agent = TestStateMixinAgent()
        state = {"key1": "value1"}

        result = agent.update_state_value(state, "key2", "value2")

        assert result == {"key2": "value2"}

    def test_merge_state(self):
        """Test merging state."""
        agent = TestStateMixinAgent()
        state = {"key1": "value1", "key2": "value2"}
        updates = {"key2": "new_value2", "key3": "value3"}

        result = agent.merge_state(state, updates)

        assert result == {
            "key1": "value1",
            "key2": "new_value2",
            "key3": "value3",
        }

    def test_validate_state_success(self):
        """Test state validation success."""
        agent = TestStateMixinAgent()
        state = {"key1": "value1", "key2": 42, "key3": []}

        result = agent.validate_state(state, ["key1", "key2"])

        assert result is True

    def test_validate_state_failure(self):
        """Test state validation failure."""
        agent = TestStateMixinAgent()
        state = {"key1": "value1"}

        with pytest.raises(ValueError, match="State missing required keys: \\['key2', 'key3'\\]"):
            agent.validate_state(state, ["key1", "key2", "key3"])

    def test_validate_state_empty_required(self):
        """Test state validation with no required keys."""
        agent = TestStateMixinAgent()
        state = {}

        result = agent.validate_state(state, [])

        assert result is True
