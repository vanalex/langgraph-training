"""Mixins for common agent functionality."""

from typing import Any, Optional
from abc import ABC

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


class MCPMixin(ABC):
    """Mixin for MCP client functionality."""

    def initialize_mcp_client(self) -> Any:
        """Initialize MCP client.

        Returns:
            MCP client instance
        """
        if not hasattr(self, 'context'):
            raise AttributeError("MCPMixin requires 'context' attribute")

        if self.context.mcp_client is None:
            from mcp_server.mcp_client_utils import get_current_session
            self.context.mcp_client = get_current_session()

        return self.context.mcp_client

    async def get_prompt(self, name: str, arguments: Optional[dict] = None) -> str:
        """Get a prompt from MCP server.

        Args:
            name: Prompt name
            arguments: Prompt arguments

        Returns:
            Prompt text
        """
        client = self.initialize_mcp_client()
        result = await client.get_prompt(name, arguments=arguments)
        return result.messages[0].content.text


class LLMMixin(ABC):
    """Mixin for LLM functionality."""

    def get_llm(self) -> ChatOpenAI:
        """Get the LLM instance.

        Returns:
            ChatOpenAI instance
        """
        if not hasattr(self, 'context'):
            raise AttributeError("LLMMixin requires 'context' attribute")

        if self.context.llm is None:
            self.initialize_llm()

        return self.context.llm

    async def generate(self, messages: list, **kwargs) -> Any:
        """Generate a response from the LLM.

        Args:
            messages: List of messages
            **kwargs: Additional LLM parameters

        Returns:
            LLM response
        """
        llm = self.get_llm()
        return await llm.ainvoke(messages, **kwargs)

    async def generate_structured(self, messages: list, schema: type, **kwargs) -> Any:
        """Generate structured output from the LLM.

        Args:
            messages: List of messages
            schema: Pydantic schema for output
            **kwargs: Additional LLM parameters

        Returns:
            Structured output
        """
        llm = self.get_llm()
        structured_llm = llm.with_structured_output(schema)
        return await structured_llm.ainvoke(messages, **kwargs)


class SearchMixin(ABC):
    """Mixin for search functionality."""

    def initialize_search(self) -> TavilySearch:
        """Initialize search client.

        Returns:
            TavilySearch instance
        """
        if not hasattr(self, 'context'):
            raise AttributeError("SearchMixin requires 'context' attribute")

        if self.context.search_client is None:
            max_results = getattr(self.config, 'search_max_results', 3)
            self.context.search_client = TavilySearch(max_results=max_results)

        return self.context.search_client

    async def search(self, query: str) -> list:
        """Perform a search.

        Args:
            query: Search query

        Returns:
            List of search results
        """
        search_client = self.initialize_search()
        results = search_client.invoke({"query": query})
        # Handle both dict and list responses
        if isinstance(results, dict):
            return results.get("results", results)
        return results


class StateMixin(ABC):
    """Mixin for state management functionality."""

    def get_state_value(self, state: dict, key: str, default: Any = None) -> Any:
        """Get a value from state with default.

        Args:
            state: State dictionary
            key: Key to retrieve
            default: Default value

        Returns:
            State value or default
        """
        return state.get(key, default)

    def update_state_value(self, state: dict, key: str, value: Any) -> dict:
        """Update a state value.

        Args:
            state: State dictionary
            key: Key to update
            value: New value

        Returns:
            Updated state dict
        """
        return {key: value}

    def merge_state(self, state: dict, updates: dict) -> dict:
        """Merge updates into state.

        Args:
            state: Current state
            updates: Updates to merge

        Returns:
            Merged state dict
        """
        return {**state, **updates}

    def validate_state(self, state: dict, required_keys: list) -> bool:
        """Validate that state has required keys.

        Args:
            state: State to validate
            required_keys: List of required keys

        Returns:
            True if valid, raises ValueError otherwise
        """
        missing = [key for key in required_keys if key not in state]
        if missing:
            raise ValueError(f"State missing required keys: {missing}")
        return True
