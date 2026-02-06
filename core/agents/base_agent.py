"""Base agent class with common functionality."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
import asyncio

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from core.config.settings import AgentConfig


@dataclass
class AgentContext:
    """Context for agent execution."""
    llm: Optional[ChatOpenAI] = None
    mcp_client: Optional[Any] = None
    search_client: Optional[Any] = None
    config: Optional[AgentConfig] = None
    checkpointer: Optional[MemorySaver] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Base class for all LangGraph agents.

    Provides common functionality:
    - Configuration management
    - LLM initialization
    - MCP client integration
    - Graph building interface
    - State management
    - Lifecycle hooks
    """

    def __init__(self, config: AgentConfig, context: Optional[AgentContext] = None):
        """Initialize the agent.

        Args:
            config: Agent configuration
            context: Execution context with dependencies
        """
        self.config = config
        self.context = context or AgentContext(config=config)
        self.graph: Optional[StateGraph] = None
        self._compiled_graph = None
        self._checkpointer = None

    @abstractmethod
    def build_graph(self) -> StateGraph:
        """Build the agent's graph.

        Returns:
            StateGraph: The constructed graph
        """
        pass

    @abstractmethod
    def get_state_schema(self) -> type:
        """Get the state schema for this agent.

        Returns:
            type: State schema class
        """
        pass

    def initialize_llm(self) -> ChatOpenAI:
        """Initialize the LLM client.

        Returns:
            ChatOpenAI: Configured LLM client
        """
        if self.context.llm is None:
            self.context.llm = ChatOpenAI(
                model=self.config.llm_model,
                temperature=self.config.llm_temperature
            )
        return self.context.llm

    def initialize_checkpointer(self) -> MemorySaver:
        """Initialize the checkpointer.

        Returns:
            MemorySaver: Checkpointer instance
        """
        if self.context.checkpointer is None:
            self.context.checkpointer = MemorySaver()
        return self.context.checkpointer

    def compile_graph(self, **kwargs) -> Any:
        """Compile the graph with optional checkpointing.

        Args:
            **kwargs: Additional arguments for compilation

        Returns:
            Compiled graph
        """
        if self._compiled_graph is not None:
            return self._compiled_graph

        if self.graph is None:
            self.graph = self.build_graph()

        compile_kwargs = {}

        if self.config.enable_checkpointing:
            self._checkpointer = self.initialize_checkpointer()
            compile_kwargs["checkpointer"] = self._checkpointer

        compile_kwargs.update(kwargs)

        self._compiled_graph = self.graph.compile(**compile_kwargs)
        return self._compiled_graph

    async def run(self, input_data: Dict[str, Any], config: Optional[Dict] = None) -> Dict[str, Any]:
        """Run the agent asynchronously.

        Args:
            input_data: Input data for the agent
            config: Runtime configuration

        Returns:
            Dict containing the final state
        """
        try:
            if self._compiled_graph is None:
                self.compile_graph()

            input_data = await self.before_run(input_data)
            result = await self._compiled_graph.ainvoke(input_data, config=config)
            result = await self.after_run(result)
            return result
        except Exception as e:
            await self.on_error(e)
            raise

    async def stream(self, input_data: Dict[str, Any], config: Optional[Dict] = None):
        """Stream agent execution.

        Args:
            input_data: Input data for the agent
            config: Runtime configuration

        Yields:
            State updates
        """
        if self._compiled_graph is None:
            self.compile_graph()

        async for event in self._compiled_graph.astream(input_data, config=config):
            yield event

    def get_state(self, config: Dict) -> Any:
        """Get the current state of the graph.

        Args:
            config: Configuration with thread_id

        Returns:
            Current state
        """
        if self._compiled_graph is None:
            raise RuntimeError("Graph not compiled. Call compile_graph() first.")

        return self._compiled_graph.get_state(config)

    async def update_state(self, config: Dict, values: Dict[str, Any], as_node: Optional[str] = None):
        """Update the graph state.

        Args:
            config: Configuration with thread_id
            values: Values to update
            as_node: Update as if coming from this node
        """
        if self._compiled_graph is None:
            raise RuntimeError("Graph not compiled. Call compile_graph() first.")

        self._compiled_graph.update_state(config, values, as_node=as_node)

    # Lifecycle hooks
    async def before_run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Hook called before running the agent.

        Args:
            input_data: Input data

        Returns:
            Modified input data
        """
        return input_data

    async def after_run(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Hook called after running the agent.

        Args:
            result: Result data

        Returns:
            Modified result data
        """
        return result

    async def on_error(self, error: Exception) -> None:
        """Hook called when an error occurs.

        Args:
            error: The exception that occurred
        """
        pass

    # Utility methods
    def get_config(self, key: Optional[str] = None, default: Any = None) -> Any:
        """Get configuration or a configuration value.

        Args:
            key: Configuration key (if None, returns whole config)
            default: Default value if key not found

        Returns:
            Configuration or configuration value
        """
        if key is None:
            return self.config
        return getattr(self.config, key, default)

    def get_context(self) -> AgentContext:
        """Get the agent context.

        Returns:
            AgentContext: The agent's execution context
        """
        return self.context

    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message.

        Args:
            message: Message to log
            level: Log level
        """
        print(f"[{level}] {self.__class__.__name__}: {message}")

    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(config={self.config.name})"
