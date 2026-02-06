# LangGraph Training - Architecture Guide

## Overview

The LangGraph training project uses a modular, component-based architecture that promotes code reuse, testability, and maintainability.

## Architecture Components

### 1. Core Framework (`core/`)

The core framework provides base classes and utilities for building LangGraph agents.

```
core/
├── agents/          # Base agent classes and mixins
├── config/          # Configuration management
├── graph/           # Graph building utilities
└── state/           # State management
```

### 2. Base Agent (`core/agents/base_agent.py`)

The `BaseAgent` class provides common functionality for all agents:

```python
from core.agents.base_agent import BaseAgent, AgentContext
from core.config.settings import AgentConfig

class MyAgent(BaseAgent):
    def build_graph(self):
        # Build your graph here
        pass

    def get_state_schema(self):
        # Return your state schema
        return MyStateSchema
```

**Key Features:**
- Configuration management
- LLM initialization
- Graph compilation
- Checkpointing support
- Lifecycle hooks (before_run, after_run, on_error)
- State management

### 3. Mixins (`core/agents/mixins.py`)

Mixins provide reusable functionality that can be composed:

#### MCPMixin
Provides MCP client functionality:
```python
class MyAgent(BaseAgent, MCPMixin):
    async def my_node(self, state):
        prompt = await self.get_prompt("my-prompt", {"arg": "value"})
        return {"result": prompt}
```

#### LLMMixin
Provides LLM functionality:
```python
class MyAgent(BaseAgent, LLMMixin):
    async def my_node(self, state):
        response = await self.generate(messages)
        structured = await self.generate_structured(messages, schema=MySchema)
        return {"response": response}
```

#### SearchMixin
Provides search functionality:
```python
class MyAgent(BaseAgent, SearchMixin):
    async def my_node(self, state):
        results = await self.search("my query")
        return {"results": results}
```

#### StateMixin
Provides state management helpers:
```python
class MyAgent(BaseAgent, StateMixin):
    def my_node(self, state):
        value = self.get_state_value(state, "key", default="")
        return self.update_state_value(state, "key", "new_value")
```

### 4. Configuration Management (`core/config/`)

Configuration is managed through YAML/TOML files:

```yaml
# config/my_agent.yaml
name: my_agent
version: 1.0.0
llm_model: gpt-4o
llm_temperature: 0.0
mcp_enabled: true
mcp_server_url: http://localhost:8000/sse
```

Load and use configuration:
```python
from core.config.settings import load_config

config = load_config("config/my_agent.yaml")
agent = MyAgent(config)
```

**Configuration Features:**
- YAML and TOML support
- Environment variable overrides
- Configuration validation
- Default values
- Metadata support

### 5. Graph Building (`core/graph/`)

#### GraphBuilder
Fluent interface for building graphs:

```python
from core.graph.builder import GraphBuilder

builder = GraphBuilder(MyStateSchema)
builder.add_node("process", process_func)
       .add_edge(START, "process")
       .add_edge("process", END)
       .build()
```

#### NodeRegistry
Register nodes with metadata:

```python
from core.graph.nodes import register_node

@register_node("process", description="Processes data", tags=["processing"])
async def process(state):
    return {"result": "processed"}
```

#### EdgeBuilder
Utilities for conditional edges:

```python
from core.graph.edges import EdgeBuilder

# Create a router
router = EdgeBuilder.create_router({
    "continue": lambda s: s["count"] < 10,
    END: lambda s: s["count"] >= 10
})
```

### 6. State Management (`core/state/`)

#### StateManager
Manage state with history and snapshots:

```python
from core.state.manager import StateManager

manager = StateManager({"count": 0})
manager.set("count", 1)
manager.create_snapshot("checkpoint1")
manager.rollback(steps=1)
```

#### StateValidator
Validate state with rules:

```python
from core.state.validators import StateValidator

validator = StateValidator()
validator.require_key("topic")
         .validate_type("count", int)
         .validate_range("temperature", min_val=0, max_val=2)

errors = validator.validate(state)
```

## Design Patterns

### 1. Dependency Injection

Dependencies are injected through `AgentContext`:

```python
from core.agents.base_agent import AgentContext
from langchain_openai import ChatOpenAI

context = AgentContext(
    llm=ChatOpenAI(model="gpt-4o"),
    mcp_client=my_mcp_client,
    config=config
)

agent = MyAgent(config, context)
```

### 2. Composition over Inheritance

Use mixins to compose functionality:

```python
class ResearchAgent(BaseAgent, MCPMixin, LLMMixin, SearchMixin):
    """Agent with MCP, LLM, and search capabilities."""
    pass
```

### 3. Separation of Concerns

- **Nodes**: Business logic (in agent methods)
- **Edges**: Control flow (using EdgeBuilder)
- **State**: Data (using state schema)
- **Configuration**: Settings (YAML/TOML files)

### 4. Builder Pattern

Use GraphBuilder for fluent graph construction:

```python
graph = (GraphBuilder(StateSchema)
    .add_node("start", start_func)
    .add_node("process", process_func)
    .add_edge(START, "start")
    .add_conditional_edge("start", router, ["process", END])
    .add_edge("process", END)
    .build())
```

## Project Structure

```
langgraph-training/
├── core/                      # Core framework
│   ├── agents/               # Base classes and mixins
│   ├── config/               # Configuration management
│   ├── graph/                # Graph building utilities
│   └── state/                # State management
├── config/                    # Configuration files
│   └── research_assistant.yaml
├── module4/                   # Agents and applications
│   ├── model/                # Data models
│   └── research_assistant.py # Original implementation
├── examples/                  # Example implementations
│   └── research_agent_refactored.py
├── tests/                     # Test suite
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── mcp_server/               # MCP server
└── docs/                     # Documentation
```

## Best Practices

### 1. Use Configuration Files

Store agent configuration in YAML/TOML:
```python
config = load_config("config/my_agent.yaml")
agent = MyAgent(config)
```

### 2. Leverage Mixins

Use mixins for cross-cutting concerns:
```python
class MyAgent(BaseAgent, MCPMixin, LLMMixin):
    # Automatically get MCP and LLM functionality
    pass
```

### 3. Validate Configuration

Always validate configuration:
```python
from core.config.validators import validate_config

config = load_config("config/my_agent.yaml")
warnings = validate_config(config)
```

### 4. Use GraphBuilder

Build graphs with the fluent interface:
```python
builder = GraphBuilder(StateSchema)
builder.add_node("node1", func1).add_edge(START, "node1")
```

### 5. Register Nodes

Use decorators to register nodes:
```python
@register_node("my_node", tags=["processing"])
async def my_node(state):
    return {"result": "done"}
```

### 6. Test Components

Test each component independently:
```python
def test_my_node():
    state = {"input": "test"}
    result = await my_node(state)
    assert result["output"] == "expected"
```

## Migration Guide

### From Old to New Architecture

**Before (Old Architecture):**
```python
def create_analysts(state: GenerateAnalystsState, llm):
    topic = state['topic']
    system_message = get_analyst_instructions(topic, "", 3)
    analysts = llm.invoke([SystemMessage(content=system_message)])
    return {"analysts": analysts.analysts}
```

**After (New Architecture):**
```python
class MyAgent(BaseAgent, MCPMixin, LLMMixin):
    async def create_analysts(self, state):
        topic = state['topic']
        system_message = await self.get_prompt("analyst-instructions",
                                                {"topic": topic})
        analysts = await self.generate_structured(
            [SystemMessage(content=system_message)],
            schema=Perspectives
        )
        return {"analysts": analysts.analysts}
```

## Benefits

### 1. **Modularity**
- Components are independent and reusable
- Easy to test in isolation
- Clear separation of concerns

### 2. **Flexibility**
- Mix and match functionality with mixins
- Configure behavior through YAML/TOML
- Easy to extend with new components

### 3. **Maintainability**
- Centralized configuration
- Consistent patterns across agents
- Well-documented interfaces

### 4. **Testability**
- Dependencies can be mocked
- Components tested independently
- Integration tests simplified

### 5. **Scalability**
- Easy to add new agents
- Shared functionality through mixins
- Configuration-driven behavior

## Examples

See `examples/research_agent_refactored.py` for a complete example of the new architecture.

## Further Reading

- [Configuration Guide](./CONFIGURATION.md)
- [Testing Guide](../tests/README.md)
- [API Reference](./API.md)
