# LangGraph Training - Test Suite

Comprehensive testing framework for the LangGraph training project.

## Test Structure

```
tests/
├── unit/              # Unit tests for individual components
│   ├── test_models.py        # Model class tests
│   ├── test_prompts.py       # Prompt function tests
│   └── test_nodes.py         # Node function tests
├── integration/       # Integration tests for component interactions
│   ├── test_graphs.py        # Graph construction and execution
│   └── test_mcp_client.py    # MCP client integration
├── e2e/              # End-to-end workflow tests
│   └── test_research_workflow.py
├── fixtures/         # Test utilities and helpers
│   └── test_helpers.py
└── conftest.py       # Shared fixtures and configuration
```

## Running Tests

### Run All Tests
```bash
make test
# or
uv run pytest tests/
```

### Run Specific Test Categories
```bash
# Unit tests only
make test-unit

# Integration tests only
make test-integration

# End-to-end tests only
make test-e2e

# Skip slow tests
make test-fast
```

### Run with Coverage
```bash
make test-coverage
```

View coverage report: `open htmlcov/index.html`

### Run Specific Test Files
```bash
uv run pytest tests/unit/test_models.py -v
```

### Run Specific Test Classes or Methods
```bash
uv run pytest tests/unit/test_models.py::TestAnalyst -v
uv run pytest tests/unit/test_models.py::TestAnalyst::test_analyst_creation -v
```

### Run Tests Matching Pattern
```bash
uv run pytest tests/ -k "analyst" -v
```

## Test Markers

Tests are organized with pytest markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.mcp` - Tests requiring MCP server
- `@pytest.mark.asyncio` - Async tests

### Run Tests by Marker
```bash
# Run only unit tests
uv run pytest tests/ -m unit

# Run non-MCP tests
uv run pytest tests/ -m "not mcp"

# Run fast tests only
uv run pytest tests/ -m "not slow"
```

## Fixtures

### Available Fixtures

From `conftest.py`:

**Mock Components:**
- `mock_llm` - Mock ChatOpenAI instance
- `mock_tavily_search` - Mock TavilySearch instance
- `mock_mcp_client` - Mock MultiServerMCPClient
- `mock_mcp_context` - Mock MCP context manager

**Model Fixtures:**
- `sample_analyst` - Single Analyst instance
- `sample_analysts` - List of 3 Analyst instances
- `sample_generate_analysts_state` - GenerateAnalystsState
- `sample_interview_state` - InterviewState
- `sample_research_graph_state` - ResearchGraphState

**Environment:**
- `mock_env_vars` - Mock environment variables

### Using Fixtures

```python
@pytest.mark.unit
def test_example(sample_analyst, mock_llm):
    # Use fixtures in your tests
    assert sample_analyst.name == "Dr. Jane Smith"
    mock_llm.invoke.return_value = AIMessage(content="test")
```

## Test Helpers

From `tests/fixtures/test_helpers.py`:

### Factories

```python
from tests.fixtures.test_helpers import (
    MockLLMFactory,
    MockMCPFactory,
    StateBuilder,
    AnalystFactory,
    MessageFactory
)

# Create mock LLM with responses
llm = MockLLMFactory.create_mock_llm(["Response 1", "Response 2"])

# Create analysts
analysts = AnalystFactory.create_analysts(count=3)

# Build state objects
state = StateBuilder.create_interview_state(analyst=analyst)

# Create messages
messages = MessageFactory.create_conversation(turns=5)
```

### Assertions

```python
from tests.fixtures.test_helpers import (
    assert_state_structure,
    assert_message_sequence
)

# Check state has required keys
assert_state_structure(state, ["topic", "analysts", "sections"])

# Check message type sequence
assert_message_sequence(messages, [HumanMessage, AIMessage, HumanMessage])
```

## Writing New Tests

### Unit Test Template

```python
import pytest

@pytest.mark.unit
class TestMyComponent:
    """Tests for MyComponent."""

    def test_component_creation(self):
        """Test creating component."""
        component = MyComponent()
        assert component is not None

    @pytest.mark.asyncio
    async def test_async_method(self, mock_llm):
        """Test async method."""
        result = await component.process(mock_llm)
        assert result is not None
```

### Integration Test Template

```python
import pytest

@pytest.mark.integration
@pytest.mark.asyncio
class TestMyIntegration:
    """Integration tests for feature."""

    async def test_components_interact(self, mock_llm, mock_mcp_client):
        """Test components work together."""
        # Setup
        component1 = Component1()
        component2 = Component2()

        # Execute
        result = await component1.process()
        final = await component2.handle(result)

        # Assert
        assert final is not None
```

## CI/CD

Tests run automatically on:
- Push to main/master/develop branches
- Pull requests

See `.github/workflows/test.yml` for configuration.

## Best Practices

1. **Keep tests isolated** - Each test should be independent
2. **Use fixtures** - Leverage shared fixtures from conftest.py
3. **Mock external services** - Don't call real APIs in tests
4. **Test edge cases** - Include error scenarios
5. **Use descriptive names** - Test names should describe what they test
6. **Keep tests fast** - Mark slow tests with `@pytest.mark.slow`
7. **Test one thing** - Each test should verify one behavior

## Troubleshooting

### Tests fail with "No module named X"
```bash
# Reinstall dependencies
make install
```

### Tests hang or timeout
```bash
# Check for missing @pytest.mark.asyncio on async tests
# Reduce test scope or mark as slow
```

### Coverage not working
```bash
# Install coverage extras
uv pip install pytest-cov

# Run with verbose output
uv run pytest tests/ --cov --cov-report=term -v
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [LangGraph Testing Guide](https://langchain-ai.github.io/langgraph/testing/)
