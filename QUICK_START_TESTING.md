# Quick Start - Testing Guide

## 🚀 Running Tests

### Install Dependencies
```bash
make install
# or
uv sync
```

### Run Tests
```bash
# All tests (64 tests, ~0.1s)
make test

# By category
make test-unit           # Unit tests only
make test-integration    # Integration tests only
make test-e2e           # End-to-end tests only

# With coverage
make test-coverage       # Generates HTML report

# Skip slow tests
make test-fast

# Specific file
uv run pytest tests/unit/test_models.py -v

# Specific test
uv run pytest tests/unit/test_models.py::TestAnalyst::test_analyst_creation -v

# Watch mode (for TDD)
make test-watch
```

## 📝 Writing Tests

### Unit Test Template
```python
import pytest

@pytest.mark.unit
class TestMyFeature:
    def test_something(self):
        # Arrange
        data = create_test_data()

        # Act
        result = my_function(data)

        # Assert
        assert result == expected
```

### Async Test
```python
@pytest.mark.unit
@pytest.mark.asyncio
class TestAsyncFeature:
    async def test_async_function(self, mock_llm):
        result = await my_async_function(mock_llm)
        assert result is not None
```

### Using Fixtures
```python
def test_with_fixtures(sample_analyst, mock_llm):
    # Use the fixtures
    assert sample_analyst.name == "Dr. Jane Smith"
    mock_llm.invoke.return_value = "test"
```

## 🎯 Available Fixtures

From `tests/conftest.py`:
- `mock_llm` - Mock ChatOpenAI
- `mock_tavily_search` - Mock search
- `mock_mcp_client` - Mock MCP client
- `sample_analyst` - Sample analyst instance
- `sample_analysts` - List of analysts
- `sample_generate_analysts_state` - Test state
- `sample_interview_state` - Test state
- `sample_research_graph_state` - Test state
- `mock_env_vars` - Mock environment

## 🔍 Test Markers

Run tests by marker:
```bash
# Only unit tests
uv run pytest -m unit

# Exclude slow tests
uv run pytest -m "not slow"

# Only MCP tests
uv run pytest -m mcp

# Multiple markers
uv run pytest -m "unit and not slow"
```

Available markers:
- `@pytest.mark.unit`
- `@pytest.mark.integration`
- `@pytest.mark.e2e`
- `@pytest.mark.slow`
- `@pytest.mark.mcp`
- `@pytest.mark.asyncio`

## 🛠️ Debugging Tests

### Verbose Output
```bash
uv run pytest tests/ -v
```

### Show Print Statements
```bash
uv run pytest tests/ -s
```

### Stop on First Failure
```bash
uv run pytest tests/ -x
```

### Run Last Failed
```bash
uv run pytest tests/ --lf
```

### Debug with PDB
```bash
uv run pytest tests/ --pdb
```

## 📊 Coverage

```bash
# Generate coverage report
make test-coverage

# View HTML report
open htmlcov/index.html
```

## 🐛 Common Issues

### Import Errors
```bash
# Reinstall dependencies
make install
```

### Async Warnings
- Remove `@pytest.mark.asyncio` from non-async functions
- Or ignore warnings: `pytest -W ignore::pytest.PytestWarning`

### MCP Client Errors
- Ensure `_CURRENT_CLIENT` is mocked in tests
- Use fixtures from conftest.py

## 📚 Documentation

- Full guide: `tests/README.md`
- Test results: `TEST_RESULTS.md`
- Implementation: `TESTING_SUMMARY.md`

## ✨ Tips

1. **Write tests first** (TDD)
2. **Keep tests fast** - Use mocks
3. **One assertion per test** (when possible)
4. **Use descriptive names** - `test_analyst_creation_with_valid_data`
5. **Leverage fixtures** - Don't repeat setup code
6. **Test edge cases** - Not just happy path
7. **Run tests often** - Catch bugs early

## 🎓 Examples

See existing tests for patterns:
- `tests/unit/test_models.py` - Model validation
- `tests/unit/test_nodes.py` - Node testing
- `tests/integration/test_graphs.py` - Graph testing
- `tests/e2e/test_research_workflow.py` - Full workflows

---

**Quick Reference Card**

| Command | What It Does |
|---------|--------------|
| `make test` | Run all tests |
| `make test-unit` | Unit tests only |
| `make test-coverage` | With coverage |
| `pytest -k analyst` | Tests matching "analyst" |
| `pytest --lf` | Run last failed |
| `pytest -x` | Stop on first fail |
| `pytest -v` | Verbose output |

**Happy Testing!** 🧪
