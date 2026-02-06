"""Unit tests for configuration management."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open

from core.config.settings import AgentConfig, load_config, save_config
from core.config.validators import validate_config
from core.state.validators import StateValidator


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AgentConfig()

        assert config.name == "default_agent"
        assert config.version == "1.0.0"
        assert config.description == ""
        assert config.llm_model == "gpt-4o"
        assert config.llm_temperature == 0.0
        assert config.llm_max_tokens is None
        assert config.mcp_enabled is True
        assert config.mcp_server_url == "http://localhost:8000/sse"
        assert config.search_enabled is True
        assert config.search_max_results == 3
        assert config.max_iterations == 25
        assert config.interrupt_before == []
        assert config.interrupt_after == []
        assert config.timeout == 600
        assert config.retry_attempts == 3
        assert config.enable_checkpointing is True
        assert config.langsmith_tracing is False
        assert config.langsmith_project is None
        assert config.metadata == {}

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AgentConfig(
            name="custom_agent",
            version="2.0.0",
            llm_model="gpt-4o-mini",
            llm_temperature=0.7,
            mcp_enabled=False,
            search_max_results=5,
            metadata={"key": "value"}
        )

        assert config.name == "custom_agent"
        assert config.version == "2.0.0"
        assert config.llm_model == "gpt-4o-mini"
        assert config.llm_temperature == 0.7
        assert config.mcp_enabled is False
        assert config.search_max_results == 5
        assert config.metadata == {"key": "value"}

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "name": "test_agent",
            "llm_model": "gpt-4o",
            "llm_temperature": 0.5,
            "metadata": {"max_analysts": 3}
        }

        config = AgentConfig.from_dict(data)

        assert config.name == "test_agent"
        assert config.llm_model == "gpt-4o"
        assert config.llm_temperature == 0.5
        assert config.metadata == {"max_analysts": 3}

    def test_from_dict_with_extra_keys(self):
        """Test from_dict ignores extra keys."""
        data = {
            "name": "test_agent",
            "extra_key": "should_be_ignored",
            "another_extra": 123
        }

        config = AgentConfig.from_dict(data)

        assert config.name == "test_agent"
        assert not hasattr(config, "extra_key")

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = AgentConfig(
            name="test_agent",
            llm_temperature=0.7,
            metadata={"key": "value"}
        )

        data = config.to_dict()

        assert data["name"] == "test_agent"
        assert data["llm_temperature"] == 0.7
        assert data["metadata"] == {"key": "value"}
        assert "llm_model" in data
        assert "mcp_enabled" in data


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_yaml(self):
        """Test loading YAML configuration."""
        yaml_content = """
name: test_agent
version: 1.0.0
llm_model: gpt-4o
llm_temperature: 0.5
metadata:
  max_analysts: 3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = load_config(temp_path)

            assert config.name == "test_agent"
            assert config.version == "1.0.0"
            assert config.llm_model == "gpt-4o"
            assert config.llm_temperature == 0.5
            assert config.metadata == {"max_analysts": 3}
        finally:
            os.unlink(temp_path)

    def test_load_yml(self):
        """Test loading .yml configuration."""
        yaml_content = """
name: test_agent
llm_model: gpt-4o-mini
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = load_config(temp_path)

            assert config.name == "test_agent"
            assert config.llm_model == "gpt-4o-mini"
        finally:
            os.unlink(temp_path)

    def test_load_toml(self):
        """Test loading TOML configuration."""
        toml_content = """
name = "test_agent"
version = "1.0.0"
llm_model = "gpt-4o"
llm_temperature = 0.5

[metadata]
max_analysts = 3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(toml_content)
            temp_path = f.name

        try:
            config = load_config(temp_path)

            assert config.name == "test_agent"
            assert config.version == "1.0.0"
            assert config.llm_model == "gpt-4o"
            assert config.llm_temperature == 0.5
            assert config.metadata == {"max_analysts": 3}
        finally:
            os.unlink(temp_path)

    def test_load_explicit_format(self):
        """Test loading with explicit format specification."""
        yaml_content = "name: test_agent\nllm_model: gpt-4o"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = load_config(temp_path, format='yaml')

            assert config.name == "test_agent"
            assert config.llm_model == "gpt-4o"
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_unsupported_format(self):
        """Test loading unsupported format raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"name": "test"}')
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported configuration format"):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML raises error."""
        invalid_yaml = """
name: test_agent
  invalid: indentation
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            temp_path = f.name

        try:
            with pytest.raises(Exception):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)


class TestSaveConfig:
    """Tests for save_config function."""

    def test_save_yaml(self):
        """Test saving configuration as YAML."""
        config = AgentConfig(
            name="test_agent",
            llm_model="gpt-4o",
            llm_temperature=0.7
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            save_config(config, temp_path)

            loaded = load_config(temp_path)
            assert loaded.name == config.name
            assert loaded.llm_model == config.llm_model
            assert loaded.llm_temperature == config.llm_temperature
        finally:
            os.unlink(temp_path)

    def test_save_toml(self):
        """Test saving configuration as TOML."""
        config = AgentConfig(
            name="test_agent",
            llm_model="gpt-4o",
            metadata={"key": "value"}
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            temp_path = f.name

        try:
            save_config(config, temp_path)

            loaded = load_config(temp_path)
            assert loaded.name == config.name
            assert loaded.llm_model == config.llm_model
            assert loaded.metadata == config.metadata
        finally:
            os.unlink(temp_path)

    def test_save_explicit_format(self):
        """Test saving with explicit format."""
        config = AgentConfig(name="test_agent")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name

        try:
            save_config(config, temp_path, format='yaml')

            loaded = load_config(temp_path, format='yaml')
            assert loaded.name == config.name
        finally:
            os.unlink(temp_path)

    def test_save_creates_directory(self):
        """Test saving creates parent directory if needed."""
        config = AgentConfig(name="test_agent")

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, "subdir", "config.yaml")

            save_config(config, temp_path)

            assert os.path.exists(temp_path)
            loaded = load_config(temp_path)
            assert loaded.name == config.name


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_validate_valid_config(self):
        """Test validating valid configuration."""
        config = AgentConfig(
            name="test_agent",
            llm_model="gpt-4o",
            llm_temperature=0.5
        )

        warnings = validate_config(config)

        assert len(warnings) == 0

    def test_validate_temperature_out_of_range(self):
        """Test validation warns about temperature out of range."""
        config = AgentConfig(llm_temperature=2.5)

        warnings = validate_config(config)

        assert any("temperature" in w.lower() for w in warnings)

    def test_validate_negative_max_results(self):
        """Test validation warns about negative max_results."""
        config = AgentConfig(search_max_results=-1, search_enabled=True)

        warnings = validate_config(config)

        assert any("search_max_results" in w for w in warnings)

    def test_validate_zero_max_iterations(self):
        """Test validation warns about zero max_iterations."""
        config = AgentConfig(max_iterations=0)

        warnings = validate_config(config)

        assert any("max_iterations" in w for w in warnings)

    def test_validate_negative_timeout(self):
        """Test validation warns about negative timeout."""
        config = AgentConfig(timeout=-100)

        warnings = validate_config(config)

        assert any("timeout" in w for w in warnings)

    def test_validate_multiple_warnings(self):
        """Test validation returns multiple warnings."""
        config = AgentConfig(
            llm_temperature=3.0,
            search_max_results=-5, search_enabled=True,
            max_iterations=0
        )

        warnings = validate_config(config)

        assert len(warnings) >= 3


class TestStateValidator:
    """Tests for StateValidator class."""

    def test_require_key_success(self):
        """Test require_key with valid state."""
        validator = StateValidator()
        validator.require_key("name")
        validator.require_key("age")

        state = {"name": "test", "age": 30}
        errors = validator.validate(state)

        assert len(errors) == 0

    def test_require_key_failure(self):
        """Test require_key with missing key."""
        validator = StateValidator()
        validator.require_key("name")

        state = {"age": 30}
        errors = validator.validate(state)

        assert len(errors) == 1
        assert "name" in errors[0]

    def test_validate_type_success(self):
        """Test validate_type with correct type."""
        validator = StateValidator()
        validator.validate_type("name", str)
        validator.validate_type("age", int)

        state = {"name": "test", "age": 30}
        errors = validator.validate(state)

        assert len(errors) == 0

    def test_validate_type_failure(self):
        """Test validate_type with incorrect type."""
        validator = StateValidator()
        validator.validate_type("age", int)

        state = {"age": "not an int"}
        errors = validator.validate(state)

        assert len(errors) == 1
        assert "age" in errors[0]

    def test_validate_range_success(self):
        """Test validate_range with value in range."""
        validator = StateValidator()
        validator.validate_range("temperature", min_val=0.0, max_val=2.0)

        state = {"temperature": 1.0}
        errors = validator.validate(state)

        assert len(errors) == 0

    def test_validate_range_too_low(self):
        """Test validate_range with value too low."""
        validator = StateValidator()
        validator.validate_range("temperature", min_val=0.0, max_val=2.0)

        state = {"temperature": -1.0}
        errors = validator.validate(state)

        assert len(errors) == 1
        assert "temperature" in errors[0]

    def test_validate_range_too_high(self):
        """Test validate_range with value too high."""
        validator = StateValidator()
        validator.validate_range("temperature", min_val=0.0, max_val=2.0)

        state = {"temperature": 3.0}
        errors = validator.validate(state)

        assert len(errors) == 1
        assert "temperature" in errors[0]

    def test_validate_range_only_min(self):
        """Test validate_range with only minimum."""
        validator = StateValidator()
        validator.validate_range("count", min_val=0)

        state_valid = {"count": 5}
        state_invalid = {"count": -1}

        errors_valid = validator.validate(state_valid)
        errors_invalid = validator.validate(state_invalid)

        assert len(errors_valid) == 0
        assert len(errors_invalid) == 1

    def test_validate_range_only_max(self):
        """Test validate_range with only maximum."""
        validator = StateValidator()
        validator.validate_range("count", max_val=100)

        state_valid = {"count": 50}
        state_invalid = {"count": 150}

        errors_valid = validator.validate(state_valid)
        errors_invalid = validator.validate(state_invalid)

        assert len(errors_valid) == 0
        assert len(errors_invalid) == 1

    def test_validate_custom_rule_success(self):
        """Test validate_custom_rule with passing rule."""
        validator = StateValidator()
        validator.validate_custom_rule(
            "email",
            lambda v: "@" in v,
            "email must contain @"
        )

        state = {"email": "test@example.com"}
        errors = validator.validate(state)

        assert len(errors) == 0

    def test_validate_custom_rule_failure(self):
        """Test validate_custom_rule with failing rule."""
        validator = StateValidator()
        validator.validate_custom_rule(
            "email",
            lambda v: "@" in v,
            "email must contain @"
        )

        state = {"email": "invalid"}
        errors = validator.validate(state)

        assert len(errors) == 1
        assert "@" in errors[0]

    def test_chaining_validators(self):
        """Test chaining multiple validators."""
        validator = (StateValidator()
                     .require_key("name")
                     .validate_type("age", int)
                     .validate_range("age", min_val=0, max_val=120))

        state = {"name": "test", "age": 30}
        errors = validator.validate(state)

        assert len(errors) == 0

    def test_multiple_errors(self):
        """Test validation with multiple errors."""
        validator = (StateValidator()
                     .require_key("name")
                     .require_key("age")
                     .validate_type("age", int))

        state = {"age": "not an int"}
        errors = validator.validate(state)

        assert len(errors) >= 2

    def test_validate_empty_state(self):
        """Test validation with empty state."""
        validator = StateValidator()

        state = {}
        errors = validator.validate(state)

        assert len(errors) == 0

    def test_validate_missing_optional_key(self):
        """Test validation doesn't fail on missing optional key."""
        validator = StateValidator()
        validator.validate_type("optional_field", str)

        state = {}
        errors = validator.validate(state)

        # Should not error if key is not required
        assert len(errors) == 0
