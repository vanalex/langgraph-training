"""Configuration validation utilities."""

from typing import List, Optional
from core.config.settings import AgentConfig


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


def validate_config(config: AgentConfig) -> List[str]:
    """Validate agent configuration.

    Args:
        config: Configuration to validate

    Returns:
        List of validation warnings (empty if all valid)

    Raises:
        ConfigValidationError: If configuration is invalid
    """
    errors = []
    warnings = []

    # Validate LLM configuration
    if not config.llm_model:
        errors.append("llm_model is required")

    if config.llm_temperature < 0 or config.llm_temperature > 2:
        warnings.append("llm_temperature should be between 0 and 2")

    if config.llm_max_tokens is not None and config.llm_max_tokens <= 0:
        errors.append("llm_max_tokens must be positive")

    # Validate MCP configuration
    if config.mcp_enabled and not config.mcp_server_url:
        errors.append("mcp_server_url is required when mcp_enabled is True")

    if config.mcp_server_url and not config.mcp_server_url.startswith(('http://', 'https://')):
        warnings.append("mcp_server_url should start with http:// or https://")

    # Validate search configuration
    if config.search_enabled and config.search_max_results <= 0:
        warnings.append("search_max_results must be positive")

    # Validate graph configuration
    if config.max_iterations <= 0:
        warnings.append("max_iterations must be positive")

    # Validate runtime configuration
    if config.timeout <= 0:
        warnings.append("timeout must be positive")

    if config.retry_attempts < 0:
        errors.append("retry_attempts must be non-negative")

    # Raise if there are errors
    if errors:
        raise ConfigValidationError(
            f"Configuration validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
        )

    return warnings


def validate_required_fields(config: AgentConfig, required: List[str]) -> None:
    """Validate that required fields are present.

    Args:
        config: Configuration to validate
        required: List of required field names

    Raises:
        ConfigValidationError: If required fields are missing
    """
    missing = []

    for field in required:
        value = getattr(config, field, None)
        if value is None or (isinstance(value, str) and not value):
            missing.append(field)

    if missing:
        raise ConfigValidationError(
            f"Missing required configuration fields: {', '.join(missing)}"
        )


def validate_llm_config(config: AgentConfig) -> None:
    """Validate LLM-specific configuration.

    Args:
        config: Configuration to validate

    Raises:
        ConfigValidationError: If LLM config is invalid
    """
    supported_models = [
        'gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o',
        'claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku'
    ]

    if config.llm_model not in supported_models:
        raise ConfigValidationError(
            f"Unsupported LLM model: {config.llm_model}. "
            f"Supported models: {', '.join(supported_models)}"
        )


def suggest_optimizations(config: AgentConfig) -> List[str]:
    """Suggest configuration optimizations.

    Args:
        config: Configuration to analyze

    Returns:
        List of optimization suggestions
    """
    suggestions = []

    # Performance suggestions
    if config.llm_temperature > 0.5:
        suggestions.append(
            "High temperature may reduce consistency. "
            "Consider lowering for more deterministic outputs."
        )

    if config.max_iterations > 20:
        suggestions.append(
            "High max_iterations may impact performance. "
            "Consider reducing or adding early stopping conditions."
        )

    if not config.enable_checkpointing:
        suggestions.append(
            "Checkpointing is disabled. Consider enabling for better recovery."
        )

    # Cost optimization
    if config.llm_model in ['gpt-4', 'gpt-4-turbo', 'claude-3-opus']:
        suggestions.append(
            f"Using expensive model '{config.llm_model}'. "
            "Consider using cheaper alternatives for development."
        )

    # Observability suggestions
    if not config.langsmith_tracing:
        suggestions.append(
            "LangSmith tracing is disabled. Enable for better debugging."
        )

    return suggestions
