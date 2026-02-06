"""Configuration management system."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import toml
import os


@dataclass
class AgentConfig:
    """Configuration for an agent.

    Supports YAML and TOML configuration files.
    """

    # Agent identification
    name: str = "default_agent"
    version: str = "1.0.0"
    description: str = ""

    # LLM configuration
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.0
    llm_max_tokens: Optional[int] = None

    # MCP configuration
    mcp_enabled: bool = True
    mcp_server_url: str = "http://localhost:8000/sse"

    # Search configuration
    search_enabled: bool = True
    search_max_results: int = 3

    # Graph configuration
    max_iterations: int = 25
    interrupt_before: list = field(default_factory=list)
    interrupt_after: list = field(default_factory=list)

    # Runtime configuration
    timeout: int = 600  # seconds
    retry_attempts: int = 3
    enable_checkpointing: bool = True

    # Environment configuration
    langsmith_tracing: bool = False
    langsmith_project: Optional[str] = None

    # Custom metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """Create config from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            AgentConfig instance
        """
        # Filter out keys that aren't in the dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Configuration as dictionary
        """
        return asdict(self)

    def update(self, **kwargs) -> "AgentConfig":
        """Update configuration values.

        Args:
            **kwargs: Values to update

        Returns:
            Updated config instance
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.metadata[key] = value
        return self

    def __repr__(self) -> str:
        """String representation."""
        return f"AgentConfig(name='{self.name}', version='{self.version}')"


def load_config(path: str | Path, format: Optional[str] = None) -> AgentConfig:
    """Load configuration from file.

    Args:
        path: Path to configuration file
        format: File format ('yaml', 'toml', or None for auto-detect)

    Returns:
        AgentConfig instance

    Raises:
        ValueError: If format is unsupported
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    # Auto-detect format from extension
    if format is None:
        format = path.suffix.lstrip('.')

    # Read file content
    content = path.read_text()

    # Parse based on format
    if format in ('yaml', 'yml'):
        data = yaml.safe_load(content)
    elif format == 'toml':
        data = toml.loads(content)
    else:
        raise ValueError(f"Unsupported configuration format: {format}")

    return AgentConfig.from_dict(data)


def save_config(config: AgentConfig, path: str | Path, format: Optional[str] = None) -> None:
    """Save configuration to file.

    Args:
        config: Configuration to save
        path: Path to save to
        format: File format ('yaml' or 'toml', None for auto-detect from extension)

    Raises:
        ValueError: If format is unsupported
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-detect format from extension if not specified
    if format is None:
        format = path.suffix.lstrip('.')
        if not format:
            format = 'yaml'

    data = config.to_dict()

    if format in ('yaml', 'yml'):
        content = yaml.dump(data, default_flow_style=False, sort_keys=False)
    elif format == 'toml':
        content = toml.dumps(data)
    else:
        raise ValueError(f"Unsupported configuration format: {format}")

    path.write_text(content)


def load_config_from_env(prefix: str = "AGENT_") -> Dict[str, Any]:
    """Load configuration from environment variables.

    Args:
        prefix: Prefix for environment variables

    Returns:
        Dictionary of configuration values
    """
    config = {}

    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):].lower()

            # Try to convert to appropriate type
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '', 1).isdigit():
                value = float(value)

            config[config_key] = value

    return config


def merge_configs(*configs: AgentConfig) -> AgentConfig:
    """Merge multiple configurations.

    Later configurations override earlier ones.

    Args:
        *configs: Configurations to merge

    Returns:
        Merged configuration
    """
    if not configs:
        return AgentConfig()

    merged = configs[0].to_dict()

    for config in configs[1:]:
        merged.update(config.to_dict())

    return AgentConfig.from_dict(merged)
