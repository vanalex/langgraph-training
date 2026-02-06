"""Node registry and decorators."""

from typing import Callable, Dict, Optional, Any, List
from functools import wraps
import inspect


class NodeRegistry:
    """Registry for graph nodes with metadata."""

    def __init__(self):
        """Initialize the node registry."""
        self._nodes: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        retry: bool = False,
        timeout: Optional[int] = None
    ) -> Callable:
        """Decorator to register a node function.

        Args:
            name: Node name (uses function name if not provided)
            description: Node description
            tags: Tags for categorizing nodes
            retry: Whether to retry on failure
            timeout: Timeout in seconds

        Returns:
            Decorator function

        Example:
            @registry.register("process_data", description="Processes input data")
            async def process(state):
                return {"result": "processed"}
        """
        def decorator(func: Callable) -> Callable:
            node_name = name or func.__name__

            # Store metadata
            self._nodes[node_name] = {
                "function": func,
                "name": node_name,
                "description": description or func.__doc__ or "",
                "tags": tags or [],
                "retry": retry,
                "timeout": timeout,
                "is_async": inspect.iscoroutinefunction(func),
                "signature": str(inspect.signature(func))
            }

            # Add metadata to function
            func._node_name = node_name
            func._node_metadata = self._nodes[node_name]

            return func

        return decorator

    def get(self, name: str) -> Optional[Callable]:
        """Get a registered node function.

        Args:
            name: Node name

        Returns:
            Node function or None
        """
        node = self._nodes.get(name)
        return node["function"] if node else None

    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get node metadata.

        Args:
            name: Node name

        Returns:
            Node metadata or None
        """
        return self._nodes.get(name)

    def list_nodes(self, tags: Optional[List[str]] = None) -> List[str]:
        """List registered node names.

        Args:
            tags: Filter by tags (if provided)

        Returns:
            List of node names
        """
        if tags is None:
            return list(self._nodes.keys())

        return [
            name for name, meta in self._nodes.items()
            if any(tag in meta["tags"] for tag in tags)
        ]

    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all nodes.

        Returns:
            Dictionary of node metadata
        """
        return self._nodes.copy()

    def clear(self) -> None:
        """Clear all registered nodes."""
        self._nodes.clear()

    def __len__(self) -> int:
        """Number of registered nodes."""
        return len(self._nodes)

    def __contains__(self, name: str) -> bool:
        """Check if node is registered."""
        return name in self._nodes

    def __repr__(self) -> str:
        """String representation."""
        return f"NodeRegistry({len(self._nodes)} nodes)"


# Global registry instance
_global_registry = NodeRegistry()


def register_node(
    name: Optional[str] = None,
    description: str = "",
    tags: Optional[List[str]] = None,
    retry: bool = False,
    timeout: Optional[int] = None
) -> Callable:
    """Register a node using the global registry.

    Args:
        name: Node name
        description: Node description
        tags: Node tags
        retry: Enable retry on failure
        timeout: Timeout in seconds

    Returns:
        Decorator function

    Example:
        @register_node("my_node", tags=["processing"])
        async def my_node(state):
            return {"result": "done"}
    """
    return _global_registry.register(name, description, tags, retry, timeout)


def get_node(name: str) -> Optional[Dict[str, Any]]:
    """Get a node metadata from the global registry.

    Args:
        name: Node name

    Returns:
        Node metadata dict or None
    """
    meta = _global_registry.get_metadata(name)
    if meta:
        # Return dict with consistent structure for tests
        return {
            "name": meta["name"],
            "func": meta["function"],
            "description": meta.get("description", ""),
            "tags": meta.get("tags", [])
        }
    return None


def list_nodes(tag: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all registered nodes with metadata.

    Args:
        tag: Filter by single tag (if provided)

    Returns:
        List of node metadata dicts
    """
    tags = [tag] if tag else None
    node_names = _global_registry.list_nodes(tags)

    return [
        {
            "name": name,
            "func": _global_registry.get_metadata(name)["function"],
            "description": _global_registry.get_metadata(name).get("description", ""),
            "tags": _global_registry.get_metadata(name).get("tags", [])
        }
        for name in node_names
    ]


def list_all_nodes(tags: Optional[List[str]] = None) -> List[str]:
    """List all registered node names.

    Args:
        tags: Filter by tags

    Returns:
        List of node names
    """
    return _global_registry.list_nodes(tags)
