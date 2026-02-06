"""State management utilities."""

from typing import Any, Dict, Optional, List
from copy import deepcopy
import json


class StateManager:
    """Manager for agent state with history and rollback support."""

    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        """Initialize state manager.

        Args:
            initial_state: Initial state dictionary
        """
        self._current_state = initial_state or {}
        self._history: List[Dict[str, Any]] = [deepcopy(self._current_state)]
        self._snapshots: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from current state.

        Args:
            key: Key to retrieve
            default: Default value if key not found

        Returns:
            State value
        """
        return self._current_state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in current state.

        Args:
            key: Key to set
            value: Value to set
        """
        self._current_state[key] = value
        self._save_to_history()

    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple values in state.

        Args:
            updates: Dictionary of updates
        """
        self._current_state.update(updates)
        self._save_to_history()

    def delete(self, key: str) -> None:
        """Delete a key from state.

        Args:
            key: Key to delete
        """
        if key in self._current_state:
            del self._current_state[key]
            self._save_to_history()

    def get_state(self) -> Dict[str, Any]:
        """Get the entire current state.

        Returns:
            Current state dictionary
        """
        return deepcopy(self._current_state)

    def set_state(self, state: Dict[str, Any]) -> None:
        """Replace the entire state.

        Args:
            state: New state dictionary
        """
        self._current_state = deepcopy(state)
        self._save_to_history()

    def has(self, key: str) -> bool:
        """Check if key exists in state.

        Args:
            key: Key to check

        Returns:
            True if key exists
        """
        return key in self._current_state

    def keys(self) -> List[str]:
        """Get all state keys.

        Returns:
            List of state keys
        """
        return list(self._current_state.keys())

    def values(self) -> List[Any]:
        """Get all state values.

        Returns:
            List of state values
        """
        return list(self._current_state.values())

    def items(self) -> List[tuple[str, Any]]:
        """Get all state items.

        Returns:
            List of (key, value) tuples
        """
        return list(self._current_state.items())

    def clear(self) -> None:
        """Clear all state."""
        self._current_state.clear()
        self._save_to_history()

    # History management
    def _save_to_history(self) -> None:
        """Save current state to history."""
        self._history.append(deepcopy(self._current_state))

    def get_history(self) -> List[Dict[str, Any]]:
        """Get state history.

        Returns:
            List of historical states
        """
        return deepcopy(self._history)

    def get_history_length(self) -> int:
        """Get number of states in history.

        Returns:
            History length
        """
        return len(self._history)

    def rollback(self, steps: int = 1) -> None:
        """Rollback state to previous version.

        Args:
            steps: Number of steps to roll back
        """
        if steps >= len(self._history):
            raise ValueError(f"Cannot rollback {steps} steps (only {len(self._history)} in history)")

        target_index = len(self._history) - steps - 1
        self._current_state = deepcopy(self._history[target_index])
        # Do not save to history - rollback is temporary

    # Snapshot management
    def create_snapshot(self, name: str) -> None:
        """Create a named snapshot of current state.

        Args:
            name: Snapshot name
        """
        self._snapshots[name] = deepcopy(self._current_state)

    def restore_snapshot(self, name: str) -> None:
        """Restore state from a named snapshot.

        Args:
            name: Snapshot name

        Raises:
            KeyError: If snapshot doesn't exist
        """
        if name not in self._snapshots:
            raise KeyError(f"Snapshot '{name}' not found")

        self._current_state = deepcopy(self._snapshots[name])
        self._save_to_history()

    def list_snapshots(self) -> List[str]:
        """List all snapshot names.

        Returns:
            List of snapshot names
        """
        return list(self._snapshots.keys())

    def delete_snapshot(self, name: str) -> None:
        """Delete a named snapshot.

        Args:
            name: Snapshot name
        """
        if name in self._snapshots:
            del self._snapshots[name]

    # Serialization
    def to_json(self) -> str:
        """Serialize state to JSON.

        Returns:
            JSON string
        """
        return json.dumps(self._current_state, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "StateManager":
        """Create StateManager from JSON.

        Args:
            json_str: JSON string

        Returns:
            StateManager instance
        """
        state = json.loads(json_str)
        return cls(initial_state=state)

    def __repr__(self) -> str:
        """String representation."""
        return f"StateManager({len(self._current_state)} keys, {len(self._history)} history)"

    def __len__(self) -> int:
        """Number of keys in state."""
        return len(self._current_state)

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._current_state

    def __getitem__(self, key: str) -> Any:
        """Get item by key."""
        return self._current_state[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item by key."""
        self.set(key, value)
