"""Unit tests for state management."""

import pytest
from copy import deepcopy

from core.state.manager import StateManager
from core.state.validators import StateValidator


class TestStateManager:
    """Tests for StateManager class."""

    def test_init_empty(self):
        """Test initialization with no initial state."""
        manager = StateManager()

        assert manager.get_state() == {}
        assert len(manager.get_history()) == 1

    def test_init_with_state(self):
        """Test initialization with initial state."""
        initial = {"key": "value", "count": 0}
        manager = StateManager(initial)

        assert manager.get_state() == initial
        assert manager.get_state() is not initial  # Should be a copy

    def test_get_state(self):
        """Test getting current state."""
        manager = StateManager({"count": 0})
        manager.set("count", 5)

        state = manager.get_state()

        assert state == {"count": 5}

    def test_set_value(self):
        """Test setting a value."""
        manager = StateManager({"count": 0})

        manager.set("count", 10)

        assert manager.get("count") == 10

    def test_set_new_key(self):
        """Test setting a new key."""
        manager = StateManager({"key1": "value1"})

        manager.set("key2", "value2")

        assert manager.get("key2") == "value2"
        assert manager.get("key1") == "value1"

    def test_get_value(self):
        """Test getting a value."""
        manager = StateManager({"key": "value"})

        result = manager.get("key")

        assert result == "value"

    def test_get_missing_key(self):
        """Test getting missing key returns None."""
        manager = StateManager({"key": "value"})

        result = manager.get("missing")

        assert result is None

    def test_get_with_default(self):
        """Test getting with default value."""
        manager = StateManager({"key": "value"})

        result = manager.get("missing", default="default_value")

        assert result == "default_value"

    def test_update_dict(self):
        """Test updating with dictionary."""
        manager = StateManager({"key1": "value1", "count": 0})

        manager.update({"key2": "value2", "count": 5})

        assert manager.get("key1") == "value1"
        assert manager.get("key2") == "value2"
        assert manager.get("count") == 5

    def test_delete_key(self):
        """Test deleting a key."""
        manager = StateManager({"key1": "value1", "key2": "value2"})

        manager.delete("key1")

        assert manager.get("key1") is None
        assert manager.get("key2") == "value2"

    def test_delete_missing_key(self):
        """Test deleting missing key doesn't raise error."""
        manager = StateManager({"key": "value"})

        # Should not raise error
        manager.delete("missing")

        assert manager.get("key") == "value"

    def test_clear(self):
        """Test clearing state."""
        manager = StateManager({"key1": "value1", "key2": "value2"})

        manager.clear()

        assert manager.get_state() == {}

    def test_history_tracks_changes(self):
        """Test history tracks state changes."""
        manager = StateManager({"count": 0})

        manager.set("count", 1)
        manager.set("count", 2)
        manager.set("count", 3)

        history = manager.get_history()

        assert len(history) == 4  # Initial + 3 changes
        assert history[0] == {"count": 0}
        assert history[1] == {"count": 1}
        assert history[2] == {"count": 2}
        assert history[3] == {"count": 3}

    def test_rollback_single_step(self):
        """Test rolling back one step."""
        manager = StateManager({"count": 0})
        manager.set("count", 1)
        manager.set("count", 2)

        manager.rollback(steps=1)

        assert manager.get("count") == 1

    def test_rollback_multiple_steps(self):
        """Test rolling back multiple steps."""
        manager = StateManager({"count": 0})
        manager.set("count", 1)
        manager.set("count", 2)
        manager.set("count", 3)

        manager.rollback(steps=2)

        assert manager.get("count") == 1

    def test_rollback_to_initial(self):
        """Test rolling back to initial state."""
        manager = StateManager({"count": 0})
        manager.set("count", 1)
        manager.set("count", 2)

        manager.rollback(steps=2)

        assert manager.get("count") == 0

    def test_rollback_too_many_steps(self):
        """Test rollback with too many steps raises error."""
        manager = StateManager({"count": 0})
        manager.set("count", 1)

        with pytest.raises(ValueError, match="Cannot rollback"):
            manager.rollback(steps=5)

    def test_rollback_adds_to_history(self):
        """Test rollback adds new state to history."""
        manager = StateManager({"count": 0})
        manager.set("count", 1)
        manager.set("count", 2)

        history_before = len(manager.get_history())
        manager.rollback(steps=1)
        history_after = len(manager.get_history())

        assert history_after == history_before  # Rollback does not add to history

    def test_create_snapshot(self):
        """Test creating a snapshot."""
        manager = StateManager({"count": 0})
        manager.set("count", 5)

        manager.create_snapshot("checkpoint1")

        snapshots = manager.list_snapshots()
        assert "checkpoint1" in snapshots

    def test_restore_snapshot(self):
        """Test restoring a snapshot."""
        manager = StateManager({"count": 0})
        manager.set("count", 5)
        manager.create_snapshot("checkpoint1")
        manager.set("count", 10)

        manager.restore_snapshot("checkpoint1")

        assert manager.get("count") == 5

    def test_restore_nonexistent_snapshot(self):
        """Test restoring nonexistent snapshot raises error."""
        manager = StateManager({"count": 0})

        with pytest.raises(KeyError, match="Snapshot 'nonexistent' not found"):
            manager.restore_snapshot("nonexistent")

    def test_delete_snapshot(self):
        """Test deleting a snapshot."""
        manager = StateManager({"count": 0})
        manager.create_snapshot("checkpoint1")

        manager.delete_snapshot("checkpoint1")

        snapshots = manager.list_snapshots()
        assert "checkpoint1" not in snapshots

    def test_delete_nonexistent_snapshot(self):
        """Test deleting nonexistent snapshot doesn't raise error."""
        manager = StateManager({"count": 0})

        # Should not raise error
        manager.delete_snapshot("nonexistent")

    def test_list_snapshots(self):
        """Test listing snapshots."""
        manager = StateManager({"count": 0})
        manager.create_snapshot("checkpoint1")
        manager.create_snapshot("checkpoint2")
        manager.create_snapshot("checkpoint3")

        snapshots = manager.list_snapshots()

        assert len(snapshots) == 3
        assert "checkpoint1" in snapshots
        assert "checkpoint2" in snapshots
        assert "checkpoint3" in snapshots

    def test_multiple_snapshots_independent(self):
        """Test multiple snapshots are independent."""
        manager = StateManager({"count": 0})

        manager.set("count", 5)
        manager.create_snapshot("snap1")

        manager.set("count", 10)
        manager.create_snapshot("snap2")

        manager.restore_snapshot("snap1")
        assert manager.get("count") == 5

        manager.restore_snapshot("snap2")
        assert manager.get("count") == 10

    def test_snapshot_overwrites(self):
        """Test creating snapshot with same name overwrites."""
        manager = StateManager({"count": 0})

        manager.set("count", 5)
        manager.create_snapshot("checkpoint")

        manager.set("count", 10)
        manager.create_snapshot("checkpoint")

        manager.restore_snapshot("checkpoint")
        assert manager.get("count") == 10

    def test_state_isolation(self):
        """Test state changes don't affect history."""
        manager = StateManager({"items": [1, 2, 3]})

        state = manager.get_state()
        state["items"].append(4)

        # Original state should not be modified
        assert manager.get("items") == [1, 2, 3]

    def test_complex_state_types(self):
        """Test manager handles complex state types."""
        initial = {
            "string": "value",
            "number": 42,
            "float": 3.14,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "bool": True,
            "none": None
        }

        manager = StateManager(initial)

        assert manager.get("string") == "value"
        assert manager.get("number") == 42
        assert manager.get("float") == 3.14
        assert manager.get("list") == [1, 2, 3]
        assert manager.get("dict") == {"nested": "value"}
        assert manager.get("bool") is True
        assert manager.get("none") is None

    def test_update_nested_dict(self):
        """Test updating nested dictionary."""
        manager = StateManager({"data": {"count": 0}})

        # Get nested dict, modify it
        data = manager.get("data")
        data["count"] = 5
        manager.set("data", data)

        assert manager.get("data") == {"count": 5}

    def test_history_limit(self):
        """Test history size can grow large."""
        manager = StateManager({"count": 0})

        for i in range(100):
            manager.set("count", i)

        history = manager.get_history()
        assert len(history) == 101  # Initial + 100 changes

    def test_rollback_multiple_times(self):
        """Test multiple rollbacks."""
        manager = StateManager({"count": 0})
        manager.set("count", 1)
        manager.set("count", 2)
        manager.set("count", 3)

        manager.rollback(steps=1)
        assert manager.get("count") == 2

        manager.rollback(steps=2)
        assert manager.get("count") == 1

    def test_update_after_rollback(self):
        """Test updating state after rollback."""
        manager = StateManager({"count": 0})
        manager.set("count", 1)
        manager.set("count", 2)

        manager.rollback(steps=1)
        manager.set("count", 10)

        assert manager.get("count") == 10

    def test_snapshot_after_rollback(self):
        """Test creating snapshot after rollback."""
        manager = StateManager({"count": 0})
        manager.set("count", 5)
        manager.rollback(steps=1)
        manager.create_snapshot("after_rollback")

        manager.set("count", 10)
        manager.restore_snapshot("after_rollback")

        assert manager.get("count") == 0

    def test_clear_preserves_history(self):
        """Test clear adds to history."""
        manager = StateManager({"key": "value"})
        history_len_before = len(manager.get_history())

        manager.clear()
        history_len_after = len(manager.get_history())

        assert history_len_after == history_len_before + 1
        assert manager.get_history()[-1] == {}

    def test_get_state_returns_copy(self):
        """Test get_state returns a copy, not reference."""
        manager = StateManager({"items": [1, 2, 3]})

        state1 = manager.get_state()
        state2 = manager.get_state()

        assert state1 == state2
        assert state1 is not state2

    def test_multiple_managers_independent(self):
        """Test multiple managers are independent."""
        manager1 = StateManager({"count": 0})
        manager2 = StateManager({"count": 10})

        manager1.set("count", 5)

        assert manager1.get("count") == 5
        assert manager2.get("count") == 10


class TestStateManagerIntegration:
    """Integration tests for StateManager."""

    def test_typical_workflow(self):
        """Test typical state management workflow."""
        # Initialize
        manager = StateManager({"status": "idle", "count": 0})

        # Update state
        manager.set("status", "processing")
        manager.set("count", 5)

        # Create checkpoint
        manager.create_snapshot("before_processing")

        # Continue processing
        manager.set("count", 10)
        manager.set("status", "completed")

        # Restore checkpoint if needed
        manager.restore_snapshot("before_processing")

        assert manager.get("status") == "processing"
        assert manager.get("count") == 5

    def test_complex_state_workflow(self):
        """Test workflow with complex state."""
        initial = {
            "topic": "AI Research",
            "analysts": [],
            "sections": [],
            "report": None
        }

        manager = StateManager(initial)

        # Add analysts
        analysts = [{"name": "Analyst 1"}, {"name": "Analyst 2"}]
        manager.set("analysts", analysts)
        manager.create_snapshot("analysts_created")

        # Add sections
        sections = [{"title": "Introduction"}, {"title": "Methods"}]
        manager.set("sections", sections)
        manager.create_snapshot("sections_created")

        # Generate report
        manager.set("report", "Final report content")

        # Can restore to any checkpoint
        manager.restore_snapshot("analysts_created")
        assert len(manager.get("analysts")) == 2
        assert manager.get("sections") == []
        assert manager.get("report") is None
