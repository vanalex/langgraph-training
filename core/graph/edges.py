"""Edge building utilities."""

from typing import Callable, Any, List, Dict
from langgraph.constants import END


class EdgeBuilder:
    """Utilities for building graph edges with conditions."""

    @staticmethod
    def create_router(
        routes: Dict[str, Callable[[Dict], bool]],
        default: str = None
    ) -> Callable:
        """Create a routing function for conditional edges.

        Args:
            routes: Mapping of target names to condition functions
            default: Default target if no condition matches (None raises error)

        Returns:
            Router function

        Example:
            router = EdgeBuilder.create_router({
                "process": lambda state: state.get("ready"),
                "error": lambda state: state.get("has_error"),
                END: lambda state: state.get("done")
            })
        """
        def router(state: Dict[str, Any]) -> str:
            for target, condition in routes.items():
                if condition(state):
                    return target
            if default is not None:
                return default
            raise ValueError("No matching condition found in router")

        return router

    @staticmethod
    def create_value_router(
        state_key: str,
        value_map: Dict[Any, str],
        default: str = END
    ) -> Callable:
        """Create a router based on state value.

        Args:
            state_key: State key to check
            value_map: Mapping of values to target nodes
            default: Default target if no match

        Returns:
            Router function

        Example:
            router = EdgeBuilder.create_value_router(
                "status",
                {"pending": "process", "done": END},
                default="error"
            )
        """
        def router(state: Dict[str, Any]) -> str:
            value = state.get(state_key)
            return value_map.get(value, default)

        return router

    @staticmethod
    def create_count_router(
        state_key: str,
        max_count: int,
        continue_target: str,
        end_target: str = END
    ) -> Callable:
        """Create a router based on iteration count.

        Args:
            state_key: State key containing count
            max_count: Maximum count before ending
            continue_target: Target to continue to
            end_target: Target when max reached

        Returns:
            Router function
        """
        def router(state: Dict[str, Any]) -> str:
            count = state.get(state_key, 0)
            if count < max_count:
                return continue_target
            return end_target

        return router

    @staticmethod
    def create_feedback_router(
        feedback_key: str,
        regenerate_target: str,
        continue_target: str
    ) -> Callable:
        """Create a router for human feedback loops.

        Args:
            feedback_key: State key for feedback
            regenerate_target: Target to regenerate
            continue_target: Target to continue

        Returns:
            Router function
        """
        def router(state: Dict[str, Any]) -> str:
            feedback = state.get(feedback_key)
            if feedback:
                return regenerate_target
            return continue_target

        return router

    @staticmethod
    def create_multi_condition_router(
        conditions: List[tuple[str, Callable[[Dict], bool]]],
        default: str = END
    ) -> Callable:
        """Create router with ordered conditions.

        Evaluates conditions in order and returns first match.

        Args:
            conditions: List of (target, condition) tuples
            default: Default target if no match

        Returns:
            Router function
        """
        def router(state: Dict[str, Any]) -> str:
            for target, condition in conditions:
                if condition(state):
                    return target
            return default

        return router

    @staticmethod
    def create_field_router(
        field: str,
        path_map: Dict[Any, str],
        default: str = None
    ) -> Callable:
        """Create a router based on a field value.

        Args:
            field: State field to check
            path_map: Mapping of field values to paths
            default: Default path if value not in map

        Returns:
            Router function
        """
        def router(state: Dict[str, Any]) -> str:
            value = state.get(field)
            if value in path_map:
                return path_map[value]
            if default is not None:
                return default
            raise ValueError(f"No path found for field '{field}' with value '{value}'")

        return router

    @staticmethod
    def create_boolean_router(
        field: str,
        true_path: str,
        false_path: str
    ) -> Callable:
        """Create a router for boolean field values.

        Args:
            field: Boolean state field to check
            true_path: Path when field is True
            false_path: Path when field is False or missing

        Returns:
            Router function
        """
        def router(state: Dict[str, Any]) -> str:
            if state.get(field, False):
                return true_path
            return false_path

        return router
