"""State validation utilities."""

from typing import Dict, Any, List, Optional, Callable, Type
from dataclasses import dataclass


@dataclass
class ValidationRule:
    """A validation rule for state."""
    key: str
    validator: Callable[[Any], bool]
    message: str
    required: bool = False


class StateValidator:
    """Validator for state dictionaries."""

    def __init__(self):
        """Initialize the validator."""
        self._rules: List[ValidationRule] = []

    def add_rule(
        self,
        key: str,
        validator: Callable[[Any], bool],
        message: str,
        required: bool = False
    ) -> "StateValidator":
        """Add a validation rule.

        Args:
            key: State key to validate
            validator: Function that returns True if valid
            message: Error message if validation fails
            required: Whether the key is required

        Returns:
            Self for chaining
        """
        self._rules.append(ValidationRule(key, validator, message, required))
        return self

    def require_key(self, key: str, message: Optional[str] = None) -> "StateValidator":
        """Require a key to be present.

        Args:
            key: Required key
            message: Error message

        Returns:
            Self for chaining
        """
        msg = message or f"Required key '{key}' is missing"
        return self.add_rule(key, lambda x: x is not None, msg, required=True)

    def validate_type(
        self,
        key: str,
        expected_type: Type,
        message: Optional[str] = None
    ) -> "StateValidator":
        """Validate that a key has the expected type.

        Args:
            key: State key
            expected_type: Expected type
            message: Error message

        Returns:
            Self for chaining
        """
        msg = message or f"Key '{key}' must be of type {expected_type.__name__}"

        def type_validator(value: Any) -> bool:
            return value is None or isinstance(value, expected_type)

        return self.add_rule(key, type_validator, msg)

    def validate_range(
        self,
        key: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        message: Optional[str] = None
    ) -> "StateValidator":
        """Validate that a numeric value is in range.

        Args:
            key: State key
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive)
            message: Error message

        Returns:
            Self for chaining
        """
        msg = message or f"Key '{key}' must be between {min_val} and {max_val}"

        def range_validator(value: Any) -> bool:
            if value is None:
                return True
            if min_val is not None and value < min_val:
                return False
            if max_val is not None and value > max_val:
                return False
            return True

        return self.add_rule(key, range_validator, msg)

    def validate_length(
        self,
        key: str,
        min_len: Optional[int] = None,
        max_len: Optional[int] = None,
        message: Optional[str] = None
    ) -> "StateValidator":
        """Validate the length of a collection.

        Args:
            key: State key
            min_len: Minimum length
            max_len: Maximum length
            message: Error message

        Returns:
            Self for chaining
        """
        msg = message or f"Key '{key}' length must be between {min_len} and {max_len}"

        def length_validator(value: Any) -> bool:
            if value is None:
                return True
            length = len(value)
            if min_len is not None and length < min_len:
                return False
            if max_len is not None and length > max_len:
                return False
            return True

        return self.add_rule(key, length_validator, msg)

    def validate_one_of(
        self,
        key: str,
        valid_values: List[Any],
        message: Optional[str] = None
    ) -> "StateValidator":
        """Validate that value is one of the allowed values.

        Args:
            key: State key
            valid_values: List of valid values
            message: Error message

        Returns:
            Self for chaining
        """
        msg = message or f"Key '{key}' must be one of {valid_values}"

        def one_of_validator(value: Any) -> bool:
            return value is None or value in valid_values

        return self.add_rule(key, one_of_validator, msg)

    def validate_custom_rule(
        self,
        key: str,
        validator: Callable[[Any], bool],
        message: str
    ) -> "StateValidator":
        """Add a custom validation rule.

        Args:
            key: State key
            validator: Custom validation function
            message: Error message

        Returns:
            Self for chaining
        """
        return self.add_rule(key, validator, message)

    def validate(self, state: Dict[str, Any]) -> List[str]:
        """Validate state against all rules.

        Args:
            state: State dictionary to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        for rule in self._rules:
            value = state.get(rule.key)

            # Check required keys
            if rule.required and value is None:
                errors.append(rule.message)
                continue

            # Skip validation if value is None and not required
            if value is None:
                continue

            # Run validator
            try:
                if not rule.validator(value):
                    errors.append(rule.message)
            except Exception as e:
                errors.append(f"Validation error for '{rule.key}': {e}")

        return errors

    def validate_or_raise(self, state: Dict[str, Any]) -> None:
        """Validate state and raise exception if invalid.

        Args:
            state: State dictionary to validate

        Raises:
            ValueError: If validation fails
        """
        errors = self.validate(state)
        if errors:
            raise ValueError(
                f"State validation failed:\n" +
                "\n".join(f"  - {err}" for err in errors)
            )

    def is_valid(self, state: Dict[str, Any]) -> bool:
        """Check if state is valid.

        Args:
            state: State dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        return len(self.validate(state)) == 0

    def clear_rules(self) -> None:
        """Clear all validation rules."""
        self._rules.clear()

    def __len__(self) -> int:
        """Number of validation rules."""
        return len(self._rules)

    def __repr__(self) -> str:
        """String representation."""
        return f"StateValidator({len(self._rules)} rules)"


def validate_state(
    state: Dict[str, Any],
    required_keys: Optional[List[str]] = None,
    type_checks: Optional[Dict[str, Type]] = None
) -> List[str]:
    """Quick validation function.

    Args:
        state: State to validate
        required_keys: List of required keys
        type_checks: Dictionary of key to expected type

    Returns:
        List of error messages
    """
    errors = []

    # Check required keys
    if required_keys:
        for key in required_keys:
            if key not in state or state[key] is None:
                errors.append(f"Required key '{key}' is missing")

    # Check types
    if type_checks:
        for key, expected_type in type_checks.items():
            if key in state and state[key] is not None:
                if not isinstance(state[key], expected_type):
                    errors.append(
                        f"Key '{key}' has type {type(state[key]).__name__}, "
                        f"expected {expected_type.__name__}"
                    )

    return errors
