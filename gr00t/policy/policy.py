from abc import ABC, abstractmethod
from typing import Any


class BasePolicy(ABC):
    """Abstract base class for robotic control policies.

    This class defines the interface that all policies must implement, including
    methods for action computation, input/output validation, and state management.

    Subclasses must implement:
        - check_observation(): Validate observation format
        - check_action(): Validate action format
        - _get_action(): Core action computation logic
        - reset(): Reset policy to initial state
    """

    def __init__(self, *, strict: bool = True):
        self.strict = strict

    @abstractmethod
    def check_observation(self, observation: dict[str, Any]) -> None:
        """Check if the observation is valid.

        Args:
            observation: Dictionary containing the current state/observation of the environment

        Raises:
            AssertionError: If the observation is invalid.
        """
        pass

    @abstractmethod
    def check_action(self, action: dict[str, Any]) -> None:
        """Check if the action is valid.

        Args:
            action: Dictionary containing the action to be executed

        Raises:
            AssertionError: If the action is invalid.
        """
        pass

    @abstractmethod
    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compute and return the next action based on current observation.

        This method should be overridden by subclasses to implement policy-specific
        action computation. Input validation is handled by the public get_action() method.

        Args:
            observation: Dictionary containing the current state/observation
            options: Optional configuration dict for action computation

        Returns:
            Tuple of (action, info):
                - action: Dictionary containing the action to be executed
                - info: Dictionary containing additional metadata (e.g., confidence scores)
        """
        pass

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compute and return the next action based on current observation with validation.

        This is the main public interface. It validates the observation, calls
        the internal _get_action(), and validates the resulting action.

        Args:
            observation: Dictionary containing the current state/observation
            options: Optional configuration dict for action computation

        Returns:
            Tuple of (action, info):
                - action: Dictionary containing the validated action
                - info: Dictionary containing additional metadata

        Raises:
            AssertionError/ValueError: If observation or action validation fails
        """
        if self.strict:
            self.check_observation(observation)
        action, info = self._get_action(observation, options)
        if self.strict:
            self.check_action(action)
        return action, info

    @abstractmethod
    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy to its initial state.

        Args:
            options: Dictionary containing the options for the reset

        Returns:
            Dictionary containing the info after resetting the policy
        """
        pass


class PolicyWrapper(BasePolicy):
    """Base wrapper class for composing policy behaviors.

    Note: This base implementation only forwards reset(). Subclasses should
    implement validation logic and additional functionality as needed.
    """

    def __init__(self, policy: BasePolicy, *, strict: bool = True):
        super().__init__(strict=strict)
        self.policy = policy

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.policy.reset(options)
