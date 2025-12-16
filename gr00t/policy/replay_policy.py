"""Replay Policy implementation for replaying actions from a dataset.

This module provides a policy that replays recorded actions from a LeRobot-style dataset,
with observation validation matching the Gr00tPolicy interface.
"""

from pathlib import Path
from typing import Any

import numpy as np

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.types import ModalityConfig

from .policy import BasePolicy


class ReplayPolicy(BasePolicy):
    """Policy that replays recorded actions from a LeRobot-style dataset.

    This policy loads actions from a dataset and replays them in sequence. It validates
    that incoming observations match the expected format (compatible with Gr00tPolicy).

    The policy expects observations with specific modalities (video, state, language)
    and returns actions in the format defined by the dataset's modality configuration.

    Example:
        >>> policy = ReplayPolicy(
        ...     dataset_path="/path/to/lerobot_dataset",
        ...     modality_configs={
        ...         "video": ModalityConfig(delta_indices=[0], modality_keys=["front_cam"]),
        ...         "state": ModalityConfig(delta_indices=[0], modality_keys=["joint_positions"]),
        ...         "action": ModalityConfig(
        ...             delta_indices=list(range(16)), modality_keys=["joint_velocities"]
        ...         ),
        ...         "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
        ...     },
        ...     execution_horizon=16,
        ... )
        >>> action, info = policy.get_action(observation)
    """

    def __init__(
        self,
        dataset_path: str | Path,
        modality_configs: dict[str, ModalityConfig],
        execution_horizon: int,
        *,
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict[str, Any] | None = None,
        strict: bool = True,
    ):
        """Initialize the Replay Policy.

        Args:
            dataset_path: Path to the LeRobot-style dataset directory
            modality_configs: Dictionary mapping modality names to ModalityConfig objects
                that specify temporal sampling and data keys to load
            execution_horizon: Policy execution horizon during inference. Will determine the number of steps to skip per get_action call.
            video_backend: Video decoding backend ('torchcodec', 'decord', etc.)
            video_backend_kwargs: Additional arguments for the video backend
            strict: Whether to enforce strict input validation (default: True)
        """
        super().__init__(strict=strict)
        self.dataset_path = Path(dataset_path)
        self.modality_configs = modality_configs
        self.episode_index = 0
        self.execution_horizon = execution_horizon

        # Validate modality configs
        for modality in ["video", "state", "action", "language"]:
            if modality not in modality_configs:
                raise ValueError(f"Modality config must contain '{modality}' key")

        # Extract and validate language configuration
        language_keys = self.modality_configs["language"].modality_keys
        language_delta_indices = self.modality_configs["language"].delta_indices
        assert len(language_delta_indices) == 1, "Only one language delta index is supported"
        assert len(language_keys) == 1, "Only one language key is supported"
        self.language_key = language_keys[0]

        # Initialize episode loader for data access
        self.episode_loader = LeRobotEpisodeLoader(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
        )

        # Load the episode data
        self.episode_data = self.episode_loader[self.episode_index]
        self.episode_length = len(self.episode_data)

        # Current step index in the episode
        self.current_step = 0

        # Preload all actions from the episode
        self._preload_actions()

    def _preload_actions(self) -> None:
        """Preload all actions from the current episode for efficient replay."""
        action_keys = self.modality_configs["action"].modality_keys
        self.actions: dict[str, np.ndarray] = {}

        for key in action_keys:
            col_name = f"action.{key}"
            if col_name in self.episode_data.columns:
                # Stack all actions: shape (episode_length, action_dim)
                action_list = []
                for i in range(self.episode_length):
                    action_array = np.array(self.episode_data[col_name].iloc[i]).astype(np.float32)
                    action_list.append(action_array)
                self.actions[key] = np.stack(action_list, axis=0)
            else:
                raise ValueError(f"Action key '{col_name}' not found in episode data")

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate that the observation has the correct structure and types.

        This method ensures that all required modalities are present and that their
        data types, shapes, and dimensions match the expected format (same as Gr00tPolicy).

        Expected observation structure:
            - video: dict[str, np.ndarray[np.uint8, (B, T, H, W, C)]]
                - B: batch size
                - T: temporal horizon (number of frames)
                - H, W: image height and width
                - C: number of channels (must be 3 for RGB)
            - state: dict[str, np.ndarray[np.float32, (B, T, D)]]
                - B: batch size
                - T: temporal horizon (number of state observations)
                - D: state dimension
            - language: dict[str, list[list[str]]]
                - Shape: (B, T) where each element is a string
                - T: temporal horizon (typically 1 for language)

        Args:
            observation: Dictionary containing video, state, and language modalities

        Raises:
            AssertionError: If any validation check fails
        """
        # Check that observation contains all required top-level modality keys
        for modality in ["video", "state", "language"]:
            assert modality in observation, f"Observation must contain a '{modality}' key"
            assert isinstance(observation[modality], dict), (
                f"Observation '{modality}' must be a dictionary. Got {type(observation[modality])}"
            )

        # Track batch size across modalities to ensure consistency
        bs = -1

        # ===== VIDEO VALIDATION =====
        for video_key in self.modality_configs["video"].modality_keys:
            if bs == -1:
                bs = len(observation["video"][video_key])
            else:
                assert len(observation["video"][video_key]) == bs, (
                    f"Video key '{video_key}' must have batch size {bs}. Got {len(observation['video'][video_key])}"
                )

            assert video_key in observation["video"], (
                f"Video key '{video_key}' must be in observation"
            )

            batched_video = observation["video"][video_key]

            assert isinstance(batched_video, np.ndarray), (
                f"Video key '{video_key}' must be a numpy array. Got {type(batched_video)}"
            )

            assert batched_video.dtype == np.uint8, (
                f"Video key '{video_key}' must be a numpy array of type np.uint8. Got {batched_video.dtype}"
            )

            assert batched_video.ndim == 5, (
                f"Video key '{video_key}' must be a numpy array of shape (B, T, H, W, C), got {batched_video.shape}"
            )

            assert batched_video.shape[1] == len(self.modality_configs["video"].delta_indices), (
                f"Video key '{video_key}'s horizon must be {len(self.modality_configs['video'].delta_indices)}. Got {batched_video.shape[1]}"
            )

            assert batched_video.shape[-1] == 3, (
                f"Video key '{video_key}'s channel 'C' must be 3. Got {batched_video.shape[-1]}"
            )

        # ===== STATE VALIDATION =====
        for state_key in self.modality_configs["state"].modality_keys:
            if bs == -1:
                bs = len(observation["state"][state_key])
            else:
                assert len(observation["state"][state_key]) == bs, (
                    f"State key '{state_key}' must have batch size {bs}. Got {len(observation['state'][state_key])}"
                )

            assert state_key in observation["state"], (
                f"State key '{state_key}' must be in observation"
            )

            batched_state = observation["state"][state_key]

            assert isinstance(batched_state, np.ndarray), (
                f"State key '{state_key}' must be a numpy array. Got {type(batched_state)}"
            )

            assert batched_state.dtype == np.float32, (
                f"State key '{state_key}' must be a numpy array of type np.float32. Got {batched_state.dtype}"
            )

            assert batched_state.ndim == 3, (
                f"State key '{state_key}' must be a numpy array of shape (B, T, D), got {batched_state.shape}"
            )

            assert batched_state.shape[1] == len(self.modality_configs["state"].delta_indices), (
                f"State key '{state_key}'s horizon must be {len(self.modality_configs['state'].delta_indices)}. Got {batched_state.shape[1]}"
            )

        # ===== LANGUAGE VALIDATION =====
        for language_key in self.modality_configs["language"].modality_keys:
            if bs == -1:
                bs = len(observation["language"][language_key])
            else:
                assert len(observation["language"][language_key]) == bs, (
                    f"Language key '{language_key}' must have batch size {bs}. Got {len(observation['language'][language_key])}"
                )

            assert language_key in observation["language"], (
                f"Language key '{language_key}' must be in observation"
            )

            batched_language: list[list[str]] = observation["language"][language_key]

            assert isinstance(batched_language, list), (
                f"Language key '{language_key}' must be a list. Got {type(batched_language)}"
            )

            for batch_item in batched_language:
                assert len(batch_item) == len(self.modality_configs["language"].delta_indices), (
                    f"Language key '{language_key}'s horizon must be {len(self.modality_configs['language'].delta_indices)}. Got {len(batch_item)}"
                )

                assert isinstance(batch_item, list), (
                    f"Language batch item must be a list. Got {type(batch_item)}"
                )

                assert len(batch_item) == 1, (
                    f"Language batch item must have exactly one item. Got {len(batch_item)}"
                )

                assert isinstance(batch_item[0], str), (
                    f"Language batch item must be a string. Got {type(batch_item[0])}"
                )

    def check_action(self, action: dict[str, Any]) -> None:
        """Validate that the action has the correct structure and types.

        Expected action structure:
            - action: dict[str, np.ndarray[np.float32, (B, T, D)]]
                - B: batch size
                - T: action horizon (number of future action steps)
                - D: action dimension

        Args:
            action: Dictionary containing action arrays for each action key

        Raises:
            AssertionError: If any validation check fails
        """
        for action_key in self.modality_configs["action"].modality_keys:
            assert action_key in action, f"Action key '{action_key}' must be in action"

            action_arr = action[action_key]

            assert isinstance(action_arr, np.ndarray), (
                f"Action key '{action_key}' must be a numpy array. Got {type(action_arr)}"
            )

            assert action_arr.dtype == np.float32, (
                f"Action key '{action_key}' must be a numpy array of type np.float32. Got {action_arr.dtype}"
            )

            assert action_arr.ndim == 3, (
                f"Action key '{action_key}' must be a numpy array of shape (B, T, D), got {action_arr.shape}"
            )

            action_horizon = (
                self.modality_configs["action"].delta_indices[-1]
                - self.modality_configs["action"].delta_indices[0]
                + 1
            )
            assert action_arr.shape[1] == action_horizon, (
                f"Action key '{action_key}'s horizon must be {action_horizon}. "
                f"Got {action_arr.shape[1]}"
            )

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Replay the next action chunk from the dataset.

        Args:
            observation: Batched observation dictionary (used for validation, not inference)
            options: Optional parameters (currently unused)

        Returns:
            Tuple of (actions_dict, info_dict) where actions_dict contains action chunks
            with shape (B, action_horizon, D) for each action key
        """
        # Infer batch size from observation
        first_video_key = self.modality_configs["video"].modality_keys[0]
        batch_size = observation["video"][first_video_key].shape[0]
        # Note that this can differ form the execution horizon, as the policy can predict more steps than what's actually executed.
        action_horizon = (
            self.modality_configs["action"].delta_indices[-1]
            - self.modality_configs["action"].delta_indices[0]
            + 1
        )
        assert self.execution_horizon <= action_horizon, (
            f"Execution horizon must be less than or equal to the model's action horizon. Got {self.execution_horizon} and {action_horizon}"
        )

        # Extract action chunk starting from current step
        action_chunk = {}
        for key, actions in self.actions.items():
            if self.current_step >= self.episode_length:
                # Past the end of episode: return last action repeated
                chunk = np.tile(actions[-1:], (action_horizon, 1))  # (action_horizon, D)
            else:
                end_step = self.current_step + action_horizon
                if end_step <= self.episode_length:
                    # Normal case: extract without padding
                    chunk = actions[self.current_step : end_step]  # (action_horizon, D)
                else:
                    # Near end of episode: pad with last action
                    remaining = self.episode_length - self.current_step
                    valid_chunk = actions[self.current_step :]  # (remaining, D)
                    padding = np.tile(
                        actions[-1:], (action_horizon - remaining, 1)
                    )  # (action_horizon - remaining, D)
                    chunk = np.concatenate([valid_chunk, padding], axis=0)

            # Expand to batch dimension: (action_horizon, D) -> (B, action_horizon, D)
            action_chunk[key] = np.tile(chunk[np.newaxis, :, :], (batch_size, 1, 1))

        info = {
            "current_step": self.current_step,
            "episode_length": self.episode_length,
            "episode_index": self.episode_index,
        }

        # Advance step counter by execution_horizon (the number of actions the policy will execute)
        self.current_step += self.execution_horizon

        return action_chunk, info

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy to replay from the beginning of the episode.

        Args:
            options: Dictionary containing optional reset options:
                - episode_index: int - Switch to a different episode
                - step_index: int - Start from a specific step within the episode

        Returns:
            Dictionary containing info about the reset state
        """
        # Check if we should switch episodes
        if options is not None:
            if "episode_index" in options:
                new_episode_index = options["episode_index"]
                if new_episode_index != self.episode_index:
                    self.episode_index = new_episode_index
                    self.episode_data = self.episode_loader[self.episode_index]
                    self.episode_length = len(self.episode_data)
                    self._preload_actions()

            if "step_index" in options:
                self.current_step = options["step_index"]
            else:
                self.current_step = 0
        else:
            self.current_step = 0

        return {
            "episode_index": self.episode_index,
            "episode_length": self.episode_length,
            "current_step": self.current_step,
        }

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        """Get the modality configuration.

        Returns:
            Dictionary mapping modality names to their configurations
        """
        return self.modality_configs

    @property
    def num_episodes(self) -> int:
        """Return the total number of episodes in the dataset."""
        return len(self.episode_loader)
