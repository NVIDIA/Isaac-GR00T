from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from gr00t.data.embodiment_tags import EmbodimentTag


class MessageType(Enum):
    START_OF_EPISODE = "start_of_episode"
    END_OF_EPISODE = "end_of_episode"
    EPISODE_STEP = "episode_step"
    IMAGE = "image"
    TEXT = "text"


class ActionRepresentation(Enum):
    RELATIVE = "relative"
    DELTA = "delta"
    ABSOLUTE = "absolute"


class ActionType(Enum):
    EEF = "eef"
    NON_EEF = "non_eef"


class ActionFormat(Enum):
    DEFAULT = "default"
    XYZ_ROT6D = "xyz+rot6d"
    XYZ_ROTVEC = "xyz+rotvec"


@dataclass
class VLAStepData:
    """
    Represents a single step of VLA (Vision-Language-Action) data.

    This is the core data structure returned by datasets, containing raw observation
    and action data that will be processed by the SequenceVLAProcessor.
    """

    # Core data
    images: dict[str, list[np.ndarray]]  # view_name -> list[np.ndarray] (for temporal stacking)
    states: dict[
        str, np.ndarray
    ]  # state_name -> np.ndarray (dim,) for single step or (horizon, dim) for trajectory
    actions: dict[str, np.ndarray]  # action_name -> np.ndarray (horizon, dim) for action chunk
    text: str | None = None  # Optional task description or instruction
    embodiment: EmbodimentTag = (
        EmbodimentTag.NEW_EMBODIMENT
    )  # Optional embodiment tag for cross-embodiment training
    is_demonstration: bool = False  # Whether the step is a demonstration. If True, no loss should be computed for this step.

    # Flexible metadata that can be extended by users
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionConfig:
    rep: ActionRepresentation
    type: ActionType
    format: ActionFormat
    state_key: str | None = None


@dataclass
class ModalityConfig:
    """Configuration for a modality defining how data should be sampled and loaded.

    This class specifies which indices to sample relative to a base index and which
    keys to load for a particular modality (e.g., video, state, action).
    """

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""
    sin_cos_embedding_keys: list[str] | None = None
    """Optional list of keys to apply sin/cos encoding. If None or empty, use min/max normalization for all keys."""
    mean_std_embedding_keys: list[str] | None = None
    """Optional list of keys to apply mean/std normalization. If None or empty, use min/max normalization for all keys."""
    action_configs: list[ActionConfig] | None = None

    def __post_init__(self):
        """Set default values for action-related fields if not specified."""
        if self.action_configs is not None:
            assert len(self.action_configs) == len(self.modality_keys), (
                f"Number of action configs ({len(self.action_configs)}) must match number of modality keys ({len(self.modality_keys)})"
            )
            parsed_action_configs = []
            for action_config in self.action_configs:
                if isinstance(action_config, dict):
                    action_config = ActionConfig(
                        rep=ActionRepresentation[action_config["rep"]],
                        type=ActionType[action_config["type"]],
                        format=ActionFormat[action_config["format"]],
                        state_key=action_config.get("state_key", None),
                    )
                parsed_action_configs.append(action_config)
            self.action_configs = parsed_action_configs
