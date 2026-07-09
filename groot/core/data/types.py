# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, ClassVar
import warnings

import numpy as np
import polars as pl
import pyarrow as pa
from transformers import ProcessorMixin
import tyro

from groot.core.data.schema import EmbodimentTag

# Episode container type — a polars DataFrame with typed columns:
#   state/action: pl.Array(pl.Float32, width)  — fixed-size 1D vectors
#   video:        pl.Object                    — arbitrary np.ndarray per cell
#   language:     pl.String                    — text annotations
EpisodeData = pl.DataFrame


def episode_data_from_arrow(table: pa.Table) -> EpisodeData:
    """Convert a pyarrow Table (from parquet) to a polars DataFrame.

    This is useful as a first step before ``_extract_joint_groups`` slices
    the raw state/action vectors into per-joint-group Array columns. At
    this stage, list columns stay as ``pl.List`` (variable-length); they
    become ``pl.Array`` only after slicing to a known fixed width.
    """
    return pl.from_arrow(table)


def build_episode_data(columns: dict[str, list]) -> EpisodeData:
    """Build an EpisodeData DataFrame from a column-oriented dict.

    Chooses optimal polars dtypes per column:
    - 1D np.ndarray → ``pl.Array(pl.Float32, width)``
    - Multi-dim np.ndarray (video frames) → ``pl.Object``
    - Strings / scalars → auto-detected (``pl.String``, ``pl.Int64``, etc.)
    """
    series_list: list[pl.Series] = []
    for name, values in columns.items():
        if not values:
            series_list.append(pl.Series(name, values))
        elif isinstance(values[0], np.ndarray) and values[0].ndim == 1:
            width = len(values[0])
            data = [v.astype(np.float32).tolist() for v in values]
            series_list.append(pl.Series(name, data, dtype=pl.Array(pl.Float32, width)))
        elif isinstance(values[0], np.ndarray):
            series_list.append(pl.Series(name, values, dtype=pl.Object))
        else:
            series_list.append(pl.Series(name, values))
    return pl.DataFrame(series_list)


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
class VLAStepDataId:
    """Uniquely identifies a data point within a training run."""

    STEP_DATA_ID_KEY: ClassVar[str] = "step_data_id"

    dataset_idx: int = 0
    episode_idx: int = 0
    step_idx: int = 0


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
    languages: dict[str, list[str]] = field(
        default_factory=dict
    )  # language_key -> list[str] (horizon)

    # [Deprecated] Optional task description or instruction
    _text: str | None = field(default=None, init=False, repr=False)
    _text_warning_shown: ClassVar[bool] = False
    text: InitVar[str | None] = None

    rl_info: dict[str, np.ndarray] | None = None  # Optional RL info data
    embodiment: EmbodimentTag = (
        EmbodimentTag.UNKNOWN
    )  # Optional embodiment tag for cross-embodiment training
    is_demonstration: bool = (
        False  # Whether the step is a demonstration. If True, no loss should be computed for this step.
    )

    # Flexible metadata that can be extended by users
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self, text: str | None):
        if text is not None:
            self._text = text

    @property
    def text(self) -> str | None:  # noqa: F811
        if not VLAStepData._text_warning_shown:
            warnings.warn(
                "The 'text' attribute is deprecated. Use 'languages' dict instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            VLAStepData._text_warning_shown = True
        return self._text

    @text.setter
    def text(self, value: str | None):
        self._text = value

    @property
    def num_frames(self) -> int:
        """Number of temporal frames per camera view (0 if no video)."""
        if not self.images:
            return 0
        return sum(len(frames) for frames in self.images.values())


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

    normalization_mode: dict[str, str | None] | None = None
    """Per-key normalization mode (e.g. {"eef_9d": "sin_cos", "hand": None}). None falls back to min/max for all keys."""
    action_configs: Annotated[list[ActionConfig] | None, tyro.conf.Suppress] = None
    """Action configs for each modality key. Suppressed from CLI as it's set programmatically."""
    normalize_rotation: bool = True
    """If False, skip rotation normalization for 9D EEF actions (xyz + rot6d). Default True for backward compatibility."""
    exclude_state: bool = False
    """If True, exclude the state from the modality."""

    # [Deprecate Soon] Use normalization_mode instead.
    sin_cos_embedding_keys: list[str] | None = None
    """[Deprecate Soon] Use normalization_mode instead. Keys to apply sin/cos encoding."""
    mean_std_embedding_keys: list[str] | None = None
    """[Deprecate Soon] Use normalization_mode instead. Keys to apply mean/std normalization."""

    @staticmethod
    def _parse_enum(enum_cls, raw):
        """Look up by value first, then by name."""
        try:
            return enum_cls(raw)
        except ValueError:
            return enum_cls[raw]

    def __post_init__(self):
        """Parse and validate action configs."""
        if self.action_configs is not None:
            assert len(self.action_configs) == len(
                self.modality_keys
            ), f"Number of action configs ({len(self.action_configs)}) must match number of modality keys ({len(self.modality_keys)})"
            parsed_action_configs = []
            for action_config in self.action_configs:
                if isinstance(action_config, dict):
                    action_config = ActionConfig(
                        rep=self._parse_enum(ActionRepresentation, action_config["rep"]),
                        type=self._parse_enum(ActionType, action_config["type"]),
                        format=self._parse_enum(ActionFormat, action_config["format"]),
                        state_key=action_config.get("state_key", None),
                    )
                parsed_action_configs.append(action_config)
            self.action_configs = parsed_action_configs


class OnlineDataFilter(ABC):
    @abstractmethod
    def __call__(self, df: EpisodeData, meta: dict | None = None) -> EpisodeData:
        """Filter the data."""
        pass


class VLATokenizer(ABC):
    """Base interface for VLA tokenizers (state and action)."""

    @property
    def vocabulary(self) -> dict[str, int]:
        """Return tokenizer's vocabulary mapping token_string -> token_id."""
        return {}

    @abstractmethod
    def encode(self, data: dict[str, np.ndarray], embodiment_tag: str) -> str:
        """Encode continuous data to string representation."""
        pass

    def decode(self, text: str, embodiment_tag: str) -> dict[str, np.ndarray]:
        """Decode string representation back to continuous data."""
        raise NotImplementedError("Subclasses should implement decode method if needed")

    @abstractmethod
    def set_statistics(self, statistics: dict[str, Any], override: bool = False) -> None:
        """Set normalization statistics."""
        pass

    @abstractmethod
    def get_statistics(self) -> dict[str, Any]:
        """Get normalization statistics."""
        pass

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    @classmethod
    @abstractmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path) -> "VLATokenizer":
        pass

    @abstractmethod
    def save_pretrained(self, save_directory: str | Path) -> list[Path]:
        pass


class BaseProcessor(ProcessorMixin):
    def __call__(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Process a list of messages and return a dictionary of model inputs.

        Args:
            messages (list[dict[str, Any]]): List of messages to process.

        Returns:
            dict[str, Any]: Dictionary of model inputs.

        Example:
        >>> processor = BaseProcessor()
        >>> messages = [
        >>>    {"type": MessageType.START_OF_EPISODE.value, "content": ""},
        >>>    {"type": MessageType.EPISODE_STEP.value, "content": VLAStepData},
        >>>    {"type": MessageType.TEXT.value, "role" : "user", "content": "Please give me the apple"},
        >>>    {"type": MessageType.TEXT.value, "role" : "assistant", "content": "I need to move my left hand to get the apple"},
        >>>    {"type": MessageType.EPISODE_STEP.value, "content": VLAStepData},
        >>>    {"type": MessageType.EPISODE_STEP.value, "content": VLAStepData},
        >>>    {"type": MessageType.END_OF_EPISODE.value, "content": ""},
        >>> ]
        >>> model_input = processor(messages)
        >>> print(model_input)
        """
        raise NotImplementedError("Subclasses must implement __call__")

    def decode_action(
        self,
        action: np.ndarray,
        embodiment_tag: EmbodimentTag,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Decode the action from the model output."""
        raise NotImplementedError("Subclasses must implement decode_action")

    @property
    def collator(self):
        raise NotImplementedError("Subclasses must implement collator")

    @abstractmethod
    def set_statistics(self, statistics: dict[str, Any], override: bool = False) -> None:
        """Set normalization statistics."""
        pass

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def get_modality_configs(self) -> dict[str, dict[str, ModalityConfig]]:
        """Get the modality configurations.

        Returns:
            dict[str, dict[str, ModalityConfig]]: The modality configurations, where
                modality_configs[embodiment_tag][modality] = ModalityConfig
        """
        return getattr(self, "modality_configs")


class ShardedDataset(ABC):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of shards."""
        pass

    @abstractmethod
    def get_shard_length(self, idx: int) -> int:
        """Get the length of the shard at index idx."""
        pass

    @abstractmethod
    def get_shard(self, idx: int) -> list:
        """Get the shard at index idx."""
        pass

    def set_processor(self, processor: BaseProcessor):
        self.processor = processor

    def get_dataset_statistics(self) -> dict[str, Any]:
        """Get the dataset statistics. This is only required for dataloaders for robtics datasets."""
        raise NotImplementedError()

    def set_dataset_statistics(self, statistics: dict[str, Any]) -> None:
        """Install dataset statistics that were computed outside the dataset."""
        raise NotImplementedError()


# # Example chat formats (processor input)
# # Single step
# messages = [
#     {"type": "episode_step", "content": VLAStepData},
# ]
# # Single episode
# messages = [
#     {"type": MessageType.START_OF_EPISODE.value, "content": ""},
#     {"type": MessageType.EPISODE_STEP.value, "content": VLAStepData},
#     {"type": MessageType.EPISODE_STEP.value, "content": VLAStepData},
#     {"type": MessageType.EPISODE_STEP.value, "content": VLAStepData},
#     {"type": MessageType.END_OF_EPISODE.value, "content": ""},
# ]
# # Multiple episodes
# messages = [
#     {"type": MessageType.START_OF_EPISODE.value, "content": ""},
#     {"type": MessageType.EPISODE_STEP.value, "content": VLAStepData},
#     {"type": MessageType.EPISODE_STEP.value, "content": VLAStepData},
#     {"type": MessageType.END_OF_EPISODE.value, "content": ""},
#     {"type": MessageType.START_OF_EPISODE.value, "content": ""},
#     {"type": MessageType.EPISODE_STEP.value, "content": VLAStepData},
#     {"type": MessageType.EPISODE_STEP.value, "content": VLAStepData},
#     {"type": MessageType.END_OF_EPISODE.value, "content": ""},
# ]

# # Example usage
# messages = dataset[idx]
# model_input = processor(messages)
# model_output = model(**model_input) # or model.generate(**model_input)
# decoded_action = processor.decode_action(model_output)
