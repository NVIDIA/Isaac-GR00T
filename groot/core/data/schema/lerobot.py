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

import enum
from typing import Generic, TypeVar

import numpy as np
import pydantic

from groot.core.data.schema import embodiment_tags

# Common schema


class RotationType(enum.Enum):
    """Type of rotation representation."""

    AXIS_ANGLE = "axis_angle"
    QUATERNION = "quaternion"
    ROTATION_6D = "rotation_6d"
    MATRIX = "matrix"
    EULER_ANGLES_RPY = "euler_angles_rpy"
    EULER_ANGLES_RYP = "euler_angles_ryp"
    EULER_ANGLES_PRY = "euler_angles_pry"
    EULER_ANGLES_PYR = "euler_angles_pyr"
    EULER_ANGLES_YRP = "euler_angles_yrp"
    EULER_ANGLES_YPR = "euler_angles_ypr"


# LeRobot schema


_OriginalKeyT = TypeVar("_OriginalKeyT", bound=str | None)


class _LeRobotModalityFieldBase(pydantic.BaseModel, Generic[_OriginalKeyT]):
    """Base metadata for a LeRobot modality field."""

    original_key: _OriginalKeyT


class LeRobotModalityField(_LeRobotModalityFieldBase[str | None]):
    """Metadata for a LeRobot modality field."""

    original_key: str | None = pydantic.Field(
        default=None,
        description="The original key of the modality in the LeRobot dataset",
    )


class LeRobotStateActionMetadata(_LeRobotModalityFieldBase[str]):
    """Metadata for a LeRobot modality."""

    start: int = pydantic.Field(
        ...,
        description="The start index of the modality in the concatenated state/action vector",
    )
    end: int = pydantic.Field(
        ...,
        description="The end index of the modality in the concatenated state/action vector",
    )
    rotation_type: RotationType | None = pydantic.Field(
        default=None, description="The type of rotation for the modality"
    )
    absolute: bool = pydantic.Field(default=True, description="Whether the modality is absolute")
    dtype: str = pydantic.Field(
        default="float64",
        description="The data type of the modality. Defaults to float64.",
    )
    range: tuple[float, float] | None = pydantic.Field(
        default=None,
        description="The range of the modality, if applicable. Defaults to None.",
    )
    original_key: str = pydantic.Field(
        ...,
        description="The original key of the modality in the LeRobot dataset.",
    )


class LeRobotStateMetadata(LeRobotStateActionMetadata):
    """Metadata for a LeRobot state modality."""

    original_key: str = pydantic.Field(
        default="observation.state",  # LeRobot convention for states
        description="The original key of the state modality in the LeRobot dataset",
    )


class LeRobotActionMetadata(LeRobotStateActionMetadata):
    """Metadata for a LeRobot action modality."""

    original_key: str = pydantic.Field(
        default="action",  # LeRobot convention for actions
        description="The original key of the action modality in the LeRobot dataset",
    )


class LeRobotModalityMetadata(pydantic.BaseModel):
    """Metadata for a LeRobot modality."""

    state: dict[str, LeRobotStateMetadata] = pydantic.Field(
        ...,
        description=(
            "The metadata for the state modality. "
            "The keys are the names of each split of the state vector."
        ),
    )
    action: dict[str, LeRobotActionMetadata] = pydantic.Field(
        ...,
        description=(
            "The metadata for the action modality. "
            "The keys are the names of each split of the action vector."
        ),
    )
    video: dict[str, LeRobotModalityField] = pydantic.Field(
        ...,
        description="The metadata for the video modality. The keys are the new names of each video modality.",
    )
    annotation: dict[str, LeRobotModalityField] | None = pydantic.Field(
        default=None,
        description=(
            "The metadata for the annotation modality. "
            "The keys are the new names of each annotation modality."
        ),
    )

    @pydantic.model_validator(mode="after")
    def check_original_keys(self):
        # Generate default original keys for state
        for key in self.state:
            if self.state[key].original_key is None:
                self.state[key].original_key = "observation.state"

        # Generate default original keys for action
        for key in self.action:
            if self.action[key].original_key is None:
                self.action[key].original_key = "action"

        # Generate default original keys for video
        for key in self.video:
            if self.video[key].original_key is None:
                self.video[key].original_key = "observation.images." + key

        # Generate default original keys for annotation
        if self.annotation is not None:
            for key in self.annotation:
                if self.annotation[key].original_key is None:
                    self.annotation[key].original_key = "annotation." + key

        return self

    def get_key_meta(self, key: str) -> LeRobotModalityField | LeRobotStateActionMetadata:
        """Get the metadata for a key in the LeRobot modality metadata.

        Args:
            key: The key to get the metadata for.

        Returns:
            LeRobotModalityField | LeRobotStateActionMetadata: The metadata for the key.

        Raises:
            ValueError: If the key's subkey is not found or the modality is unexpected.

        Example:
            lerobot_modality_meta = LeRobotModalityMetadata.model_validate(U.load_json(modality_meta_path))
            lerobot_modality_meta.get_key_meta("state.joint_shoulder_y")
            lerobot_modality_meta.get_key_meta("video.main_camera")
            lerobot_modality_meta.get_key_meta("annotation.human.action.task_description")
        """
        split_key = key.split(".")
        modality = split_key[0]
        subkey = ".".join(split_key[1:])
        if modality == "state":
            if subkey not in self.state:
                raise ValueError(
                    f"Key: {key}, state key {subkey} not found in metadata, "
                    f"available state keys: {self.state.keys()}"
                )
            return self.state[subkey]
        elif modality == "action":
            if subkey not in self.action:
                raise ValueError(
                    f"Key: {key}, action key {subkey} not found in metadata, "
                    f"available action keys: {self.action.keys()}"
                )
            return self.action[subkey]
        elif modality == "video":
            if subkey not in self.video:
                raise ValueError(
                    f"Key: {key}, video key {subkey} not found in metadata, "
                    f"available video keys: {self.video.keys()}"
                )
            return self.video[subkey]
        elif modality == "annotation":
            assert (
                self.annotation is not None
            ), "Trying to get annotation metadata for a dataset with no annotations"
            if subkey not in self.annotation:
                raise ValueError(
                    f"Key: {key}, annotation key {subkey} not found in metadata, "
                    f"available annotation keys: {self.annotation.keys()}"
                )
            return self.annotation[subkey]
        else:
            raise ValueError(f"Key: {key}, unexpected modality: {modality}")

    def get_original_key(self, key: str) -> str:
        """Get the original key for a key in the LeRobot modality metadata.

        Args:
            key: The key to get the original key for.

        Returns:
            str: The original key for the key.

        Example:
            lerobot_modality_meta = LeRobotModalityMetadata.model_validate(U.load_json(modality_meta_path))
            lerobot_modality_meta.get_original_key("state.joint_shoulder_y")
            lerobot_modality_meta.get_original_key("video.main_camera")
            lerobot_modality_meta.get_original_key("annotation.human.action.task_description")
        """
        original_key = self.get_key_meta(key).original_key
        assert original_key is not None, f"Key: {key}, original key is not set"
        return original_key


# Dataset schema (parsed from LeRobot schema and simplified)


class DatasetStatisticalValues(pydantic.BaseModel):
    max: np.ndarray = pydantic.Field(..., description="Maximum values")
    min: np.ndarray = pydantic.Field(..., description="Minimum values")
    mean: np.ndarray = pydantic.Field(..., description="Mean values")
    std: np.ndarray = pydantic.Field(..., description="Standard deviation")
    q01: np.ndarray = pydantic.Field(..., description="1st percentile values")
    q99: np.ndarray = pydantic.Field(..., description="99th percentile values")

    model_config = {"arbitrary_types_allowed": True}

    @pydantic.field_serializer("*", when_used="json")
    def serialize_ndarray(self, v: np.ndarray) -> list[float]:
        return v.tolist()  # type: ignore[return-value]

    @pydantic.field_validator("*", mode="before")
    @classmethod
    def validate_ndarray(_cls, v) -> np.ndarray:
        return np.array(v)


class DatasetStatistics(pydantic.BaseModel):
    state: dict[str, DatasetStatisticalValues] = pydantic.Field(
        ..., description="Statistics of the state"
    )
    action: dict[str, DatasetStatisticalValues] = pydantic.Field(
        ..., description="Statistics of the action"
    )


class VideoMetadata(pydantic.BaseModel):
    """Metadata of the video modality."""

    resolution: tuple[int, int] = pydantic.Field(..., description="Resolution of the video")
    channels: int = pydantic.Field(..., description="Number of channels in the video", gt=0)
    fps: float = pydantic.Field(..., description="Frames per second", gt=0)


class StateActionMetadata(pydantic.BaseModel):
    absolute: bool = pydantic.Field(..., description="Whether the state or action is absolute")
    rotation_type: RotationType | None = pydantic.Field(
        None, description="Type of rotation, if any"
    )
    shape: tuple[int, ...] = pydantic.Field(..., description="Shape of the state or action")
    continuous: bool = pydantic.Field(..., description="Whether the state or action is continuous")


class DatasetModalities(pydantic.BaseModel):
    video: dict[str, VideoMetadata] = pydantic.Field(..., description="Metadata of the video")
    state: dict[str, StateActionMetadata] = pydantic.Field(..., description="Metadata of the state")
    action: dict[str, StateActionMetadata] = pydantic.Field(
        ..., description="Metadata of the action"
    )


class DatasetMetadata(pydantic.BaseModel):
    """Metadata of the trainable dataset.

    Changes:
        - Update to use the new RawCommitHashMetadataMetadata_V1_2
    """

    statistics: DatasetStatistics = pydantic.Field(..., description="Statistics of the dataset")
    modalities: DatasetModalities = pydantic.Field(..., description="Metadata of the modalities")
    embodiment_tag: embodiment_tags.EmbodimentTag = pydantic.Field(
        ..., description="Embodiment tag of the dataset"
    )
