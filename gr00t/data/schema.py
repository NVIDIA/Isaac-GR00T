# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from enum import Enum
from typing import Optional

from numpydantic import NDArray
from pydantic import BaseModel, Field, field_serializer

from .embodiment_tags import EmbodimentTag

# Common schema


class RotationType(Enum):
    """Type of rotation representation"""

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


class LeRobotModalityField(BaseModel):
    """Metadata for a LeRobot modality field."""

    original_key: Optional[str] = Field(
        default=None,
        description="The original key of the modality in the LeRobot dataset",
    )


class LeRobotStateActionMetadata(LeRobotModalityField):
    """Metadata for a LeRobot modality."""

    start: int = Field(
        ...,
        description="The start index of the modality in the concatenated state/action vector",
    )
    end: int = Field(
        ...,
        description="The end index of the modality in the concatenated state/action vector",
    )
    rotation_type: Optional[RotationType] = Field(
        default=None, description="The type of rotation for the modality"
    )
    absolute: bool = Field(default=True, description="Whether the modality is absolute")
    dtype: str = Field(
        default="float64",
        description="The data type of the modality. Defaults to float64.",
    )
    range: Optional[tuple[float, float]] = Field(
        default=None,
        description="The range of the modality, if applicable. Defaults to None.",
    )
    original_key: Optional[str] = Field(
        default=None,
        description="The original key of the modality in the LeRobot dataset.",
    )


class LeRobotStateMetadata(LeRobotStateActionMetadata):
    """Metadata for a LeRobot state modality."""

    original_key: Optional[str] = Field(
        default="observation.state",  # LeRobot convention for states
        description="The original key of the state modality in the LeRobot dataset",
    )


class LeRobotActionMetadata(LeRobotStateActionMetadata):
    """Metadata for a LeRobot action modality."""

    original_key: Optional[str] = Field(
        default="action",  # LeRobot convention for actions
        description="The original key of the action modality in the LeRobot dataset",
    )


class LeRobotModalityMetadata(BaseModel):
    """Metadata for a LeRobot modality."""

    state: dict[str, LeRobotStateMetadata] = Field(
        ...,
        description="The metadata for the state modality. The keys are the names of each split of the state vector.",
    )
    action: dict[str, LeRobotActionMetadata] = Field(
        ...,
        description="The metadata for the action modality. The keys are the names of each split of the action vector.",
    )
    video: dict[str, LeRobotModalityField] = Field(
        ...,
        description="The metadata for the video modality. The keys are the new names of each video modality.",
    )
    annotation: Optional[dict[str, LeRobotModalityField]] = Field(
        default=None,
        description="The metadata for the annotation modality. The keys are the new names of each annotation modality.",
    )

    def get_key_meta(self, key: str) -> LeRobotModalityField:
        """Get the metadata for a key in the LeRobot modality metadata.

        Args:
            key (str): The key to get the metadata for.

        Returns:
            LeRobotModalityField: The metadata for the key.

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
                    f"Key: {key}, state key {subkey} not found in metadata, available state keys: {self.state.keys()}"
                )
            return self.state[subkey]
        elif modality == "action":
            if subkey not in self.action:
                raise ValueError(
                    f"Key: {key}, action key {subkey} not found in metadata, available action keys: {self.action.keys()}"
                )
            return self.action[subkey]
        elif modality == "video":
            if subkey not in self.video:
                raise ValueError(
                    f"Key: {key}, video key {subkey} not found in metadata, available video keys: {self.video.keys()}"
                )
            return self.video[subkey]
        elif modality == "annotation":
            assert (
                self.annotation is not None
            ), "Trying to get annotation metadata for a dataset with no annotations"
            if subkey not in self.annotation:
                raise ValueError(
                    f"Key: {key}, annotation key {subkey} not found in metadata, available annotation keys: {self.annotation.keys()}"
                )
            return self.annotation[subkey]
        else:
            raise ValueError(f"Key: {key}, unexpected modality: {modality}")


# Dataset schema (parsed from LeRobot schema and simplified)


class DatasetStatisticalValues(BaseModel):
    max: NDArray = Field(..., description="Maximum values")
    min: NDArray = Field(..., description="Minimum values")
    mean: NDArray = Field(..., description="Mean values")
    std: NDArray = Field(..., description="Standard deviation")
    q01: NDArray = Field(..., description="1st percentile values")
    q99: NDArray = Field(..., description="99th percentile values")

    @field_serializer("*", when_used="json")
    def serialize_ndarray(self, v: NDArray) -> list[float]:
        return v.tolist()  # type: ignore


class DatasetStatistics(BaseModel):
    state: dict[str, DatasetStatisticalValues] = Field(..., description="Statistics of the state")
    action: dict[str, DatasetStatisticalValues] = Field(..., description="Statistics of the action")


class VideoMetadata(BaseModel):
    """Metadata of the video modality"""

    resolution: tuple[int, int] = Field(..., description="Resolution of the video")
    channels: int = Field(..., description="Number of channels in the video", gt=0)
    fps: float = Field(..., description="Frames per second", gt=0)


class StateActionMetadata(BaseModel):
    absolute: bool = Field(..., description="Whether the state or action is absolute")
    rotation_type: Optional[RotationType] = Field(None, description="Type of rotation, if any")
    shape: tuple[int, ...] = Field(..., description="Shape of the state or action")
    continuous: bool = Field(..., description="Whether the state or action is continuous")


class DatasetModalities(BaseModel):
    video: dict[str, VideoMetadata] = Field(..., description="Metadata of the video")
    state: dict[str, StateActionMetadata] = Field(..., description="Metadata of the state")
    action: dict[str, StateActionMetadata] = Field(..., description="Metadata of the action")


class DatasetMetadata(BaseModel):
    """Metadata of the trainable dataset

    Changes:
        - Update to use the new RawCommitHashMetadataMetadata_V1_2
    """

    statistics: DatasetStatistics = Field(..., description="Statistics of the dataset")
    modalities: DatasetModalities = Field(..., description="Metadata of the modalities")
    embodiment_tag: EmbodimentTag = Field(..., description="Embodiment tag of the dataset")
