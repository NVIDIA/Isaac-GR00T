from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import Field, PrivateAttr
import torch

from gr00t.data.schema import DatasetMetadata, StateActionMetadata
from gr00t.data.transform.base import InvertibleModalityTransform, ModalityTransform


class ConcatTransform(InvertibleModalityTransform):
    """
    Concatenate the keys according to specified order.
    """

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )

    video_concat_order: list[str] = Field(
        ...,
        description="Concatenation order for each video modality. "
        "Format: ['video.ego_view_pad_res224_freq20', ...]",
    )

    state_concat_order: Optional[list[str]] = Field(
        default=None,
        description="Concatenation order for each state modality. "
        "Format: ['state.position', 'state.velocity', ...].",
    )

    action_concat_order: Optional[list[str]] = Field(
        default=None,
        description="Concatenation order for each action modality. "
        "Format: ['action.position', 'action.velocity', ...].",
    )

    action_dims: dict[str, int] = Field(
        default_factory=dict,
        description="The dimensions of the action keys.",
    )
    state_dims: dict[str, int] = Field(
        default_factory=dict,
        description="The dimensions of the state keys.",
    )

    action_dims_post_transform: dict[str, int] = Field(
        default_factory=dict,
        description="The new dimensions of the action keys after transform is applied.",
    )
    state_dims_post_transform: dict[str, int] = Field(
        default_factory=dict,
        description="The new dimensions of the state keys after transform is applied.",
    )
    # Store the transform pipeline to examine for dimension changes
    _transform_pipeline: List[ModalityTransform] = PrivateAttr(default_factory=list)

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {
                "apply_to",
                "video_concat_order",
                "state_concat_order",
                "action_concat_order",
            }
        else:
            include = kwargs.pop("include", None)

        return super().model_dump(*args, include=include, **kwargs)

    def set_transform_pipeline(self, transforms: List[ModalityTransform]):
        """Set the transform pipeline so this transform can examine it for dimension changes."""
        self._transform_pipeline = transforms

    def _get_target_rotations_from_pipeline(self) -> Dict[str, str]:
        """Extract target_rotations from StateActionTransform instances in the pipeline."""
        target_rotations = {}
        for transform in self._transform_pipeline:
            if hasattr(transform, "target_rotations"):
                transform_target_rotations = getattr(transform, "target_rotations", {})
                if transform_target_rotations:
                    target_rotations.update(transform_target_rotations)
        return target_rotations

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        grouped_keys = {}
        for key in data.keys():
            try:
                modality, _ = key.split(".")
            except:  # noqa: E722
                ### Handle language annotation special case
                if "annotation" in key:
                    modality = "language"
                else:
                    modality = "others"
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)

        if "video" in grouped_keys:
            # Check if keys in video_concat_order, state_concat_order, action_concat_order are
            # ineed contained in the data. If not, then the keys are misspecified
            video_keys = grouped_keys["video"]
            assert self.video_concat_order is not None, f"{self.video_concat_order=}, {video_keys=}"
            assert all(
                item in video_keys for item in self.video_concat_order
            ), f"keys in video_concat_order are misspecified, \n{video_keys=}, \n{self.video_concat_order=}"

            # Process each video view
            unsqueezed_videos = []
            for video_key in self.video_concat_order:
                video_data = data.pop(video_key)
                unsqueezed_video = np.expand_dims(
                    video_data, axis=-4
                )  # [..., H, W, C] -> [..., 1, H, W, C]
                unsqueezed_videos.append(unsqueezed_video)
            # Concatenate along the new axis
            unsqueezed_video = np.concatenate(unsqueezed_videos, axis=-4)  # [..., V, H, W, C]

            # Video
            data["video"] = unsqueezed_video

        # "state"
        if "state" in grouped_keys:
            state_keys = grouped_keys["state"]
            assert self.state_concat_order is not None, f"{self.state_concat_order=}"
            assert all(
                item in state_keys for item in self.state_concat_order
            ), f"keys in state_concat_order are misspecified, \n{state_keys=}, \n{self.state_concat_order=}"
            # Check the state dims
            for key in self.state_concat_order:
                target_shapes = [self.state_dims[key]]
                if self.is_rotation_key(key):
                    target_shapes.extend(
                        [3, 4, 6]
                    )  # 3 -> axis_angle, 4 -> quaternion, 6 -> rotation_6d
                target_shapes.append(self.state_dims[key] * 2)  # Allow for sin-cos transform
                assert (
                    data[key].shape[-1] in target_shapes
                ), f"State dim mismatch for {key=}, {data[key].shape[-1]=}, {target_shapes=}"
            # Concatenate the state keys
            # We'll have StateActionToTensor before this transform, so here we use torch.cat
            data["state"] = torch.cat(
                [data.pop(key) for key in self.state_concat_order], dim=-1
            )  # [T, D_state]

        if "action" in grouped_keys:
            action_keys = grouped_keys["action"]
            assert self.action_concat_order is not None, f"{self.action_concat_order=}"
            # Check if all keys in concat_order are present
            assert set(self.action_concat_order) == set(
                action_keys
            ), f"{set(self.action_concat_order)=}, {set(action_keys)=}"
            # Record the action dims
            for key in self.action_concat_order:
                target_shapes = [self.action_dims[key]]
                if self.is_rotation_key(key):
                    target_shapes.extend(
                        [3, 4, 6]
                    )  # 3 -> axis_angle, 4 -> quaternion, 6 -> rotation_6d
                assert (
                    data[key].shape[-1] in target_shapes
                ), f"Action dim mismatch for {key=}, {data[key].shape[-1]=}, {target_shapes=}"
            # Concatenate the action keys
            # We'll have StateActionToTensor before this transform, so here we use torch.cat
            data["action"] = torch.cat(
                [data.pop(key) for key in self.action_concat_order], dim=-1
            )  # [T, D_action]

        return data

    def unapply(self, data: dict) -> dict:
        start_dim = 0
        assert "action" in data, f"{data.keys()=}"
        # For those dataset without actions (LAPA), we'll never run unapply
        assert self.action_concat_order is not None, f"{self.action_concat_order=}"
        action_tensor = data.pop("action")
        for key in self.action_concat_order:
            if key not in self.action_dims:
                raise ValueError(f"Action dim {key} not found in action_dims.")
            end_dim = start_dim + self.get_state_action_dims_post_transform(key)
            data[key] = action_tensor[..., start_dim:end_dim]
            start_dim = end_dim
        if "state" in data:
            assert self.state_concat_order is not None, f"{self.state_concat_order=}"
            start_dim = 0
            state_tensor = data.pop("state")
            for key in self.state_concat_order:
                end_dim = start_dim + self.get_state_action_dims_post_transform(key)
                data[key] = state_tensor[..., start_dim:end_dim]
                start_dim = end_dim
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)

    def get_modality_metadata(self, key: str) -> StateActionMetadata:
        modality, subkey = key.split(".")
        assert self.dataset_metadata is not None, "Metadata not set"
        modality_config = getattr(self.dataset_metadata.modalities, modality)
        assert subkey in modality_config, f"{subkey=} not found in {modality_config=}"
        assert isinstance(
            modality_config[subkey], StateActionMetadata
        ), f"Expected {StateActionMetadata} for {subkey=}, got {type(modality_config[subkey])=}"
        return modality_config[subkey]

    def get_state_action_dims(self, key: str) -> int:
        """Get the dimension of a state or action key from the dataset metadata."""
        modality_config = self.get_modality_metadata(key)
        shape = modality_config.shape
        assert len(shape) == 1, f"{shape=}"
        return shape[0]

    def get_state_action_dims_post_transform(self, key: str) -> int:
        """
        This function is used to get the dims of the state/action keys after transform is applied.
        It is different from the `get_state_action_dims` function, because this function accounts for
        the case where we apply transforms and the # of dims is change eg. after applying axis_angle transform on
        quaternion, the dims change from 4D to 3D.
        """
        modality_config = self.get_modality_metadata(key)
        shape = modality_config.shape
        assert len(shape) == 1, f"{shape=}"

        if self.is_rotation_key(key):
            target_rotations = self._get_target_rotations_from_pipeline()
            if key in target_rotations:
                target_rotation = target_rotations[key]
                if target_rotation == "axis_angle":
                    return 3
                elif target_rotation == "quaternion":
                    return 4
                elif target_rotation == "rotation_6d":
                    return 6
                elif target_rotation == "euler_angles":
                    return 3
                else:
                    raise ValueError(f"Unknown target rotation type: {target_rotation}")
            else:
                # No target rotation specified, return original dimension
                return shape[0]
        else:
            return shape[0]

    def is_rotation_key(self, key: str) -> bool:
        modality_config = self.get_modality_metadata(key)
        return modality_config.rotation_type is not None

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set the metadata and compute the dimensions of the state and action keys."""
        super().set_metadata(dataset_metadata)
        # Pre-compute the dimensions of the state and action keys
        if self.action_concat_order is not None:
            for key in self.action_concat_order:
                self.action_dims[key] = self.get_state_action_dims(key)
        if self.state_concat_order is not None:
            for key in self.state_concat_order:
                self.state_dims[key] = self.get_state_action_dims(key)
