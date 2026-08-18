#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# ruff: noqa: B006, D415, T201, TID252

"""Custom data collators for conversation datasets."""

import random
from typing import Any

import torch

from ..types import VLAStepDataId


class BasicDataCollator:
    """Minimal data collator that stacks tensor fields into a batch.

    Only fields whose values are ``torch.Tensor`` instances are included in the
    returned batch.  Non-tensor fields (e.g. string metadata, integer scalars)
    are silently skipped rather than raising an error.
    """

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        fields = features[0].keys()
        batch = {}
        for key in fields:
            if key == VLAStepDataId.STEP_DATA_ID_KEY:
                batch[key] = [item[key] for item in features]
            elif not isinstance(features[0][key], torch.Tensor):
                continue
            else:
                batch[key] = torch.stack([item[key] for item in features])
        return batch


class FilteringDataCollator:
    """Memory-efficient data collator that filters fields and pads sequences.

    Only keeps the essential fields needed for training:
    - input_ids
    - attention_mask
    - labels
    - Any vision-related fields (pixel_values, image_grid_thw, etc.)
    """

    def __init__(self, pad_token_id: int, padding_side: str = "left", max_length: int = 2048):
        """Initialize the filtering collator.

        Args:
            pad_token_id: Token ID to use for padding
            padding_side: Side to pad on ("left" or "right")
            max_length: Maximum sequence length to prevent OOM
        """
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side
        self.max_length = max_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate and filter batch of samples with memory-efficient padding.

        Args:
            features: List of sample dictionaries

        Returns:
            Filtered batch with only training-relevant fields
        """
        # Fields to keep
        text_fields = ["input_ids", "attention_mask", "labels"]

        batch = {}
        batch_size = len(features)
        # Process text fields with padding
        for key in text_fields:
            if key in features[0]:
                # Determine padding value
                if key == "labels":
                    pad_value = -100
                elif key == "attention_mask":
                    pad_value = 0 if self.padding_side == "left" else 1
                else:  # input_ids
                    pad_value = self.pad_token_id

                # Pre-allocate padded tensor
                padded_batch = torch.full(
                    (batch_size, self.max_length),
                    pad_value,
                    dtype=features[0][key].dtype,
                )

                for idx, sample in enumerate(features):
                    tensor = sample[key]
                    # Remove batch dimension if present
                    if tensor.dim() == 2 and tensor.shape[0] == 1:
                        tensor = tensor.squeeze(0)

                    if tensor.shape[0] > self.max_length:
                        print(
                            f"WARNING: Sequence length {tensor.shape[0]} exceeds max length {self.max_length}"
                        )

                    seq_len = min(tensor.shape[0], self.max_length)
                    if self.padding_side == "left":
                        padded_batch[idx, -seq_len:] = tensor[:seq_len]
                    else:  # right padding
                        padded_batch[idx, :seq_len] = tensor[:seq_len]

                batch[key] = padded_batch

        # process vision fields
        # qwen does not expect a batch dimension, see
        # NOTE: I don't know how well this works with sequences of variable number of images
        if any("pixel_values" in f for f in features):
            batch["pixel_values"] = torch.cat(
                [f["pixel_values"].squeeze(0) for f in features if "pixel_values" in f], dim=0
            )
            for key in [
                "image_grid_thw",
                "image_grid_hws",
            ]:  # Qwen: image_grid_thw, Kimi: image_grid_hws
                image_grid = [f[key] for f in features if key in f]
                if len(image_grid) > 0:
                    batch[key] = torch.cat(
                        image_grid, dim=0
                    )  # this needs batch dimension (num_images_or_videos, 3)
        else:
            print("WARNING: no pixel_values found in features")
            print("features[0]: ", features[0])
        return batch


class MixedTrainingCollator(FilteringDataCollator):
    """Chop up trajectory by turn and pad to the same length

    Only keeps the essential fields needed for training:
    - input_ids
    - attention_mask
    - labels
    - Any vision-related fields (pixel_values, image_grid_thw, etc.)
    """

    def __init__(
        self,
        pad_token_id: int,
        padding_side: str = "left",
        max_length: int = 2048,
        end_of_turn_tokens: list[int] = [153706, 151645, 198],  # action eos, im_end, newline
        random_chop: float = 0.0,
    ):
        """Initialize the filtering collator.

        Args:
            pad_token_id: Token ID to use for padding
            padding_side: Side to pad on ("left" or "right")
            max_length: Maximum sequence length to prevent OOM
            end_of_turn_tokens: List of end-of-turn tokens
            random_chop: Randomly chop up the trajectory by turn, if random.rand() > random_chop, do not chop
        """
        super().__init__(pad_token_id, padding_side, max_length)
        self.end_of_turn_tokens = end_of_turn_tokens
        self.random_chop = random_chop
        self.text_fields = ["input_ids", "attention_mask", "labels"]

    def chop_trajectory_by_turn(self, features: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
        """Chop up the trajectory by turn"""
        # find the end of turn tokens
        # remove batch dimension if present
        new_features = {}
        for key in self.text_fields:
            tensor = features[key]
            if tensor.dim() == 2 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            new_features[key] = tensor

        end_of_turn_indices = [0]
        for i in range(len(new_features["input_ids"]) - len(self.end_of_turn_tokens) + 1):
            if (
                new_features["input_ids"][i : i + len(self.end_of_turn_tokens)].tolist()
                == self.end_of_turn_tokens
            ):
                end_of_turn_indices.append(i + len(self.end_of_turn_tokens))

        # chop up the trajectory by turn
        new_feature_list = []
        max_length = -1
        for i in range(len(end_of_turn_indices) - 1):
            start_idx = end_of_turn_indices[i]
            end_idx = end_of_turn_indices[i + 1]
            max_length = max(max_length, end_idx - start_idx)
            chop_feature = {}
            for key in self.text_fields:
                chop_feature[key] = new_features[key][start_idx:end_idx]
            new_feature_list.append(chop_feature)
        return new_feature_list, max_length

    def chop_trajectories(self, features: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        """Chop up the trajectories by turn"""
        new_feature_list = []
        max_length = -1
        for feature in features:
            nfl, ml = self.chop_trajectory_by_turn(feature)
            new_feature_list.extend(nfl)
            max_length = max(max_length, ml)
        return new_feature_list, max_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate and filter batch of samples with memory-efficient padding.

        Args:
            features: List of sample dictionaries

        Returns:
            Filtered batch with only training-relevant fields
        """
        # decide whether to chop or not
        uniform_random_value = random.uniform(0, 1)
        if uniform_random_value > self.random_chop:
            return super().__call__(features)
        chopped_features, max_length = self.chop_trajectories(features)

        batch = {}
        batch_size = len(chopped_features)
        # Process text fields with padding
        for key in self.text_fields:
            if key in chopped_features[0]:
                # Determine padding value
                if key == "labels":
                    pad_value = -100
                elif key == "attention_mask":
                    pad_value = 0 if self.padding_side == "left" else 1
                else:  # input_ids
                    pad_value = self.pad_token_id

                # Pre-allocate padded tensor
                padded_batch = torch.full(
                    (batch_size, max_length),
                    pad_value,
                    dtype=chopped_features[0][key].dtype,
                )

                for idx, sample in enumerate(chopped_features):
                    tensor = sample[key]
                    # Remove batch dimension if present
                    if tensor.dim() == 2 and tensor.shape[0] == 1:
                        tensor = tensor.squeeze(0)

                    if tensor.shape[0] > max_length:
                        print(
                            f"WARNING: Sequence length {tensor.shape[0]} exceeds max length {max_length}"
                        )

                    seq_len = min(tensor.shape[0], max_length)
                    if self.padding_side == "left":
                        padded_batch[idx, -seq_len:] = tensor[:seq_len]
                    else:  # right padding
                        padded_batch[idx, :seq_len] = tensor[:seq_len]

                batch[key] = padded_batch

        if "pixel_values" in features[0]:
            batch["pixel_values"] = torch.cat(
                [f["pixel_values"].squeeze(0) for f in features], dim=0
            )
            batch["image_grid_thw"] = torch.cat(
                [f["image_grid_thw"] for f in features], dim=0
            )  # this needs batch dimension (num_images_or_videos, 3)
        else:
            print("WARNING: no pixel_values found in features")
            print("features[0]: ", features[0])
        return batch
