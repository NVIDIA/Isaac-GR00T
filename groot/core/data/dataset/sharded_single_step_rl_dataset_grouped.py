# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Grouped single-step RL dataset with per-step priority labels.

Subclass of ``ShardedSingleStepDatasetGrouped`` (not the base RL dataset) so that
``shard_dataset()`` label tracking is inherited.  Only ``get_shard()`` is overridden
to produce ``(cur_dp, next_dp)`` tuples with labels on both.
"""

import time

import structlog
from structlog.contextvars import bound_contextvars

from groot.core.data.dataset import sharded_single_step_dataset_grouped
from groot.core.data import types


class ShardedSingleStepRLDatasetGrouped(
    sharded_single_step_dataset_grouped.ShardedSingleStepDatasetGrouped,
):
    """RL variant that yields ``(cur_dp, next_dp)`` tuples with group labels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.allow_padding, (
            "ShardedSingleStepRLDatasetGrouped requires allow_padding=True: next-state "
            "at step_index + action_horizon may exceed episode length near episode end. "
            "Set DataConfig.allow_padding=True in your data yaml."
        )

    def get_shard(self, idx: int) -> list:
        with bound_contextvars(
            component="get_shard",
            dataset_path=str(self.dataset_path),
            shard_idx=int(idx),
            dataset_idx=int(self.dataset_index),
        ):
            start_t = time.perf_counter()
            episodes = self.sharded_episodes[idx]
            ep_indices = [ep_idx for (ep_idx, _) in episodes]
            self.episode_loader.register_video_files_for_episodes(ep_indices)
            shard_labels = (
                self.sharded_group_labels[idx] if self.sharded_group_labels is not None else None
            )
            datapoints = []
            for sample_idx, (ep_idx, step_indices) in enumerate(episodes):
                with bound_contextvars(
                    sample_idx=sample_idx,
                    episode_index=int(ep_idx),
                    episode_length=int(self.episode_loader.get_episode_length(ep_idx)),
                ):
                    sample_t0 = time.perf_counter()
                    episode_data = self.episode_loader[ep_idx]
                    labels_for_ep = shard_labels[sample_idx] if shard_labels is not None else None
                    for j, step_index in enumerate(step_indices):
                        cur_step = int(step_index)
                        next_step = cur_step + self.action_horizon
                        with bound_contextvars(step_index=cur_step):
                            cur_dp = self.get_datapoint(
                                episode_data,
                                cur_step,
                                step_data_id=self._make_step_data_id(ep_idx, cur_step),
                            )
                        with bound_contextvars(step_index=next_step):
                            next_dp = self.get_datapoint(
                                episode_data,
                                next_step,
                                step_data_id=self._make_step_data_id(ep_idx, next_step),
                            )
                        if labels_for_ep is not None:
                            label = int(labels_for_ep[j])
                            cur_dp["step_group"] = label
                            next_dp["step_group"] = label
                        datapoints.append((cur_dp, next_dp))
                    structlog.get_logger().debug(
                        "get_sample_completed",
                        step_indices=[int(step_index) for step_index in step_indices],
                        duration_ms=round((time.perf_counter() - sample_t0) * 1000, 2),
                    )

            self.episode_loader.release_video_files_for_episodes(ep_indices)
            structlog.get_logger().info(
                "get_shard_completed",
                shard_length=int(self.shard_lengths[idx]),
                duration_ms=round((time.perf_counter() - start_t) * 1000, 2),
            )
        return datapoints

    def _make_step_data_id(self, episode_idx: int, step_idx: int) -> types.VLAStepDataId:
        return types.VLAStepDataId(
            dataset_idx=self.dataset_index,
            episode_idx=episode_idx,
            step_idx=step_idx,
        )
