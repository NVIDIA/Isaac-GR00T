# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from groot.core.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset


class ShardedSingleStepRLDataset(ShardedSingleStepDataset):
    """
    Single-step RL dataset that creates shards from individual timesteps across episodes.
    """

    def get_shard(self, idx: int) -> list:
        """
        Load and process all timesteps in a specific shard.

        Loads the required episodes and extracts all timesteps assigned to this shard,
        applying the configured processor to each timestep.

        Args:
            idx: Shard index to load

        Returns:
            List of processed timesteps ready for model training
        """
        episodes = self.sharded_episodes[idx]
        datapoints = []
        for ep_idx, step_indices in episodes:
            # Load episode data once per episode in shard
            episode_data = self.episode_loader[ep_idx]
            for step_index in step_indices:
                datapoints.append(
                    (
                        self.get_datapoint(episode_data, step_index),
                        self.get_datapoint(episode_data, step_index + self.action_horizon),
                    )
                )
        return datapoints
