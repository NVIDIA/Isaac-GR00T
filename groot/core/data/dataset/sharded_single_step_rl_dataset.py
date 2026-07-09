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
