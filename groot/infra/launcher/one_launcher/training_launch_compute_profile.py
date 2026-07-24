# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compute-family launch profile for one-launcher training jobs."""

from __future__ import annotations

from dataclasses import dataclass

_GPUS_PER_NODE_BY_COMPUTE_TYPE = {
    "colossus": 1,
    "gb200": 4,
    "gb300": 4,
    "h100": 8,
}
_POOL_BY_COMPUTE_TYPE = {
    "colossus": "",
    "gb300": "good_cliques",
    "h100": "groot-h100-02",
}
_PER_DEVICE_BATCH_SIZE_BY_COMPUTE_TYPE = {
    "colossus": 12,
    "gb200": 16,
    "gb300": 12,
    "h100": 16,
}


@dataclass(frozen=True)
class TrainingLaunchComputeProfile:
    compute_type: str
    pool_override: str = ""

    def pool(self) -> str:
        if self.pool_override:
            return self.pool_override
        try:
            return _POOL_BY_COMPUTE_TYPE[self.compute_type]
        except KeyError as exc:
            raise ValueError(f"No default pool configured for {self.compute_type}") from exc

    def gpus_per_node(self, num_gpus: int) -> int:
        return _GPUS_PER_NODE_BY_COMPUTE_TYPE.get(self.compute_type, max(1, num_gpus))

    def per_device_batch_size(self) -> int:
        try:
            return _PER_DEVICE_BATCH_SIZE_BY_COMPUTE_TYPE[self.compute_type]
        except KeyError as exc:
            raise ValueError(
                f"No per-device batch size configured for {self.compute_type}"
            ) from exc

    def num_gpus_for_global_batch_size(self, global_batch_size: int) -> int:
        per_device_batch_size = self.per_device_batch_size()
        if global_batch_size % per_device_batch_size != 0:
            raise ValueError(
                "training.global_batch_size must be divisible by "
                f"{self.compute_type} per-device batch size {per_device_batch_size}"
            )
        return global_batch_size // per_device_batch_size

    def num_nodes_for_num_gpus(self, num_gpus: int) -> int:
        gpus_per_node = self.gpus_per_node(num_gpus)
        worker_nodes = (num_gpus + gpus_per_node - 1) // gpus_per_node
        if self.compute_type in {"colossus", "gb300"}:
            # gb300/Colossus bad nodes are handled at cluster allocation time, not runtime.
            return worker_nodes
        # Reserve one extra pod so runtime GPU preflight can drop one bad node.
        return worker_nodes + 1
