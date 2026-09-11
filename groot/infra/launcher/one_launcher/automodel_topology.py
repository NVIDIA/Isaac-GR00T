# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Typed distributed topology for AutoModel training experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AutoModelTopology:
    data_parallel: int
    context_parallel: int = 1
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    local_batch_size: int = 1
    gradient_accumulation_steps: int = 1

    def validate(self, *, experiment: str, gpus_per_node: int) -> None:
        for name, value in (
            ("data_parallel", self.data_parallel),
            ("context_parallel", self.context_parallel),
            ("tensor_parallel", self.tensor_parallel),
            ("pipeline_parallel", self.pipeline_parallel),
            ("local_batch_size", self.local_batch_size),
            ("gradient_accumulation_steps", self.gradient_accumulation_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Experiment {experiment} topology.{name} must be positive")

        if self.world_size % gpus_per_node:
            raise ValueError(
                f"Experiment {experiment} world size {self.world_size} must fill complete "
                f"{gpus_per_node}-GPU training nodes"
            )
        if self.pipeline_parallel != 1:
            raise ValueError(
                f"Experiment {experiment} pipeline parallelism is not supported in "
                "the One Launcher AutoModel v1 runtime"
            )
        node_local_group = self.context_parallel * self.tensor_parallel
        if node_local_group > gpus_per_node or gpus_per_node % node_local_group:
            raise ValueError(
                f"Experiment {experiment} requires context_parallel * tensor_parallel "
                f"({node_local_group}) to divide one {gpus_per_node}-GPU node"
            )

    @property
    def world_size(self) -> int:
        return (
            self.data_parallel
            * self.context_parallel
            * self.tensor_parallel
            * self.pipeline_parallel
        )

    @property
    def global_batch_size(self) -> int:
        return self.data_parallel * self.local_batch_size * self.gradient_accumulation_steps

    @property
    def model_parallel_size(self) -> int:
        return self.context_parallel * self.tensor_parallel * self.pipeline_parallel

    def training_nodes(self, *, gpus_per_node: int) -> int:
        return self.world_size // gpus_per_node

    def as_dict(self) -> dict[str, int]:
        return {
            "data_parallel": self.data_parallel,
            "context_parallel": self.context_parallel,
            "tensor_parallel": self.tensor_parallel,
            "pipeline_parallel": self.pipeline_parallel,
            "local_batch_size": self.local_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }
