# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Resource specifications for Xenna workers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Resources:
    """Resource requirements for a worker.

    Args:
        cpus: Number of CPUs per worker. Fractional values allowed.
        gpus: Number of GPUs per worker. Fractional values allowed (e.g., 0.5 for GPU sharing).
        spmd: If True, GPUs form an SPMD sub-group per logical worker.
            Each logical worker gets ``gpus`` actors (one per GPU), coordinated
            via NCCL. The framework handles rank assignment, NCCL rendezvous
            env vars, and collecting output from rank 0 only.
    """

    cpus: float = 1.0
    gpus: float = 0.0
    spmd: bool = False

    def __post_init__(self) -> None:
        if self.cpus < 0:
            raise ValueError(f"cpus must be >= 0, got {self.cpus}")
        if self.gpus < 0:
            raise ValueError(f"gpus must be >= 0, got {self.gpus}")
        if self.spmd and self.gpus < 2:
            raise ValueError(f"spmd=True requires gpus >= 2, got gpus={self.gpus}")
