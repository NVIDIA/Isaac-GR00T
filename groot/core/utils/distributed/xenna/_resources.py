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
