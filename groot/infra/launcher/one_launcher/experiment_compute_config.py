# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Per-compute-family experiment sizing facts."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_OSMO_EXEC_TIMEOUT = "30d"


@dataclass(frozen=True)
class ExperimentComputeConfig:
    per_device_batch_size: int
    skypilot_queue: str | None = None
    osmo_exec_timeout: str = DEFAULT_OSMO_EXEC_TIMEOUT
