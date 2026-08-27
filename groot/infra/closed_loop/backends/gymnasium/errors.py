# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Errors shared across isolated Gymnasium runtime boundaries."""


class ResetSeedReplacementError(RuntimeError):
    """Signal that a simulator layout seed is unstable and must be replaced."""

    def __init__(self, env_indices: list[int]) -> None:
        self.env_indices = env_indices
        super().__init__(f"reset requires replacement seeds for envs: {env_indices}")
