# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Standalone AutoResearch primitives."""

# Public imports define the small API other packages should depend on.
from skills.autoresearch.config import StudyConfig
from skills.autoresearch.study_state import (
    Objective,
    StageState,
    StudyState,
    Trial,
    make_trial_id,
)

# Keep exported names explicit so internal controller pieces remain private.
__all__ = [
    "Objective",
    "StageState",
    "StudyConfig",
    "StudyState",
    "Trial",
    "make_trial_id",
]
