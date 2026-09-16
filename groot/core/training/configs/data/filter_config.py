#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Filter configuration for data loading.

Contains configuration options for filtering data during training.
"""

from dataclasses import dataclass
from enum import Enum


class FilterType(str, Enum):
    """Available filter types for data filtering."""

    ANNOTATION_SOURCE = "annotation_source"
    SYS2 = "sys2"


@dataclass
class FilterConfig:
    """Configuration for data filtering during training."""

    # Filter type selection
    filter_type: FilterType | None = FilterType.SYS2
    """The type of filter to use for subtask filtering.

    Options:
    - None: Disable the base data filter
    - "annotation_source": Use AnnotationSourceFilter
    - "sys2": Use Sys2Filter
    """

    # Language/subtask filtering
    subtask_source_filter: str | None = None
    """Optional filter to only use subtasks from a specific source.

    If set, only subtasks where the 'source' field contains this string will be used.
    Common values: "datafactory", "autolabeling"
    If None, all subtasks are used.
    """

    # Sys2Filter-specific options
    language_mode: str = "default"
    """Language mode: 'default', 'task_only', 'subtask_only', 'subtask_fallback', or 'subtask_concat'."""

    subtask_frac: float = 1.0
    """Fraction of time to use subtask language vs episode-level task language.

    1.0 means always use subtask, 0.0 means always use task.
    """
