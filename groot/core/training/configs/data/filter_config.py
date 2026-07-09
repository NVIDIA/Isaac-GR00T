#!/usr/bin/env python
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
"""
Filter configuration for data loading.

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
    filter_type: FilterType = FilterType.SYS2
    """The type of filter to use for subtask filtering.

    Options:
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
    """Fraction of time to use subtask language vs episode-level task language. 1.0 means always use subtask, 0.0 means always use task."""
