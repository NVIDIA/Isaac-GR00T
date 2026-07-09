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
"""Enumerates the GPU platform a golden-run training job targets.

Attached to each :class:`YamEvalRequest` at submit time and emitted as the
``compute_type`` Grafana label so dashboards can slice success / partial scores
by GPU family independent of training variant, which is carried separately in
``run_type``.

Each golden-run training job records which value the pipeline is running and
forwards it to downstream submit helpers. Add a new enum member when a new GPU
family or interconnect is introduced.
"""

from enum import Enum


class ComputeType(str, Enum):
    """GPU platforms targeted by golden-run training jobs."""

    GB200 = "gb200"
    GB300 = "gb300"
    H100 = "h100"
