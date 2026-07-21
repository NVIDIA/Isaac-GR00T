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
"""Enumerates the evaluation backends behind a golden-run eval submission.

``YAM`` — real-robot evaluation by YAM contractors on physical stations through
the GEAR Dashboard YAM eval API.

``YAM_SIMULATION`` — simulation-backed YAM evaluation.

This value is attached to each :class:`YamEvalRequest` at submit time and
emitted as the ``evaluation_type`` Grafana label so dashboards can separate
real-robot scores from simulation scores once both backends exist.
"""

from enum import Enum


class EvaluationType(str, Enum):
    YAM = "YAM"
    YAM_SIMULATION = "YAM_SIMULATION"
