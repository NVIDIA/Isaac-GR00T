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
"""Constants for the OSMO-backed Ray launcher."""

from __future__ import annotations

from collections.abc import Mapping

DEFAULT_CPUS = 1
DEFAULT_IMAGE = "rayproject/ray:2.48.0-py310"
DEFAULT_MEMORY_GIB = 32
DEFAULT_RAY_CLUSTER_READY_INTERVAL_SECONDS = 30.0
DEFAULT_RAY_CLUSTER_READY_TIMEOUT_SECONDS = 1800.0
DEFAULT_RAY_DASHBOARD_PORT = 8265
DEFAULT_RAY_DASHBOARD_PROBE_TIMEOUT_SECONDS = 10.0
DEFAULT_RAY_PORT = 6376
DEFAULT_STORAGE_GIB = 32
DEFAULT_TEMPLATE_NAME = "ray_cluster"
DEFAULT_WORKDIR = "/tmp/groot-ray-workdir"  # noqa: S108

DEFAULT_GPUS_BY_TYPE: Mapping[str, int] = {"gb200": 4, "gb300": 4, "h100": 8, "l40": 8}

OSMO_WORKFLOW_TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "FAILED_EXEC_TIMEOUT",
    "FAILED_START_ERROR",
    "FAILED_SERVER_ERROR",
    "FAILED_BACKEND_ERROR",
    "FAILED_QUEUE_TIMEOUT",
    "FAILED_IMAGE_PULL",
    "FAILED_UPSTREAM",
    "FAILED_EVICTED",
    "FAILED_CANCELED",
}
