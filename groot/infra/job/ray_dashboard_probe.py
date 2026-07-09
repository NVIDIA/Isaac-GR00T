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

"""Ray dashboard readiness probe."""

from __future__ import annotations

from dataclasses import dataclass
import time

import requests


@dataclass(frozen=True)
class RayDashboardProbe:
    """HTTP readiness probe for the Ray Jobs API."""

    local_port: int
    timeout_seconds: float = 5.0

    def is_ready(self) -> tuple[bool, str]:
        url = f"http://127.0.0.1:{self.local_port}/api/jobs/"
        deadline = time.monotonic() + self.timeout_seconds
        last_error = ""
        while True:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code < 500:
                    return True, ""
                last_error = f"{url} returned HTTP {response.status_code}"
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
            if time.monotonic() >= deadline:
                return False, last_error
            time.sleep(1)
