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
"""Data credential helpers used by OSS job and data paths."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, ClassVar


@dataclass(frozen=True)
class DataCredentials:
    """Read gear-compatible data credentials for job-launched workloads."""

    DEFAULT_CREDENTIALS_FILE: ClassVar[str] = os.path.expanduser("~/.gear/data_credentials")

    credentials_file: str = DEFAULT_CREDENTIALS_FILE

    def get_s3_credentials(self, profile: str = "default") -> dict[str, Any]:
        """Load an S3 credential profile from the configured JSON file."""
        path = Path(self.credentials_file).expanduser()
        with path.open() as f:
            credentials = json.load(f)

        if profile not in credentials:
            available = ", ".join(sorted(credentials))
            raise ValueError(
                f"Profile {profile!r} not found in {path}; available profiles: {available}"
            )

        profile_credentials = credentials[profile]
        if not isinstance(profile_credentials, dict):
            raise TypeError(f"Profile {profile!r} in {path} must be a JSON object")
        return profile_credentials
