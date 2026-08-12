# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
