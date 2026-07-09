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

"""Factory function for creating storage backends.

IMPORTANT: This is an INTERNAL component.
"""

from groot.core.data.hydrator.backends import base, s3_boto3
from groot.core.utils.distributed import worker_logging
from groot.infra.job.data_credentials import DataCredentials

log = worker_logging.get_unified_logger()


def create_backend(
    backend_type: str,
    working_dir: str,
    backend_config: dict | None = None,
) -> base.StorageBackend:
    """Factory function to create storage backend.

    Args:
        backend_type: Backend type ("boto3")
        working_dir: Local working directory root for backend operations
        backend_config: Backend-specific configuration (e.g., transfer settings).
            Relevant keys:
            - ``credentials_profile``: Credentials profile to use (e.g. ``"aws"``
              or ``"default"``).  Required.
            - ``credentials_file``: Path to data credentials file.
            - ``transfer_config``: boto3 TransferConfig parameters dict.

    Returns:
        Initialized StorageBackend instance.

    Raises:
        ValueError: If backend_type is unknown
    """
    backend_config = backend_config or {}

    if backend_type == "boto3":
        credentials_file = backend_config.get(
            "credentials_file", DataCredentials.DEFAULT_CREDENTIALS_FILE
        )
        profile = backend_config.get("credentials_profile")
        if not profile:
            raise ValueError("backend_config must include 'credentials_profile'")
        return s3_boto3.S3Boto3Backend(
            working_dir=working_dir,
            transfer_config=backend_config.get("transfer_config"),
            credentials_profile=profile,
            credentials_file=credentials_file,
            s5cmd_numworkers=backend_config.get("s5cmd_numworkers", 256),
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type}. Supported: boto3")
