# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
