# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Storage backend implementations for different remote storage systems.

IMPORTANT: These are INTERNAL components of the AsyncDataServingService.
Users should NEVER import or use backend classes directly.
Backend selection and usage is managed through the service layer:
- config.backend = "boto3" → service instantiates S3Boto3Backend
- service internally uses backend for all file operations

Module Structure:
-----------------
- base.py: StorageBackend abstract interface
- s3_boto3.py: S3Boto3Backend implementation; also parse_s3_uri from utils.io, get_boto3_client_config from utils.aws
- factory.py: create_backend() factory function
"""

from groot.core.data.hydrator.backends.base import StorageBackend
from groot.core.data.hydrator.backends.factory import create_backend
from groot.core.data.hydrator.backends.s3_boto3 import S3Boto3Backend

# Public exports for backward compatibility
__all__ = [
    "StorageBackend",
    "S3Boto3Backend",
    "create_backend",
]
