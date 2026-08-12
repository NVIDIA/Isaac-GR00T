# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Hydrator-specific exceptions for callers to handle."""


class HydratorException(Exception):
    """Base exception for hydrator errors."""

    pass


class HydratorFileNotFoundError(HydratorException):
    """Raised when a requested remote file does not exist (e.g. S3 404/NoSuchKey).

    Also raised when the file remains in a non-ready state after wait (e.g. download
    failed because the object was missing). Callers can catch this to treat optional
    files as missing and continue (e.g. return None) instead of failing.
    """

    def __init__(self, path: str, message: str | None = None) -> None:
        self.path = path
        self.message = message or f"File not found: {path}"
        super().__init__(self.message)


class HydratorTimeoutError(HydratorException):
    """Raised when a download does not complete within the configured timeout."""

    pass
