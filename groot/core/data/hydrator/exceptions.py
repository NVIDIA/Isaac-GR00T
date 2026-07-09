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
