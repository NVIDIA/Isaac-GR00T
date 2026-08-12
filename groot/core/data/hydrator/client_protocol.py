# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Protocol for client I/O: implemented by Hydrator and process-pool HydratorClient."""

from pathlib import Path
from typing import Any, Protocol


class HydratorClientProtocol(Protocol):
    def hydrate(self, path: str) -> Any: ...

    def glob(self, base_path: str, pattern: str) -> list[str]: ...

    def get_file(self, path: str) -> str | None: ...

    def put_file(self, remote_path: str, content: bytes | Path) -> None: ...

    def put_file_async(self, remote_path: str, content: bytes | Path) -> None: ...

    def register_files(self, paths: list[str], modality: Any = None) -> None: ...

    def release_files(self, paths: list[str] | None = None) -> None: ...

    def purge_files(self, paths: list[str]) -> None: ...

    def get_file_state(self, path: str) -> Any: ...
