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
