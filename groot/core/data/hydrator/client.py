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

"""IPC client for process-pool workers that need Hydrator operations without creating a Hydrator.

Used by stats generation: workers do direct S3/local for hydrate and glob; get_file, put_file,
and register_files are sent to the main process via a queue/response_dict.
"""

from pathlib import Path
import time
from typing import Any
import uuid

import pyarrow as pa

from groot.core.data.hydrator import exceptions, types
from groot.core.data.hydrator.utils import io

_IPC_ERROR = "_IPC_ERROR"

# Map server exception names to client exception classes so callers can catch the same types.
_IPC_EXCEPTION_MAP = {
    "HydratorException": exceptions.HydratorException,
    "HydratorFileNotFoundError": exceptions.HydratorFileNotFoundError,
    "HydratorTimeoutError": exceptions.HydratorTimeoutError,
}


class HydratorClient:
    """Client for process-pool workers.

    Direct S3/local for hydrate and glob; IPC to main-process Hydrator for
    get_file, put_file, register_files, release_files.
    """

    def __init__(
        self,
        request_queue: Any,
        response_dict: Any,
        credentials_profile: str,
        ipc_timeout: float = 600.0,
    ) -> None:
        self._request_queue = request_queue
        self._response_dict = response_dict
        self._ipc_timeout = ipc_timeout  # seconds; default 10 minutes
        # Fallback profile for buckets not yet seen by DataSourceResolver.
        # hydrate() and glob() prefer get_profile_for_uri() from the resolver's
        # bucket→profile map; this value is only used when that lookup misses.
        self._credentials_profile = credentials_profile

    def _ipc_call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        request_id = uuid.uuid4().hex
        self._request_queue.put((request_id, method, args, kwargs))
        deadline = time.monotonic() + self._ipc_timeout
        while request_id not in self._response_dict:
            if time.monotonic() >= deadline:
                raise exceptions.HydratorTimeoutError(
                    f"IPC call {method!r} did not complete within {self._ipc_timeout}s "
                    "(server may be unresponsive)"
                )
            time.sleep(0.01)
        result = self._response_dict.pop(request_id)
        if isinstance(result, tuple) and result[0] == _IPC_ERROR:
            # New format: (_IPC_ERROR, exc_name, message); re-raise so callers can catch HydratorException etc.
            if len(result) >= 3:
                exc_name, message = result[1], result[2]
                exc_cls = _IPC_EXCEPTION_MAP.get(exc_name)
                if exc_cls is not None:
                    raise exc_cls(message)
            # Old format or unknown: (_IPC_ERROR, message)
            message = result[1] if len(result) >= 2 else str(result)
            raise RuntimeError(f"IPC {method}: {message}")
        return result

    def hydrate(self, path: str) -> list[dict] | pa.Table:
        from groot.core.data.hydrator import data_source_resolver as _dsr

        profile = _dsr.default_resolver().get_profile_for_uri(path) or self._credentials_profile
        return io.consume(path, profile=profile)

    def glob(self, base_path: str, pattern: str) -> list[str]:
        from groot.core.data.hydrator import data_source_resolver as _dsr

        profile = (
            _dsr.default_resolver().get_profile_for_uri(base_path) or self._credentials_profile
        )
        return io.glob_paths(base_path, pattern, profile=profile)

    def get_file(self, path: str) -> str | None:
        path = io.normalize_path(path)
        if not path.startswith("s3://"):
            return path if Path(path).exists() else None
        # Server calls hydrator.get_file(path), which waits for download then returns
        # local path (or S3 URI for DIRECT_READ). Same filesystem as main process.
        return self._ipc_call("get_file", path)

    def put_file(self, remote_path: str, content: bytes | Path) -> None:
        self._ipc_call("put_file", remote_path, content)

    def put_file_async(self, remote_path: str, content: bytes | Path) -> None:
        self._ipc_call("put_file_async", remote_path, content)

    def register_files(self, paths: list[str], modality: Any = None) -> None:
        self._ipc_call("register_files", paths, modality or types.Modality.AUTO)

    def release_files(self, paths: list[str] | None = None) -> None:
        self._ipc_call("release_files", paths)

    def purge_files(self, paths: list[str]) -> None:
        self._ipc_call("purge_files", paths)

    def get_file_state(self, path: str) -> types.FileInfo | None:
        return self._ipc_call("get_file_state", path)

    def shutdown(self, wait: bool = True) -> None:
        pass
