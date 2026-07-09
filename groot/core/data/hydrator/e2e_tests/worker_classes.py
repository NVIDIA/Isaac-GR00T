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

"""Worker classes for hydrator E2E tests.

This module lives under the main package (groot) instead of under tests/ because
Ray workers need to unpickle these classes. When a class is passed to WorkerGroup,
Ray stores only (module_name, class_name). Workers then do "from <module_name> import
<class_name>". If the class were defined in a test file (e.g. test_worker_pool.py),
the stored module name would be something like "test_worker_pool", which workers
cannot import (tests/ is often not on their path). By defining worker classes here,
the module name is groot.core.data.hydrator.e2e_tests.worker_classes, which
workers can import. See tests/groot/core/data/hydrator/e2e_tests/test_worker_pool.py.
"""

from groot.core.data.hydrator.backends import s3_boto3


# Used by test_worker_pool.py::_check_file_sizes_parallel via WorkerGroup; must be
# in this package so Ray workers can import it when unpickling.
class S3FileSizeWorker:
    """Worker class for xenna WorkerGroup: get S3 file size per path."""

    def __init__(self, working_dir: str) -> None:
        self.working_dir = working_dir
        self._backend: s3_boto3.S3Boto3Backend | None = None

    def _get_backend(self) -> s3_boto3.S3Boto3Backend:
        if self._backend is None:
            self._backend = s3_boto3.S3Boto3Backend(
                working_dir=self.working_dir, credentials_profile="aws"
            )
        return self._backend

    def get_file_size(self, s3_path: str) -> tuple[str, int | None, str | None]:
        """Return (s3_path, size, error_msg). size/error_msg is None on success/error."""
        try:
            backend = self._get_backend()
            size = backend.get_file_size(s3_path)
            return (s3_path, size, None)
        except Exception as e:  # noqa: BLE001
            return (s3_path, None, str(e))
