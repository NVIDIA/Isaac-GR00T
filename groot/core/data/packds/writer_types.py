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
from __future__ import annotations

from dataclasses import dataclass

from groot.core.data.packds import constants


@dataclass
class PackDSWriteConfig:
    encode_jpeg: bool = False
    encode_video_blob: bool = True
    embodiment_tag: str | None = None
    min_batch_size: int = constants.DEFAULT_WRITE_BATCH_SIZE
    finalize_on_close: bool = True
    compute_stats: bool = False
    # S3 streaming-uploader tunables (only used when path is remote).
    # Bigger ``s3_upload_threshold_bytes`` means fewer, larger uploads
    # — better network amortisation but worse per-fragment latency
    # under flush.  More ``s3_upload_workers`` parallelises the upload
    # pool but only helps if the queue stays deep enough to feed them.
    s3_upload_threshold_bytes: int = constants.S3_UPLOAD_THRESHOLD_BYTES
    s3_upload_workers: int = constants.S3_UPLOAD_WORKERS
