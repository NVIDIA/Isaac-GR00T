# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
