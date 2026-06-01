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

"""Shared ``video_backend`` types for the deployment CLIs and runtime.

The three deployment CLIs (``export_onnx_n1d7``, ``build_trt_pipeline``,
``standalone_inference_script``) all import :data:`VideoBackend` for
their ``--video-backend`` flag, so the set stays aligned in one place.

:data:`VIDEO_BACKEND_CANONICAL` lists every backend the runtime
dispatch in :mod:`gr00t.utils.video_utils` can drive (a superset of the
CLI surface, since some backends are only reachable from internal call
paths). Consistency tests pin ``VideoBackend`` and each per-function
runtime allow-list as subsets of this canonical tuple.
"""

from __future__ import annotations

from typing import Literal


VideoBackend = Literal["decord", "torchvision_av", "torchcodec"]
"""Allowed values for the ``--video-backend`` CLI flag, shared by the
three deployment CLIs."""


VIDEO_BACKEND_CANONICAL = (
    "torchcodec",
    "decord",
    "torchvision_av",
    "ffmpeg",
    "opencv",
    "pyav",
)
"""Every backend the runtime in :mod:`gr00t.utils.video_utils` can
dispatch to. Per-function subsets differ; each must be a subset of this
tuple."""
