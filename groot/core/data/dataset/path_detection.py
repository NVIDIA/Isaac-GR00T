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

"""Helpers for identifying dataset path formats used by omni loaders.

The standard PackDS path convention uses a trailing ``.packds`` suffix.
Some already-published datasets also exist under suffixless prefixes inside
``yam_packds_*`` buckets/directories, so omni still recognizes those.
"""

from __future__ import annotations

from pathlib import Path


def is_packds_path(path: str | Path) -> bool:
    """Return True when ``path`` should be treated as a PackDS dataset path.

    Recognises:
    - Paths ending with ``.packds``
    - Paths inside ``yam_packds_*`` buckets/directories
    - ``dc://`` DeltaCAT URIs (all DeltaCAT tables are PackDS-backed)
    """
    normalized = str(path).rstrip("/")
    if normalized.startswith("dc://"):
        return True
    if normalized.endswith(".packds"):
        return True

    parts = [part for part in normalized.split("/") if part]
    return any(part.startswith("yam_packds") for part in parts)
