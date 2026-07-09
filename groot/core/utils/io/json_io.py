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
"""Minimal JSON/file IO utilities for data conversion.

Vendored from groot.vla.common.utils.io to avoid pulling the full utils tree.
"""

import json
from pathlib import Path


def load_json(path_or_str, **kwargs):
    """Load JSON from file path."""
    with open(str(path_or_str)) as fp:
        return json.load(fp, **kwargs)


def json_loads(string, **kwargs):
    """Parse JSON string."""
    return json.loads(string, **kwargs)


def load_jsonl(path_or_str, **kwargs):
    """Load JSON Lines file."""
    data = []
    with open(str(path_or_str)) as fp:
        for line in fp:
            data.append(json.loads(line, **kwargs))
    return data


def dump_json(data, path_or_str, **kwargs):
    """Write data as JSON to file."""
    with open(str(path_or_str), "w") as fp:
        json.dump(data, fp, **kwargs)


def json_dumps(data, **kwargs):
    """Serialize data to JSON string."""
    return json.dumps(data, **kwargs)


def dump_jsonl(data, path_or_str):
    """Write data as JSON Lines to file."""
    with open(str(path_or_str), "w") as fp:
        for line in data:
            print(json.dumps(line), file=fp, flush=True)


def write_text(s, path_or_str):
    """Write string to file."""
    with open(str(path_or_str), "w") as fp:
        fp.write(s)
