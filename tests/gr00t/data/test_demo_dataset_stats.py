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

"""Guard that bundled demo ``stats.json`` dims match ``info.json`` shapes.

A mismatch means committed stats were produced for a different layout and
silently break normalization for consumers that read stats without regenerating.
"""

import json
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_DATA_ROOT = REPO_ROOT / "demo_data"


def _demo_datasets_with_stats() -> list[Path]:
    if not DEMO_DATA_ROOT.is_dir():
        return []
    return sorted(p.parent.parent for p in DEMO_DATA_ROOT.glob("*/meta/stats.json"))


def _load_json(path: Path):
    """Load JSON, returning None for missing / Git-LFS-pointer / unreadable files."""
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def test_demo_data_present_for_guard():
    """Fail if demo_data exists but ships no stats (else the parametrize is a
    silent zero-case pass); skip when demo_data is absent entirely."""
    if not DEMO_DATA_ROOT.is_dir():
        pytest.skip("demo_data/ not present on this runner")
    assert _demo_datasets_with_stats(), (
        f"{DEMO_DATA_ROOT} exists but no dataset has meta/stats.json to guard"
    )


@pytest.mark.parametrize("dataset_dir", _demo_datasets_with_stats(), ids=lambda p: p.name)
def test_demo_dataset_stats_match_info_shapes(dataset_dir: Path):
    """Every float feature's stats dim must equal its declared ``info.json`` shape."""
    info = _load_json(dataset_dir / "meta" / "info.json")
    stats = _load_json(dataset_dir / "meta" / "stats.json")
    if info is None or stats is None:
        pytest.skip(f"{dataset_dir.name}: meta not materialised (Git LFS pointer?)")

    features = info.get("features", {})
    checked = 0
    for name, meta in features.items():
        if name not in stats or not isinstance(stats[name], dict):
            continue
        if "float" not in str(meta.get("dtype", "")):
            continue
        shape = meta.get("shape")
        if not shape:
            continue
        expected_dim = int(shape[-1])
        for stat_key in ("min", "max", "mean", "std"):
            if stat_key not in stats[name]:
                continue
            got_dim = int(np.asarray(stats[name][stat_key]).shape[-1])
            assert got_dim == expected_dim, (
                f"{dataset_dir.name}: stats['{name}']['{stat_key}'] has dim "
                f"{got_dim} but info.json declares shape {shape} (dim {expected_dim}). "
                f"Regenerate with gr00t/data/stats.py."
            )
        checked += 1

    if checked == 0:
        pytest.skip(f"{dataset_dir.name}: no float features with shapes to check")
