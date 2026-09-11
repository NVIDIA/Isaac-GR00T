# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Prepare public LeRobot datasets for the N2 training recipes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

LIBERO_SIM_EMBODIMENT = "libero_sim"
LIBERO_10_REPO_ID = "IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot"
LIBERO_10_REVISION = "e1a223d30b896c1613f270a2bfc63d382b3de7e1"

# The public IPEC LIBERO v2.1 dataset uses standard LeRobot state/action/video
# columns but does not include GR00T's modality map. Language comes from the
# integer task_index column and is resolved through meta/tasks.jsonl.
LIBERO_SIM_MODALITY: dict[str, Any] = {
    "state": {
        "x": {"start": 0, "end": 1},
        "y": {"start": 1, "end": 2},
        "z": {"start": 2, "end": 3},
        "roll": {"start": 3, "end": 4},
        "pitch": {"start": 4, "end": 5},
        "yaw": {"start": 5, "end": 6},
        "gripper": {"start": 6, "end": 8},
    },
    "action": {
        "x": {"start": 0, "end": 1, "absolute": False},
        "y": {"start": 1, "end": 2, "absolute": False},
        "z": {"start": 2, "end": 3, "absolute": False},
        "roll": {"start": 3, "end": 4, "absolute": False},
        "pitch": {"start": 4, "end": 5, "absolute": False},
        "yaw": {"start": 5, "end": 6, "absolute": False},
        "gripper": {"start": 6, "end": 7},
    },
    "video": {
        "image": {"original_key": "observation.images.image"},
        "wrist_image": {"original_key": "observation.images.wrist_image"},
    },
    "annotation": {
        "human.action.task_description": {"original_key": "task_index"},
    },
}


def _validate_libero_info(info: dict[str, Any], *, info_path: Path) -> None:
    features = info.get("features", {})
    expected_shapes = {
        "observation.state": [8],
        "action": [7],
        "observation.images.image": [256, 256, 3],
        "observation.images.wrist_image": [256, 256, 3],
    }
    mismatches = {
        key: {"expected": shape, "actual": features.get(key, {}).get("shape")}
        for key, shape in expected_shapes.items()
        if features.get(key, {}).get("shape") != shape
    }
    if mismatches:
        raise ValueError(
            f"{info_path} is not a compatible public LIBERO v2.1 dataset: {mismatches}"
        )


def _ensure_stats(dataset_path: Path) -> None:
    from groot.core.data.stats import generate_stats

    generate_stats(dataset_path)


def prepare_lerobot_dataset(dataset_path: Path, embodiment_tag: str) -> Path | None:
    """Add missing LIBERO-10 recipe metadata and stats before training starts."""
    if embodiment_tag != LIBERO_SIM_EMBODIMENT:
        return None

    dataset_path = dataset_path.expanduser().resolve()
    meta_dir = dataset_path / "meta"
    info_path = meta_dir / "info.json"
    modality_path = meta_dir / "modality.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot dataset is missing {info_path}")

    _validate_libero_info(json.loads(info_path.read_text(encoding="utf-8")), info_path=info_path)
    if not modality_path.is_file():
        payload = json.dumps(LIBERO_SIM_MODALITY, indent=2, sort_keys=True) + "\n"
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=".modality.",
            suffix=".json",
            dir=meta_dir,
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        try:
            temporary_path.replace(modality_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    _ensure_stats(dataset_path)
    return modality_path
