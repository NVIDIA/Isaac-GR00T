# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download and prepare a small public SO100 LeRobot dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from groot.core.data.stats import generate_stats

DATASET_REPO_ID = "akira-sasaki/gr00t_red_test"
DATASET_REVISION = "1b1e3866216e474bd4dc307c0171bed0b4543638"
JOINT_NAMES = (
    "main_shoulder_pan",
    "main_shoulder_lift",
    "main_elbow_flex",
    "main_wrist_flex",
    "main_wrist_roll",
    "main_gripper",
)


def _low_dim_section(original_key: str) -> dict[str, dict[str, object]]:
    return {
        name: {
            "original_key": original_key,
            "start": index,
            "end": index + 1,
            "rotation_type": None,
            "absolute": True,
            "dtype": "float32",
            "range": None,
        }
        for index, name in enumerate(JOINT_NAMES)
    }


def prepare_dataset(output_dir: Path) -> Path:
    dataset_path = Path(
        snapshot_download(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            revision=DATASET_REVISION,
            local_dir=output_dir,
        )
    )
    modality = {
        "state": _low_dim_section("observation.state"),
        "action": _low_dim_section("action"),
        "video": {
            "image": {"original_key": "observation.images.laptop"},
            "image2": {"original_key": "observation.images.arm"},
        },
        "annotation": {"task": {"original_key": "task_index"}},
    }
    metadata_dir = dataset_path / "meta"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "modality.json").write_text(
        json.dumps(modality, indent=2) + "\n",
        encoding="utf-8",
    )
    generate_stats(dataset_path)
    return dataset_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/data/gr00t_red_test"),
    )
    args = parser.parse_args()
    print(prepare_dataset(args.output_dir).resolve())  # noqa: T201


if __name__ == "__main__":
    main()
