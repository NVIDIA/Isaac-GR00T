#!/usr/bin/env python3

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

"""
Download a small DROID sample dataset from HuggingFace and convert it to
GR00T LeRobot v2 format suitable for inference with the base model.

The full DROID dataset (lerobot/droid_1.0.1) is ~358 GB with 95k+ episodes
in LeRobot v3.0 format. This script downloads only the first data/video chunks,
then extracts a handful of episodes into the v2.0 per-episode format.

Prerequisites:
    uv pip install jsonlines    # if not already installed

Usage:
    python scripts/download_droid_sample.py
    python scripts/download_droid_sample.py --num-episodes 5 --output-dir demo_data/droid_sample

After running, test with:
    uv run python scripts/deployment/standalone_inference_script.py \\
        --model-path nvidia/GR00T-N1.7-3B \\
        --dataset-path demo_data/droid_sample \\
        --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \\
        --traj-ids 0 1 --inference-mode pytorch --action-horizon 8
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import subprocess

import jsonlines
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def euler_to_rot6d(euler_angles: np.ndarray) -> np.ndarray:
    """Convert euler angles (3D) to rotation 6D representation.

    The 6D rotation representation uses the first two rows of the rotation
    matrix flattened, matching the convention in gr00t/data/state_action/pose.py.

    Args:
        euler_angles: (..., 3) array of euler angles (XYZ convention)

    Returns:
        (..., 6) array of rot6d representation
    """
    shape = euler_angles.shape[:-1]
    flat = euler_angles.reshape(-1, 3)
    rot_matrices = Rotation.from_euler("XYZ", flat).as_matrix()  # (N, 3, 3)
    # Take first two rows of rotation matrix and flatten (matching codebase convention)
    rot6d = rot_matrices[:, :2, :].reshape(-1, 6)  # (N, 6)
    return rot6d.reshape(*shape, 6)


def compute_eef_9d(cartesian_position: np.ndarray) -> np.ndarray:
    """Convert cartesian_position (XYZ + euler 3D) to eef_9d (XYZ + rot6d).

    Args:
        cartesian_position: (..., 6) array [x, y, z, euler_x, euler_y, euler_z]

    Returns:
        (..., 9) array [x, y, z, rot6d_0..5]
    """
    xyz = cartesian_position[..., :3]
    euler = cartesian_position[..., 3:]
    rot6d = euler_to_rot6d(euler)
    return np.concatenate([xyz, rot6d], axis=-1)


REPO_ID = "lerobot/droid_1.0.1"
DEFAULT_OUTPUT_DIR = "demo_data/droid_sample"
DEFAULT_NUM_EPISODES = 3

# The 2 cameras used by the OXE_DROID model config.
# (The dataset also has exterior_2_left, but the model only uses 2 cameras.)
VIDEO_KEYS = [
    "observation.images.exterior_1_left",
    "observation.images.wrist_left",
]


def download_droid_files(cache_dir: Path) -> None:
    """Download minimal files from the DROID v3.0 dataset."""
    from huggingface_hub import hf_hub_download

    logger.info("Downloading DROID v3.0 metadata and first chunks...")

    files_to_download = [
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.parquet",
        "meta/episodes/chunk-000/file-000.parquet",
        "data/chunk-000/file-000.parquet",
    ]
    # Download video file-000 for each camera we need
    for video_key in VIDEO_KEYS:
        files_to_download.append(f"videos/{video_key}/chunk-000/file-000.mp4")

    for fname in files_to_download:
        logger.info(f"  {fname}...")
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=fname,
            local_dir=str(cache_dir),
        )


def extract_episodes(cache_dir: Path, output_dir: Path, num_episodes: int) -> None:
    """Convert downloaded v3.0 data to GR00T LeRobot v2.0 format."""

    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(exist_ok=True)

    # Load source info
    with open(cache_dir / "meta" / "info.json") as f:
        source_info = json.load(f)
    fps = source_info.get("fps", 15)

    # ── Load episodes metadata (v3.0 parquet format) ──
    episodes_pq = cache_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_df = pq.read_table(episodes_pq).to_pandas()

    # Only take episodes from file-000 (the chunk we downloaded)
    episodes = []
    for _, row in episodes_df.iterrows():
        if len(episodes) >= num_episodes:
            break
        # Skip episodes whose data is in a different file
        if int(row["data/file_index"]) != 0:
            continue
        episodes.append(row)

    if not episodes:
        raise RuntimeError("No episodes found in first data chunk")

    # ── Load tasks (v3.0: parquet with task text as index, task_index as column) ──
    tasks_df = pq.read_table(cache_dir / "meta" / "tasks.parquet").to_pandas()
    tasks_df_reset = tasks_df.reset_index()
    # columns after reset: ['index' (= task text), 'task_index']
    task_text_col = tasks_df_reset.columns[0]  # the task text column

    logger.info(f"Extracting {len(episodes)} episodes (fps={fps})")

    # ── Read the consolidated data parquet ──
    data_path = cache_dir / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data_path)
    df = table.to_pandas()

    # ── Create per-episode parquet files ──
    data_chunk_dir = output_dir / "data" / "chunk-000"
    data_chunk_dir.mkdir(parents=True, exist_ok=True)

    episode_records = []
    task_indices_used = set()

    for ep_row in episodes:
        ep_idx = int(ep_row["episode_index"])
        ep_df = df[df["episode_index"] == ep_idx].copy()

        if len(ep_df) == 0:
            logger.warning(f"Episode {ep_idx} has no data rows, skipping")
            continue

        ep_length = len(ep_df)
        new_ep_idx = len(episode_records)

        lang = (
            str(ep_df["language_instruction"].iloc[0])
            if "language_instruction" in ep_df.columns
            else ""
        )
        logger.info(f"  Episode {ep_idx} -> {new_ep_idx}: {ep_length} frames, task={lang[:60]!r}")

        if "task_index" in ep_df.columns:
            task_indices_used.update(ep_df["task_index"].unique().tolist())

        ep_df = ep_df.copy()
        ep_df["episode_index"] = new_ep_idx
        ep_df["index"] = range(len(ep_df))

        # Compute eef_9d (XYZ + rot6d) from cartesian_position (XYZ + euler)
        # for both state and action, as the model expects 17D = eef_9d(9) + gripper(1) + joints(7)
        for prefix in ["observation.state", "action"]:
            cart_col = f"{prefix}.cartesian_position"
            if cart_col in ep_df.columns:
                cart = np.stack(ep_df[cart_col].values)  # (T, 6)
                eef_9d = compute_eef_9d(cart)  # (T, 9)
                ep_df[f"{prefix}.eef_9d"] = list(eef_9d)

        # Rebuild concatenated observation.state = [eef_9d(9), gripper(1), joint(7)] = 17D
        state_parts = []
        for col in [
            "observation.state.eef_9d",
            "observation.state.gripper_position",
            "observation.state.joint_position",
        ]:
            if col in ep_df.columns:
                vals = ep_df[col].values
                arr = np.stack([np.atleast_1d(v) for v in vals])
                state_parts.append(arr)
        if state_parts:
            new_state = np.concatenate(state_parts, axis=-1)  # (T, 17)
            ep_df["observation.state"] = list(new_state)

        # Rebuild concatenated action = [eef_9d(9), gripper(1), joint(7)] = 17D
        action_parts = []
        for col in ["action.eef_9d", "action.gripper_position", "action.joint_position"]:
            if col in ep_df.columns:
                vals = ep_df[col].values
                arr = np.stack([np.atleast_1d(v) for v in vals])
                action_parts.append(arr)
        if action_parts:
            new_action = np.concatenate(action_parts, axis=-1)  # (T, 17)
            ep_df["action"] = list(new_action)

        ep_parquet = data_chunk_dir / f"episode_{new_ep_idx:06d}.parquet"
        ep_df.to_parquet(ep_parquet, index=False)

        episode_records.append(
            {
                "episode_index": new_ep_idx,
                "tasks": list(ep_row["tasks"]) if "tasks" in ep_row.index else [],
                "length": ep_length,
                "_src_row": ep_row,  # keep for video timestamp lookup
            }
        )

    if not episode_records:
        raise RuntimeError("No episodes could be extracted")

    # ── Extract per-episode video segments using timestamps from episodes metadata ──
    for video_key in VIDEO_KEYS:
        video_chunk_dir = output_dir / "videos" / "chunk-000" / video_key
        video_chunk_dir.mkdir(parents=True, exist_ok=True)

        source_video = cache_dir / "videos" / video_key / "chunk-000" / "file-000.mp4"
        if not source_video.exists():
            logger.warning(f"Video not found: {source_video}, skipping")
            continue

        for rec in episode_records:
            new_ep_idx = rec["episode_index"]
            ep_row = rec["_src_row"]

            from_ts = float(ep_row[f"videos/{video_key}/from_timestamp"])
            to_ts = float(ep_row[f"videos/{video_key}/to_timestamp"])
            duration = to_ts - from_ts

            out_video = video_chunk_dir / f"episode_{new_ep_idx:06d}.mp4"

            # Try stream copy first (fast), fall back to re-encode for AV1
            cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-ss",
                f"{from_ts:.6f}",
                "-i",
                str(source_video),
                "-t",
                f"{duration:.6f}",
                "-c",
                "copy",
                str(out_video),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                cmd_reencode = [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{from_ts:.6f}",
                    "-i",
                    str(source_video),
                    "-t",
                    f"{duration:.6f}",
                    "-c:v",
                    "libx264",
                    "-crf",
                    "23",
                    "-preset",
                    "fast",
                    str(out_video),
                ]
                subprocess.run(cmd_reencode, check=True)
                logger.info(
                    f"    {video_key} ep{new_ep_idx}: re-encoded ({rec['length']} frames, {from_ts:.1f}s-{to_ts:.1f}s)"
                )
            else:
                logger.info(
                    f"    {video_key} ep{new_ep_idx}: copied ({rec['length']} frames, {from_ts:.1f}s-{to_ts:.1f}s)"
                )

    # ── Write meta files ──

    # Clean up internal fields before writing
    for rec in episode_records:
        del rec["_src_row"]

    # meta/stats.json — copy from source dataset (used for normalization)
    src_stats = cache_dir / "meta" / "stats.json"
    if src_stats.exists():
        shutil.copy2(src_stats, meta_dir / "stats.json")
        logger.info("  Copied stats.json from source dataset")
    else:
        logger.warning(
            "  stats.json not found in source. Generate it with:\n"
            f"    python gr00t/data/stats.py {output_dir} OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
        )

    # meta/episodes.jsonl
    with jsonlines.open(meta_dir / "episodes.jsonl", mode="w") as writer:
        for rec in episode_records:
            writer.write(rec)

    # meta/tasks.jsonl
    with jsonlines.open(meta_dir / "tasks.jsonl", mode="w") as writer:
        for _, row in tasks_df_reset.iterrows():
            tidx = int(row["task_index"])
            if tidx in task_indices_used:
                writer.write({"task_index": tidx, "task": str(row[task_text_col])})

    # meta/info.json
    info = {
        "codebase_version": "v2.1",
        "robot_type": "droid",
        "total_episodes": len(episode_records),
        "total_frames": sum(r["length"] for r in episode_records),
        "fps": fps,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "chunks_size": 1000,
        "splits": {"train": f"0:{len(episode_records)}"},
        "features": {
            "observation.images.exterior_1_left": {
                "dtype": "video",
                "shape": [180, 320, 3],
            },
            "observation.images.wrist_left": {
                "dtype": "video",
                "shape": [180, 320, 3],
            },
            "observation.state": {"dtype": "float32", "shape": [17]},
            "action": {"dtype": "float32", "shape": [17]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # meta/modality.json
    modality = {
        "state": {
            "eef_9d": {"start": 0, "end": 9},
            "gripper_position": {"start": 9, "end": 10},
            "joint_position": {"start": 10, "end": 17},
        },
        "action": {
            "eef_9d": {"start": 0, "end": 9},
            "gripper_position": {"start": 9, "end": 10},
            "joint_position": {"start": 10, "end": 17},
        },
        "video": {
            "exterior_1_left": {"original_key": "observation.images.exterior_1_left"},
            "wrist_left": {"original_key": "observation.images.wrist_left"},
        },
        "annotation": {
            "language.language_instruction": {"original_key": "task_index"},
        },
    }
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)

    logger.info(f"\nDataset created at: {output_dir}")
    logger.info(f"  Episodes: {len(episode_records)}")
    logger.info(f"  Total frames: {sum(r['length'] for r in episode_records)}")


def main():
    parser = argparse.ArgumentParser(
        description="Download a small DROID sample dataset for GR00T inference testing.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir or "/tmp/droid_download_cache")
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        logger.info(f"Output already exists: {output_dir} — delete it to regenerate.")
        return

    download_droid_files(cache_dir)
    extract_episodes(cache_dir, output_dir, args.num_episodes)

    logger.info("\nTo run inference:")
    logger.info(
        f"  uv run python scripts/deployment/standalone_inference_script.py \\\n"
        f"    --model-path nvidia/GR00T-N1.7-3B \\\n"
        f"    --dataset-path {output_dir} \\\n"
        f"    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \\\n"
        f"    --traj-ids 1 2 --inference-mode pytorch --action-horizon 8"
    )


if __name__ == "__main__":
    main()
