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
Download small SimplerEnv sample datasets from HuggingFace for inference testing.

Creates two demo datasets under demo_data/:
  - simplerenv_fractal_sample  (3 episodes from IPEC-COMMUNITY/fractal20220817_data_lerobot)
  - simplerenv_bridge_sample   (3 episodes from IPEC-COMMUNITY/bridge_orig_lerobot)

Both source datasets are already in LeRobot v2 format (per-episode parquet + per-episode mp4),
so this script simply downloads the first few episodes and rewrites the meta files.

Prerequisites:
    pip install huggingface_hub jsonlines pyarrow

Usage:
    python scripts/download_simplerenv_sample.py
    python scripts/download_simplerenv_sample.py --num-episodes 3
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil

import jsonlines


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_NUM_EPISODES = 3

DATASETS = {
    "fractal": {
        "hf_repo": "IPEC-COMMUNITY/fractal20220817_data_lerobot",
        "output_dir": "demo_data/simplerenv_fractal_sample",
        "robot_type": "google_robot",
        "video_keys": ["observation.images.image"],
        "modality_source": "examples/SimplerEnv/fractal_modality.json",
        "embodiment_tag": "SIMPLER_ENV_GOOGLE",
    },
    "bridge": {
        "hf_repo": "IPEC-COMMUNITY/bridge_orig_lerobot",
        "output_dir": "demo_data/simplerenv_bridge_sample",
        "robot_type": "widowx",
        # Bridge has 4 cameras, but the model only uses image_0
        "video_keys": ["observation.images.image_0"],
        "modality_source": "examples/SimplerEnv/bridge_modality.json",
        "embodiment_tag": "SIMPLER_ENV_WIDOWX",
    },
}


def download_sample(
    dataset_key: str,
    num_episodes: int,
    repo_root: Path,
) -> None:
    """Download a small sample from a SimplerEnv dataset."""
    from huggingface_hub import hf_hub_download

    cfg = DATASETS[dataset_key]
    hf_repo = cfg["hf_repo"]
    output_dir = repo_root / cfg["output_dir"]

    if output_dir.exists():
        logger.info(f"Output already exists: {output_dir} — delete it to regenerate.")
        return

    logger.info(f"Downloading {dataset_key} sample ({num_episodes} episodes) from {hf_repo}")

    cache_dir = Path(f"/tmp/simplerenv_{dataset_key}_cache")

    # Download meta files
    for meta_file in [
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.jsonl",
        "meta/episodes.jsonl",
    ]:
        logger.info(f"  {meta_file}...")
        hf_hub_download(
            repo_id=hf_repo,
            repo_type="dataset",
            filename=meta_file,
            local_dir=str(cache_dir),
        )

    # Download first N episode data parquets
    for ep_idx in range(num_episodes):
        fname = f"data/chunk-000/episode_{ep_idx:06d}.parquet"
        logger.info(f"  {fname}...")
        hf_hub_download(
            repo_id=hf_repo,
            repo_type="dataset",
            filename=fname,
            local_dir=str(cache_dir),
        )

    # Download first N episode videos for each video key
    for video_key in cfg["video_keys"]:
        for ep_idx in range(num_episodes):
            fname = f"videos/chunk-000/{video_key}/episode_{ep_idx:06d}.mp4"
            logger.info(f"  {fname}...")
            hf_hub_download(
                repo_id=hf_repo,
                repo_type="dataset",
                filename=fname,
                local_dir=str(cache_dir),
            )

    # Assemble output dataset
    _assemble_sample(cache_dir, output_dir, num_episodes, cfg, repo_root)


def _assemble_sample(
    cache_dir: Path,
    output_dir: Path,
    num_episodes: int,
    cfg: dict,
    repo_root: Path,
) -> None:
    """Assemble the downloaded files into a proper LeRobot v2 demo dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(exist_ok=True)

    # Load source info
    with open(cache_dir / "meta" / "info.json") as f:
        source_info = json.load(f)
    fps = source_info.get("fps", 5)

    # Copy data parquets
    data_chunk_dir = output_dir / "data" / "chunk-000"
    data_chunk_dir.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq

    total_frames = 0
    for ep_idx in range(num_episodes):
        src = cache_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        dst = data_chunk_dir / f"episode_{ep_idx:06d}.parquet"
        shutil.copy2(src, dst)
        table = pq.read_table(str(src))
        total_frames += len(table)
        logger.info(f"  Copied data episode {ep_idx}: {len(table)} frames")

    # Copy video files
    for video_key in cfg["video_keys"]:
        video_chunk_dir = output_dir / "videos" / "chunk-000" / video_key
        video_chunk_dir.mkdir(parents=True, exist_ok=True)
        for ep_idx in range(num_episodes):
            src = cache_dir / "videos" / "chunk-000" / video_key / f"episode_{ep_idx:06d}.mp4"
            dst = video_chunk_dir / f"episode_{ep_idx:06d}.mp4"
            shutil.copy2(src, dst)
            logger.info(f"  Copied video {video_key} episode {ep_idx}")

    # Filter episodes.jsonl to only include our episodes
    src_episodes = cache_dir / "meta" / "episodes.jsonl"
    with jsonlines.open(meta_dir / "episodes.jsonl", mode="w") as writer:
        with jsonlines.open(src_episodes) as reader:
            for rec in reader:
                if rec["episode_index"] < num_episodes:
                    writer.write(rec)

    # Collect task indices from parquet data
    task_indices_used = set()
    for ep_idx in range(num_episodes):
        ep_path = data_chunk_dir / f"episode_{ep_idx:06d}.parquet"
        df = pq.read_table(str(ep_path)).to_pandas()
        if "task_index" in df.columns:
            task_indices_used.update(df["task_index"].unique().tolist())

    # Filter tasks.jsonl to only include tasks referenced by our episodes
    src_tasks = cache_dir / "meta" / "tasks.jsonl"
    with jsonlines.open(meta_dir / "tasks.jsonl", mode="w") as writer:
        with jsonlines.open(src_tasks) as reader:
            for rec in reader:
                if not task_indices_used or rec.get("task_index") in task_indices_used:
                    writer.write(rec)

    # Build video feature entries from source info (only for keys we include)
    video_features = {}
    for video_key in cfg["video_keys"]:
        if video_key in source_info.get("features", {}):
            video_features[video_key] = source_info["features"][video_key]
        else:
            video_features[video_key] = {"dtype": "video", "shape": [256, 256, 3]}

    # Build info.json
    features = {**video_features}
    for key in ["observation.state", "action", "task_index"]:
        if key in source_info.get("features", {}):
            features[key] = source_info["features"][key]

    info = {
        "codebase_version": "v2.1",
        "robot_type": cfg["robot_type"],
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "fps": fps,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "chunks_size": 1000,
        "splits": {"train": f"0:{num_episodes}"},
        "features": features,
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Filter stats.json to only keep keys present in info.json features
    src_stats = cache_dir / "meta" / "stats.json"
    if src_stats.exists():
        with open(src_stats) as f:
            full_stats = json.load(f)
        filtered_stats = {k: v for k, v in full_stats.items() if k in features}
        with open(meta_dir / "stats.json", "w") as f:
            json.dump(filtered_stats, f, indent=2)

    # Copy modality.json from the examples directory
    modality_src = repo_root / cfg["modality_source"]
    shutil.copy2(modality_src, meta_dir / "modality.json")

    logger.info(f"\nDataset created at: {output_dir}")
    logger.info(f"  Episodes: {num_episodes}, Total frames: {total_frames}, FPS: {fps}")


def main():
    parser = argparse.ArgumentParser(
        description="Download small SimplerEnv sample datasets for GR00T inference testing.",
    )
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    for dataset_key in args.datasets:
        download_sample(dataset_key, args.num_episodes, repo_root)

    logger.info("\nTo run inference:")
    for dataset_key in args.datasets:
        cfg = DATASETS[dataset_key]
        logger.info(
            f"\n  uv run python scripts/deployment/standalone_inference_script.py \\\n"
            f"    --model-path nvidia/GR00T-N1.7-3B \\\n"
            f"    --dataset-path {cfg['output_dir']} \\\n"
            f"    --embodiment-tag {cfg['embodiment_tag']} \\\n"
            f"    --traj-ids 0 1 --inference-mode pytorch --action-horizon 8"
        )


if __name__ == "__main__":
    main()
