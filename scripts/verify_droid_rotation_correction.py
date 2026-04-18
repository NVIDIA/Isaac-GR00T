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
Verify that DROID demo data eef_9d uses the correct rotation convention.

Computes eef_9d from raw cartesian_position two ways (with and without
DROID_EEF_ROTATION_CORRECT) and compares against the pretrained model's
normalization statistics to determine which convention matches.

Usage:
  python scripts/verify_droid_rotation_correction.py
  python scripts/verify_droid_rotation_correction.py --dataset-path demo_data/droid_sample
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DROID_EEF_ROTATION_CORRECT = np.array(
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
    dtype=np.float64,
)

EMBODIMENT_TAG = "oxe_droid_relative_eef_relative_joint"


def _euler_to_eef_9d(cartesian_position: np.ndarray, *, apply_correction: bool) -> np.ndarray:
    """Convert cartesian_position (XYZ + euler) to eef_9d (XYZ + rot6d)."""
    cart = np.asarray(cartesian_position, dtype=np.float64)
    xyz = cart[..., :3].reshape(-1, 3)
    euler = cart[..., 3:].reshape(-1, 3)
    rot = Rotation.from_euler("XYZ", euler).as_matrix()
    if apply_correction:
        rot = rot @ DROID_EEF_ROTATION_CORRECT
    rot6d = rot[:, :2, :].reshape(-1, 6)
    return np.concatenate([xyz, rot6d], axis=-1).astype(np.float32)


def _load_cartesian_positions(dataset_path: str) -> np.ndarray:
    """Load observation.state.cartesian_position from all episode parquets."""
    import pandas as pd

    all_cart = []
    for pq in sorted((Path(dataset_path) / "data").rglob("*.parquet")):
        df = pd.read_parquet(pq)
        if "observation.state.cartesian_position" in df.columns:
            all_cart.append(np.stack(df["observation.state.cartesian_position"].values))
    if not all_cart:
        raise RuntimeError("No cartesian_position found in any parquet file")
    return np.concatenate(all_cart, axis=0)


def _download_eef_stats(hf_repo_id: str) -> dict | None:
    """Download statistics.json and extract eef_9d stats for DROID."""
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=hf_repo_id, filename="statistics.json")
        with open(path) as f:
            stats = json.load(f)
        for tag_key in [EMBODIMENT_TAG, "default"]:
            eef = stats.get(tag_key, {}).get("state", {}).get("eef_9d")
            if eef:
                return eef
    except Exception as e:
        logger.warning(f"Could not download statistics from {hf_repo_id}: {e}")
    return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def verify(dataset_path: str, hf_repo_id: str) -> bool:
    """Run the verification. Returns True if with_correction is the better match."""
    logger.info(f"Loading cartesian_position from {dataset_path} ...")
    cart = _load_cartesian_positions(dataset_path)
    logger.info(f"Loaded {len(cart)} timesteps")

    eef_no_corr = _euler_to_eef_9d(cart, apply_correction=False)
    eef_with_corr = _euler_to_eef_9d(cart, apply_correction=True)

    rot6d_diff = np.abs(eef_no_corr[:, 3:] - eef_with_corr[:, 3:])
    if rot6d_diff.max() < 1e-6:
        logger.error("Correction matrix has no effect — euler angles may be degenerate")
        return False

    logger.info(f"\nComparing against model: {hf_repo_id}")
    model_stats = _download_eef_stats(hf_repo_id)
    if not model_stats:
        logger.error(f"No eef_9d stats found for {hf_repo_id} — cannot verify")
        return False

    # --- Cosine similarity of rot6d mean ---
    model_mean = np.array(model_stats["mean"])
    cos_no = _cosine_similarity(
        np.array([np.mean(eef_no_corr[:, i]) for i in range(3, 9)]), model_mean[3:9]
    )
    cos_with = _cosine_similarity(
        np.array([np.mean(eef_with_corr[:, i]) for i in range(3, 9)]), model_mean[3:9]
    )

    # --- Per-stat RMSE (rot6d dims only) ---
    stat_fns = {"mean": np.mean, "std": np.std, "min": np.min, "max": np.max}
    rmse_results: dict[str, tuple[float, float]] = {}
    for stat_name, fn in stat_fns.items():
        if stat_name not in model_stats:
            continue
        model_rot = np.array(model_stats[stat_name])[3:9]
        vals_no = np.array([fn(eef_no_corr[:, i]) for i in range(3, 9)])
        vals_with = np.array([fn(eef_with_corr[:, i]) for i in range(3, 9)])
        rmse_results[stat_name] = (
            float(np.sqrt(np.mean((vals_no - model_rot) ** 2))),
            float(np.sqrt(np.mean((vals_with - model_rot) ** 2))),
        )

    # --- Print results ---
    logger.info("")
    logger.info("  Cosine similarity of rot6d mean vs pretrained model:")
    logger.info(f"    no_correction:   {cos_no:+.6f}")
    logger.info(f"    with_correction: {cos_with:+.6f}")
    logger.info("")
    logger.info("  RMSE of rot6d stats vs pretrained model (lower = better):")
    logger.info(f"    {'stat':>5}  {'no_correction':>15}  {'with_correction':>15}  {'winner':>15}")
    with_wins = 0
    for stat_name, (rmse_no, rmse_with) in rmse_results.items():
        winner = "with_correction" if rmse_with < rmse_no else "no_correction"
        if rmse_with < rmse_no:
            with_wins += 1
        logger.info(f"    {stat_name:>5}  {rmse_no:>15.6f}  {rmse_with:>15.6f}  {winner:>15}")

    passed = cos_with > cos_no and with_wins >= len(rmse_results) // 2
    logger.info("")
    if passed:
        logger.info("  RESULT: PASS — with_correction matches the pretrained model better")
    else:
        logger.info("  RESULT: FAIL — no_correction appears closer (unexpected)")
    return passed


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset-path", default="demo_data/droid_sample", help="Path to DROID demo dataset"
    )
    parser.add_argument(
        "--hf-repo-id",
        default="nvidia/GR00T-N1.7-3B",
        help="HuggingFace model repo to compare against",
    )
    args = parser.parse_args()
    passed = verify(args.dataset_path, args.hf_repo_id)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
