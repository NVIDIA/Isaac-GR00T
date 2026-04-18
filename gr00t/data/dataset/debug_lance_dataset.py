#!/usr/bin/env python3
# Copyright (c) 2026 Tau Robotics.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone debugging script for LANCE datasets.

This script instantiates the ShardedLanceDataset directly with provided MAIN and CURATED
dataset paths, a URDF path for G1 kinematics, and a simulated modality config. It extracts
a single data point to print array shapes and calculates rolling statistics.

Usage:
  python3 debug_lance_dataset.py --main-dataset gs://bucket/main.lance \
      --curated-dataset gs://bucket/curated.lance \
      --urdf-path path/to/robot.urdf
"""

import argparse
import pprint
from pathlib import Path


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from gr00t.data.types import ModalityConfig

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset.lance_dataset import ShardedLanceDataset
from gr00t.data.dataset.g1_fk import G1ForwardKinematics

# A mock processor that simply returns the message payload directly so we can inspect it.
class MockProcessor:
    def __call__(self, messages):
        return messages[0]["content"]

def main():
    parser = argparse.ArgumentParser(description="Debug Lance Dataset Extraction & Stats")
    parser.add_argument("--main-dataset", type=str, required=True, help="Path to the MAIN Lance dataset")
    parser.add_argument("--curated-dataset", type=str, required=True, help="Path to the CURATED Lance dataset")
    parser.add_argument("--urdf-path", type=str, default=None, help="Path to the G1 URDF file for kinematics")
    parser.add_argument("--chunk-len", type=int, default=50, help="Delta indices (horizon) length to mock")

    args = parser.parse_args()

    print(f"Initializing Lance Dataset debugger...")
    print(f"MAIN: {args.main_dataset}")
    print(f"CURATED: {args.curated_dataset}")
    if args.urdf_path:
        print(f"URDF: {args.urdf_path}")

    dataset_path = {
        "MAIN_DATASET": args.main_dataset,
        "CURATED_DATASET": args.curated_dataset
    }

    # Mock modality configurations specifically for manipulation
    modality_configs = {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["image", "wrist_image"]
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["state"]
        ),
        "action": ModalityConfig(
            delta_indices=list(range(args.chunk_len)),
            modality_keys=["action"]
        )
    }

    try:
        dataset = ShardedLanceDataset(
            dataset_path=dataset_path,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            modality_configs=modality_configs,
            shard_size=10, # small shard size for quick debugging
            episode_sampling_rate=1.0,
            seed=42,
        )
        dataset.processor = MockProcessor()

        # Override the URDF path if explicitly provided
        if args.urdf_path and dataset.g1_fk is not None:
            dataset.g1_fk = G1ForwardKinematics(args.urdf_path)

    except Exception as e:
        print(f"Failed to initialize dataset. Are the paths correct? Error: {e}")
        return

    print(f"\n--- DATASET INITIALIZED ---")
    print(f"Total Shards: {len(dataset)}")

    if len(dataset) > 0:
        print("\n--- EXTRACTING DATAPOINT 0 ---")
        shard = dataset.get_shard(0)
        if len(shard) > 0:
            dp = shard[0]
            print("Extracted VLAStepData successfully!")
            print("Images Keys:", dp.images.keys() if dp.images else None)

            print("\nStates:")
            for k, v in dp.states.items():
                print(f"  {k}: shape {v.shape}")

            print("\nActions (Deltas):")
            for k, v in dp.actions.items():
                print(f"  {k}: shape {v.shape}")

            print(f"\nText Instruction: '{dp.text}'")
        else:
            print("Shard 0 returned empty datapoints.")

    print("\n--- COMPUTING STATISTICS ---")
    try:
        stats = dataset.get_dataset_statistics()
        print("\nNormalization Statistics:")
        pprint.pprint(stats)
    except Exception as e:
        print(f"Failed to compute statistics: {e}")


if __name__ == "__main__":
    main()
