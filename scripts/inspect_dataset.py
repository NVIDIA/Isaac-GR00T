#!/usr/bin/env python3
"""
GR00T Dataset Inspector

This script provides detailed inspection and analysis of GR00T LeRobot datasets.
It generates comprehensive reports about dataset structure, modalities, statistics,
and sample data to help users understand their datasets better.

Usage:
    python scripts/inspect_dataset.py /path/to/dataset --output report.json
    python scripts/inspect_dataset.py /path/to/dataset --sample-episodes 5
    python scripts/inspect_dataset.py /path/to/dataset --check-modality-consistency
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas")
    sys.exit(1)


class DatasetInspector:
    """Inspector for detailed GR00T LeRobot dataset analysis."""

    def __init__(self, dataset_path: str):
        """Initialize inspector.

        Args:
            dataset_path: Path to the dataset root directory
        """
        self.dataset_path = Path(dataset_path)
        self.report: Dict[str, Any] = {}

    def inspect(self) -> Dict[str, Any]:
        """Run all inspection checks.

        Returns:
            Dictionary containing inspection results
        """
        print(f"🔎 Inspecting GR00T dataset: {self.dataset_path}")
        print("=" * 70 + "\n")

        # Get dataset structure
        self._inspect_structure()

        # Get metadata insights
        self._inspect_metadata()

        # Get modality configuration
        self._inspect_modality()

        # Get data statistics
        self._inspect_data_statistics()

        # Get embodiment hints
        self._infer_embodiment_hints()

        return self.report

    def _inspect_structure(self) -> None:
        """Inspect and report dataset structure."""
        print("📂 Dataset Structure")
        print("-" * 70)

        structure = {
            "meta": {},
            "videos": {"chunks": 0, "total_videos": 0},
            "data": {"chunks": 0, "total_parquet_files": 0},
        }

        # Meta directory
        meta_dir = self.dataset_path / "meta"
        if meta_dir.exists():
            meta_files = list(meta_dir.glob("*"))
            structure["meta"]["files"] = [f.name for f in meta_files]
            print(f"   Meta files: {', '.join(f.name for f in meta_files)}")

        # Videos directory
        videos_dir = self.dataset_path / "videos"
        if videos_dir.exists():
            chunks = list(videos_dir.glob("chunk-*"))
            structure["videos"]["chunks"] = len(chunks)
            video_files = list(videos_dir.glob("chunk-*/**/*.mp4"))
            structure["videos"]["total_videos"] = len(video_files)
            print(f"   Videos: {len(chunks)} chunks, {len(video_files)} total MP4 files")

        # Data directory
        data_dir = self.dataset_path / "data"
        if data_dir.exists():
            chunks = list(data_dir.glob("chunk-*"))
            structure["data"]["chunks"] = len(chunks)
            parquet_files = list(data_dir.glob("chunk-*/*.parquet"))
            structure["data"]["total_parquet_files"] = len(parquet_files)
            print(
                f"   Data: {len(chunks)} chunks, {len(parquet_files)} total parquet files"
            )

        self.report["structure"] = structure

    def _inspect_metadata(self) -> None:
        """Inspect metadata files."""
        print("\n📋 Metadata Overview")
        print("-" * 70)

        meta_dir = self.dataset_path / "meta"
        metadata = {}

        # Episodes
        episodes_path = meta_dir / "episodes.jsonl"
        if episodes_path.exists():
            with open(episodes_path) as f:
                episodes = [json.loads(line) for line in f if line.strip()]
            print(f"   Episodes: {len(episodes)}")
            if episodes:
                first_episode = episodes[0]
                print(
                    f"      Sample: {len(first_episode.get('tasks', []))} tasks, "
                    f"{first_episode.get('length')} frames"
                )
            metadata["num_episodes"] = len(episodes)

        # Tasks
        tasks_path = meta_dir / "tasks.jsonl"
        if tasks_path.exists():
            with open(tasks_path) as f:
                tasks = [json.loads(line) for line in f if line.strip()]
            print(f"   Tasks: {len(tasks)} unique tasks")
            if tasks:
                sample_task = tasks[0].get("task", "")
                print(f"      Sample: '{sample_task[:60]}{'...' if len(sample_task) > 60 else ''}'")
            metadata["num_tasks"] = len(tasks)

        # Info
        info_path = meta_dir / "info.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            print(f"   Info: {json.dumps(info, indent=6)}")
            metadata["info"] = info

        self.report["metadata"] = metadata

    def _inspect_modality(self) -> None:
        """Inspect modality configuration."""
        print("\n⚙️  Modality Configuration")
        print("-" * 70)

        modality_path = self.dataset_path / "meta" / "modality.json"
        if not modality_path.exists():
            return

        with open(modality_path) as f:
            modality_config = json.load(f)

        modality_info = {}

        # State modality
        if "state" in modality_config:
            state_keys = modality_config["state"]
            total_state_dim = 0
            print(f"   State fields ({len(state_keys)}):")
            for key, indices in state_keys.items():
                dim = indices["end"] - indices["start"]
                total_state_dim += dim
                print(f"      • {key}: dims [{indices['start']}:{indices['end']}] (size: {dim})")
            modality_info["state_keys"] = state_keys
            modality_info["total_state_dim"] = total_state_dim
            print(f"      Total state dimension: {total_state_dim}")

        # Action modality
        if "action" in modality_config:
            action_keys = modality_config["action"]
            total_action_dim = 0
            print(f"\n   Action fields ({len(action_keys)}):")
            for key, indices in action_keys.items():
                dim = indices["end"] - indices["start"]
                total_action_dim += dim
                print(f"      • {key}: dims [{indices['start']}:{indices['end']}] (size: {dim})")
            modality_info["action_keys"] = action_keys
            modality_info["total_action_dim"] = total_action_dim
            print(f"      Total action dimension: {total_action_dim}")

        # Video modality
        if "video" in modality_config:
            video_keys = modality_config["video"]
            print(f"\n   Video streams ({len(video_keys)}):")
            for key, config in video_keys.items():
                original = config.get("original_key", "unknown")
                print(f"      • {key} (original: {original})")
            modality_info["video_keys"] = video_keys

        # Annotations
        if "annotation" in modality_config:
            annotation_keys = modality_config["annotation"]
            print(f"\n   Annotations ({len(annotation_keys)}):")
            for key in annotation_keys.keys():
                print(f"      • {key}")
            modality_info["annotation_keys"] = annotation_keys

        self.report["modality"] = modality_info

    def _inspect_data_statistics(self) -> None:
        """Inspect data statistics."""
        print("\n📊 Data Statistics")
        print("-" * 70)

        data_dir = self.dataset_path / "data"
        parquet_files = list(data_dir.glob("chunk-*/*.parquet"))

        print(f"   Analyzing {len(parquet_files)} parquet files...")

        stats = {
            "num_parquet_files": len(parquet_files),
            "total_frames": 0,
            "total_size_mb": 0,
            "avg_frames_per_file": 0,
            "files_with_annotations": 0,
            "unique_annotation_keys": set(),
        }

        for parquet_file in tqdm(parquet_files, desc="   Processing"):
            try:
                file_size_mb = parquet_file.stat().st_size / (1024 * 1024)
                stats["total_size_mb"] += file_size_mb

                df = pd.read_parquet(parquet_file)
                stats["total_frames"] += len(df)

                # Check for annotations
                annotation_cols = [
                    col for col in df.columns if col.startswith("annotation.")
                ]
                if annotation_cols:
                    stats["files_with_annotations"] += 1
                    stats["unique_annotation_keys"].update(annotation_cols)

            except Exception as e:
                print(f"      Warning: Could not read {parquet_file.name}: {e}")

        stats["unique_annotation_keys"] = list(stats["unique_annotation_keys"])
        if stats["num_parquet_files"] > 0:
            stats["avg_frames_per_file"] = (
                stats["total_frames"] / stats["num_parquet_files"]
            )

        print(f"   Total frames: {stats['total_frames']:,}")
        print(f"   Total size: {stats['total_size_mb']:.2f} MB")
        print(f"   Avg frames per file: {stats['avg_frames_per_file']:.1f}")
        if stats["files_with_annotations"] > 0:
            print(f"   Files with annotations: {stats['files_with_annotations']}")
            print(f"   Unique annotation types: {len(stats['unique_annotation_keys'])}")

        self.report["statistics"] = stats

    def _infer_embodiment_hints(self) -> None:
        """Infer embodiment hints from data."""
        print("\n🤖 Embodiment Hints")
        print("-" * 70)

        hints = {}

        # Check modality for embodiment clues
        modality_info = self.report.get("modality", {})
        if modality_info:
            state_dim = modality_info.get("total_state_dim", 0)
            action_dim = modality_info.get("total_action_dim", 0)

            print(f"   State dimension: {state_dim}")
            print(f"   Action dimension: {action_dim}")

            # Heuristic embodiment detection
            if state_dim == 0 and action_dim == 0:
                print("   ⚠️  Could not detect embodiment from dimensions")
            elif action_dim > 50:
                print("   🤖 Likely humanoid embodiment (high action DOF)")
                hints["likely_embodiment"] = "humanoid"
            elif action_dim > 20:
                print("   🦾 Likely arm/gripper embodiment (medium action DOF)")
                hints["likely_embodiment"] = "arm"
            elif action_dim > 10:
                print("   🏃 Likely multi-robot or bimanual embodiment")
                hints["likely_embodiment"] = "bimanual"
            else:
                print("   🤖 Likely simple embodiment (low action DOF)")
                hints["likely_embodiment"] = "simple"

        self.report["embodiment_hints"] = hints

    def print_summary(self) -> None:
        """Print inspection summary."""
        print("\n" + "=" * 70)
        print("✅ Inspection Complete!")
        print("=" * 70 + "\n")

    def save_report(self, output_path: str) -> None:
        """Save report to JSON file.

        Args:
            output_path: Path to save the report
        """
        # Convert sets to lists for JSON serialization
        def convert_sets(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_sets(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_sets(item) for item in obj]
            return obj

        report_to_save = convert_sets(self.report)

        with open(output_path, "w") as f:
            json.dump(report_to_save, f, indent=2)

        print(f"📁 Report saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Inspect GR00T LeRobot format datasets in detail",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/inspect_dataset.py /path/to/dataset
  python scripts/inspect_dataset.py /path/to/dataset --output report.json
        """,
    )

    parser.add_argument("dataset_path", help="Path to the dataset root directory")
    parser.add_argument(
        "--output",
        type=str,
        help="Save inspection report to JSON file",
    )

    args = parser.parse_args()

    # Inspect dataset
    inspector = DatasetInspector(args.dataset_path)
    report = inspector.inspect()
    inspector.print_summary()

    if args.output:
        inspector.save_report(args.output)

    sys.exit(0)


if __name__ == "__main__":
    main()
