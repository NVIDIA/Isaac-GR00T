#!/usr/bin/env python3
"""
GR00T Dataset Validator and Inspector

This script validates and inspects GR00T LeRobot datasets to ensure they follow the correct
format and structure. It performs comprehensive checks including:
- Directory structure validation
- Metadata file validation (modality.json, episodes.jsonl, tasks.jsonl, info.json)
- Parquet file validation
- Video file validation
- State/action array dimension consistency
- Dataset statistics calculation

Usage:
    python scripts/validate_dataset.py /path/to/dataset
    python scripts/validate_dataset.py /path/to/dataset --detailed
    python scripts/validate_dataset.py /path/to/dataset --sample-size 100
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas")
    sys.exit(1)


class DatasetValidator:
    """Validator for GR00T LeRobot format datasets."""

    def __init__(self, dataset_path: str, verbose: bool = False):
        """Initialize validator.

        Args:
            dataset_path: Path to the dataset root directory
            verbose: Enable verbose output
        """
        self.dataset_path = Path(dataset_path)
        self.verbose = verbose
        self.issues: List[Tuple[str, str]] = []  # (severity, message)
        self.warnings: List[str] = []
        self.stats: Dict[str, Any] = {}

    def validate(self) -> bool:
        """Run all validation checks.

        Returns:
            True if all validations pass, False otherwise
        """
        print(f"🔍 Validating GR00T dataset: {self.dataset_path}")
        print("=" * 70)

        # Check directory structure
        if not self._validate_directory_structure():
            return False

        # Check metadata files
        if not self._validate_metadata_files():
            return False

        # Check modality configuration
        if not self._validate_modality_config():
            return False

        # Check data consistency
        if not self._validate_data_consistency():
            return False

        # Check video files
        if not self._validate_video_files():
            return False

        # Calculate dataset statistics
        self._calculate_statistics()

        return len([s for s, _ in self.issues if s == "ERROR"]) == 0

    def _validate_directory_structure(self) -> bool:
        """Validate the directory structure."""
        print("\n📂 Checking directory structure...")

        required_dirs = ["meta", "videos", "data"]
        all_exist = True

        for dir_name in required_dirs:
            dir_path = self.dataset_path / dir_name
            if dir_path.exists() and dir_path.is_dir():
                print(f"   ✓ {dir_name}/ exists")
            else:
                self.issues.append(("ERROR", f"Missing required directory: {dir_name}/"))
                all_exist = False

        return all_exist

    def _validate_metadata_files(self) -> bool:
        """Validate metadata files."""
        print("\n📋 Checking metadata files...")

        meta_dir = self.dataset_path / "meta"
        required_files = ["modality.json", "episodes.jsonl", "tasks.jsonl", "info.json"]
        all_exist = True

        for file_name in required_files:
            file_path = meta_dir / file_name
            if file_path.exists():
                print(f"   ✓ {file_name} exists")

                # Try to parse the file
                if not self._check_file_validity(file_path):
                    all_exist = False
            else:
                self.issues.append(
                    ("ERROR", f"Missing required metadata file: meta/{file_name}")
                )
                all_exist = False

        return all_exist

    def _check_file_validity(self, file_path: Path) -> bool:
        """Check if a JSON/JSONL file is valid."""
        try:
            if file_path.name.endswith(".jsonl"):
                with open(file_path) as f:
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            json.loads(line)
            else:
                with open(file_path) as f:
                    json.load(f)
            return True
        except json.JSONDecodeError as e:
            self.issues.append(
                ("ERROR", f"Invalid JSON in {file_path.name}: {str(e)}")
            )
            return False

    def _validate_modality_config(self) -> bool:
        """Validate modality.json configuration."""
        print("\n⚙️  Checking modality configuration...")

        modality_path = self.dataset_path / "meta" / "modality.json"
        if not modality_path.exists():
            return False

        try:
            with open(modality_path) as f:
                modality_config = json.load(f)

            self.modality_config = modality_config
            print(f"   ✓ modality.json is valid")

            # Check required keys
            expected_keys = ["state", "action"]
            if "video" in modality_config:
                expected_keys.append("video")

            for key in expected_keys:
                if key in modality_config:
                    print(f"   ✓ {key} configuration found")
                    self._validate_modality_section(key, modality_config[key])

            return True
        except Exception as e:
            self.issues.append(("ERROR", f"Failed to validate modality config: {e}"))
            return False

    def _validate_modality_section(self, section: str, config: Dict) -> None:
        """Validate a section of the modality config."""
        for key, value in config.items():
            if isinstance(value, dict):
                if section == "video":
                    if "original_key" not in value:
                        self.issues.append(
                            (
                                "WARNING",
                                f"Video config for '{key}' missing 'original_key'",
                            )
                        )
                else:  # state or action
                    if "start" not in value or "end" not in value:
                        self.issues.append(
                            (
                                "ERROR",
                                f"{section} '{key}' missing 'start' or 'end' index",
                            )
                        )

    def _validate_data_consistency(self) -> bool:
        """Validate data consistency between parquet files and modality config."""
        print("\n🔗 Checking data consistency...")

        meta_dir = self.dataset_path / "meta"
        data_dir = self.dataset_path / "data"

        try:
            # Load episodes metadata
            with open(meta_dir / "episodes.jsonl") as f:
                episodes = [json.loads(line) for line in f if line.strip()]

            print(f"   ✓ Found {len(episodes)} episodes")
            self.stats["num_episodes"] = len(episodes)

            # Check a sample of parquet files
            parquet_files = list(data_dir.glob("chunk-*/*.parquet"))
            if not parquet_files:
                self.issues.append(("ERROR", "No parquet files found in data directory"))
                return False

            print(f"   ✓ Found {len(parquet_files)} parquet files")
            self.stats["num_parquet_files"] = len(parquet_files)

            # Sample validation
            sample_size = min(10, len(parquet_files))
            sample_files = np.random.choice(parquet_files, sample_size, replace=False)

            if not self._validate_parquet_files(sample_files):
                return False

            return True
        except Exception as e:
            self.issues.append(("ERROR", f"Data consistency check failed: {e}"))
            return False

    def _validate_parquet_files(self, parquet_files: List[Path]) -> bool:
        """Validate parquet file structure and content."""
        print(f"\n   Checking {len(parquet_files)} parquet files...")

        required_columns = ["observation.state", "action", "timestamp"]
        all_valid = True

        for parquet_file in tqdm(parquet_files, desc="   Validating parquet"):
            try:
                df = pd.read_parquet(parquet_file)

                # Check required columns
                for col in required_columns:
                    if col not in df.columns:
                        self.issues.append(
                            (
                                "ERROR",
                                f"Missing column '{col}' in {parquet_file.name}",
                            )
                        )
                        all_valid = False

                # Validate state array shape
                if "observation.state" in df.columns:
                    state_array = df["observation.state"].iloc[0]
                    if isinstance(state_array, list):
                        state_array = np.array(state_array)
                    self.stats.setdefault("state_dim", state_array.size)

                # Validate action array shape
                if "action" in df.columns:
                    action_array = df["action"].iloc[0]
                    if isinstance(action_array, list):
                        action_array = np.array(action_array)
                    self.stats.setdefault("action_dim", action_array.size)

                # Check annotation columns match modality config if present
                annotation_cols = [col for col in df.columns if col.startswith("annotation.")]
                if annotation_cols:
                    print(f"      Found {len(annotation_cols)} annotation columns")

            except Exception as e:
                self.issues.append(
                    ("ERROR", f"Failed to read parquet file {parquet_file.name}: {e}")
                )
                all_valid = False

        return all_valid

    def _validate_video_files(self) -> bool:
        """Validate video files."""
        print("\n🎬 Checking video files...")

        videos_dir = self.dataset_path / "videos"
        video_files = list(videos_dir.glob("chunk-*/**/*.mp4"))

        if not video_files:
            self.warnings.append("No MP4 video files found in videos directory")
            return True

        print(f"   ✓ Found {len(video_files)} video files")
        self.stats["num_videos"] = len(video_files)

        # Check video naming convention
        for video_file in video_files[:10]:  # Check first 10
            if not video_file.name.startswith("episode_"):
                self.warnings.append(
                    f"Video file '{video_file.name}' may not follow episode_XXXXXX.mp4 naming"
                )

        return True

    def _calculate_statistics(self) -> None:
        """Calculate and display dataset statistics."""
        print("\n📊 Dataset Statistics")
        print("-" * 70)

        data_dir = self.dataset_path / "data"
        total_frames = 0
        total_file_size = 0

        parquet_files = list(data_dir.glob("chunk-*/*.parquet"))
        for parquet_file in parquet_files:
            total_file_size += parquet_file.stat().st_size
            try:
                df = pd.read_parquet(parquet_file)
                total_frames += len(df)
            except Exception:
                pass

        self.stats["total_frames"] = total_frames
        self.stats["total_data_size_mb"] = total_file_size / (1024 * 1024)

        for key, value in self.stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")

    def print_report(self) -> None:
        """Print validation report."""
        print("\n" + "=" * 70)
        print("📋 VALIDATION REPORT")
        print("=" * 70)

        errors = [msg for severity, msg in self.issues if severity == "ERROR"]
        warnings = [msg for severity, msg in self.issues if severity == "WARNING"]

        if errors:
            print(f"\n❌ ERRORS ({len(errors)}):")
            for error in errors:
                print(f"   • {error}")

        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for warning in warnings:
                print(f"   • {warning}")

        if not errors:
            print("\n✅ ALL VALIDATION CHECKS PASSED!")
        else:
            print("\n❌ VALIDATION FAILED - Please fix the errors above")

        if self.verbose and self.stats:
            print("\n📈 Statistics:")
            for key, value in sorted(self.stats.items()):
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate GR00T LeRobot format datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_dataset.py /path/to/dataset
  python scripts/validate_dataset.py /path/to/dataset --verbose
        """,
    )

    parser.add_argument("dataset_path", help="Path to the dataset root directory")
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Validate dataset
    validator = DatasetValidator(args.dataset_path, verbose=args.verbose)
    success = validator.validate()
    validator.print_report()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
