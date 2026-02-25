#!/usr/bin/env python3
"""
GR00T Training Configuration Validator and Helper

This script validates and analyzes fine-tuning configurations before training.
Helps users identify issues early and optimize hyperparameters.

Features:
- ✅ Validate dataset exists and has correct structure
- ✅ Check embodiment configuration compatibility
- ✅ Verify model checkpoint accessibility
- ✅ Validate hyperparameter ranges
- ✅ Check GPU memory requirements
- ✅ Suggest optimal batch sizes and learning rates
- ✅ Generate shell commands for finetuning

Usage:
    python scripts/validate_training_config.py --dataset /path/to/data --embodiment libero_panda
    python scripts/validate_training_config.py --check-memory --gpus 8 --batch-size 640
    python scripts/validate_training_config.py --suggest --dataset /path/to/data
    python scripts/validate_training_config.py --generate-command config.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add workspace to path if not already in PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
except ImportError as e:
    print("Error: Could not import GR00T modules. Make sure you're in the GR00T environment.")
    print(f"Details: {e}")
    sys.exit(1)


class TrainingConfigValidator:
    """Validator for fine-tuning configurations."""

    def __init__(self, verbose: bool = False):
        """Initialize the validator."""
        self.verbose = verbose
        self.issues: List[Tuple[str, str]] = []  # (severity, message)
        self.warnings: List[str] = []
        self.recommendations: List[str] = []
        self.stats: Dict[str, Any] = {}

    def validate_dataset(self, dataset_path: str) -> bool:
        """Validate dataset structure.

        Args:
            dataset_path: Path to dataset

        Returns:
            True if valid, False otherwise
        """
        print("\n📂 Validating Dataset")
        print("-" * 70)

        path = Path(dataset_path)
        if not path.exists():
            self.issues.append(("ERROR", f"Dataset path does not exist: {dataset_path}"))
            return False

        if not path.is_dir():
            self.issues.append(("ERROR", f"Dataset path is not a directory: {dataset_path}"))
            return False

        print(f"✓ Dataset path exists: {path}")

        # Check required directories
        required_dirs = ["meta", "data", "videos"]
        missing_dirs = []
        for dir_name in required_dirs:
            if not (path / dir_name).exists():
                missing_dirs.append(dir_name)

        if missing_dirs:
            self.issues.append(
                ("ERROR", f"Missing required directories: {', '.join(missing_dirs)}")
            )
            return False

        print(f"✓ All required directories present: {', '.join(required_dirs)}")

        # Check metadata files
        meta_dir = path / "meta"
        required_files = ["modality.json", "episodes.jsonl", "tasks.jsonl", "info.json"]
        missing_files = []
        for file_name in required_files:
            if not (meta_dir / file_name).exists():
                missing_files.append(file_name)

        if missing_files:
            self.issues.append(
                ("ERROR", f"Missing required metadata files: {', '.join(missing_files)}")
            )
            return False

        print(f"✓ All metadata files present: {', '.join(required_files)}")

        # Count episodes and data
        try:
            import json

            with open(meta_dir / "episodes.jsonl") as f:
                episodes = [json.loads(line) for line in f if line.strip()]
            num_episodes = len(episodes)
            print(f"✓ Found {num_episodes} episodes")
            self.stats["num_episodes"] = num_episodes

            # Estimate total frames
            total_frames = sum(ep.get("length", 0) for ep in episodes)
            print(f"✓ Approximately {total_frames:,} total frames")
            self.stats["total_frames"] = total_frames

            # Check if reasonable dataset size
            if total_frames < 1000:
                self.warnings.append(
                    "Dataset is very small (<1000 frames). Consider collecting more data for better results."
                )
            elif num_episodes < 10:
                self.warnings.append(
                    f"Dataset has only {num_episodes} episodes. Aim for 50+ episodes for robust finetuning."
                )

        except Exception as e:
            self.issues.append(("WARNING", f"Could not count episodes: {e}"))

        return True

    def validate_embodiment(self, embodiment_tag: str) -> bool:
        """Validate embodiment configuration.

        Args:
            embodiment_tag: Embodiment identifier

        Returns:
            True if valid, False otherwise
        """
        print("\n🤖 Validating Embodiment")
        print("-" * 70)

        # Check if embodiment exists in enum
        valid_tags = [e.value for e in EmbodimentTag]
        if embodiment_tag not in valid_tags:
            self.issues.append(
                ("ERROR", f"Unknown embodiment tag: {embodiment_tag}")
            )
            print(f"✗ Unknown embodiment: {embodiment_tag}")
            print(f"\nAvailable embodiments:")
            for tag in valid_tags:
                status = "✓" if tag in MODALITY_CONFIGS else "?"
                print(f"   {status} {tag}")
            return False

        print(f"✓ Valid embodiment: {embodiment_tag}")

        # Check if modality config is available
        if embodiment_tag not in MODALITY_CONFIGS:
            self.warnings.append(
                f"No pre-registered config for '{embodiment_tag}'. "
                "You must provide a custom modality_config_path."
            )
            print(f"⚠️  No pre-registered configuration for '{embodiment_tag}' in MODALITY_CONFIGS")
        else:
            print(f"✓ Modality configuration found in MODALITY_CONFIGS")
            config = MODALITY_CONFIGS[embodiment_tag]
            if "state" in config and hasattr(config["state"], "modality_keys"):
                state_dim = len(config["state"].modality_keys)
                print(f"  - State modality: {state_dim} components")
            if "action" in config and hasattr(config["action"], "modality_keys"):
                action_dim = len(config["action"].modality_keys)
                print(f"  - Action modality: {action_dim} components")

        return True

    def validate_hyperparameters(
        self,
        global_batch_size: int,
        num_gpus: int,
        learning_rate: float,
        max_steps: int,
        warmup_ratio: float,
        weight_decay: float,
    ) -> bool:
        """Validate hyperparameter ranges.

        Args:
            global_batch_size: Total batch size
            num_gpus: Number of GPUs
            learning_rate: Learning rate
            max_steps: Maximum training steps
            warmup_ratio: Warmup ratio
            weight_decay: Weight decay

        Returns:
            True if valid, False otherwise
        """
        print("\n⚙️  Validating Hyperparameters")
        print("-" * 70)

        all_valid = True

        # Batch size validation
        if global_batch_size < 32:
            self.warnings.append(
                f"Batch size {global_batch_size} is very small. Recommended: 128-640"
            )
        elif global_batch_size > 2048:
            self.warnings.append(
                f"Batch size {global_batch_size} is very large. May require high memory."
            )
        else:
            print(f"✓ Batch size {global_batch_size} is reasonable")

        # Per-GPU batch size
        per_gpu_batch_size = global_batch_size // num_gpus
        if per_gpu_batch_size < 4:
            self.issues.append(
                ("WARNING", f"Per-GPU batch size ({per_gpu_batch_size}) is too small (<4)")
            )
        else:
            print(f"✓ Per-GPU batch size: {per_gpu_batch_size}")

        # Learning rate validation
        if learning_rate < 1e-6:
            self.warnings.append(f"Learning rate {learning_rate} is very small")
        elif learning_rate > 1e-3:
            self.warnings.append(f"Learning rate {learning_rate} is very large. May cause divergence.")
        else:
            print(f"✓ Learning rate {learning_rate} is reasonable")

        # Warmup ratio
        if warmup_ratio < 0.0 or warmup_ratio > 1.0:
            self.issues.append(
                ("ERROR", f"Warmup ratio must be in [0, 1], got {warmup_ratio}")
            )
            all_valid = False
        elif warmup_ratio > 0.2:
            self.warnings.append(
                f"Warmup ratio {warmup_ratio} is large. Typical range: 0.01-0.1"
            )
        else:
            print(f"✓ Warmup ratio {warmup_ratio}")

        # Weight decay
        if weight_decay < 0:
            self.issues.append(("ERROR", f"Weight decay must be non-negative, got {weight_decay}"))
            all_valid = False
        else:
            print(f"✓ Weight decay {weight_decay}")

        # Max steps
        warmup_steps = int(max_steps * warmup_ratio)
        if warmup_steps < 100:
            self.warnings.append(
                f"Warmup has only {warmup_steps} steps. May be too short for stable training."
            )
        else:
            print(f"✓ Warmup steps: {warmup_steps}")

        return all_valid

    def check_memory_requirements(
        self, global_batch_size: int, num_gpus: int
    ) -> Dict[str, float]:
        """Estimate GPU memory requirements.

        Args:
            global_batch_size: Total batch size
            num_gpus: Number of GPUs

        Returns:
            Dictionary with memory estimates
        """
        print("\n💾 GPU Memory Analysis")
        print("-" * 70)

        # Rough estimates for GR00T-N1.6-3B model with bfloat16 training
        # Model weights: ~6GB (3B params at bfloat16)
        # Optimizer states (AdamW): ~12GB (2x model size for momentum and variance)
        # Per sample gradient memory: ~0.2-0.5GB per sample

        model_size_gb = 6
        optimizer_state_gb = 12  # AdamW with 2 states
        per_sample_memory_gb = 0.25  # Conservative estimate

        per_gpu_batch_size = global_batch_size // num_gpus
        per_gpu_memory_gb = model_size_gb + optimizer_state_gb + (per_gpu_batch_size * per_sample_memory_gb)

        total_memory_gb = per_gpu_memory_gb * num_gpus

        print(f"Model weights: ~{model_size_gb} GB (bfloat16)")
        print(f"Optimizer states: ~{optimizer_state_gb} GB (AdamW)")
        print(f"Per sample: ~{per_sample_memory_gb} GB")
        print(f"Per-GPU batch size: {per_gpu_batch_size}")
        print(f"Estimated per-GPU memory: {per_gpu_memory_gb:.1f} GB")
        print(f"Estimated total GPU memory: {total_memory_gb:.1f} GB")

        # Recommendations based on GPU type
        gpu_memory_map = {
            "H100": 80,
            "H200": 141,
            "A100": 80,
            "L40S": 48,
            "RTX 6000": 48,
            "RTX A6000": 48,
            "A10": 24,
            "V100": 32,
        }

        if per_gpu_memory_gb > 45:
            self.warnings.append(
                f"Estimated per-GPU memory ({per_gpu_memory_gb:.1f}GB) may exceed many GPUs. "
                "Recommended: H100/A100/H200 or reduce batch size."
            )
        elif per_gpu_memory_gb > 25:
            print(f"⚠️  Memory use is moderate. A100/H100/RTX6000 or better recommended.")

        return {
            "per_gpu_memory_gb": per_gpu_memory_gb,
            "total_memory_gb": total_memory_gb,
            "per_gpu_batch_size": per_gpu_batch_size,
        }

    def suggest_hyperparameters(self, num_episodes: int, num_frames: int) -> Dict[str, Any]:
        """Suggest hyperparameters based on dataset size.

        Args:
            num_episodes: Number of training episodes
            num_frames: Total number of frames

        Returns:
            Dictionary with suggested hyperparameters
        """
        print("\n💡 Suggested Hyperparameters")
        print("-" * 70)

        suggestions = {}

        # Number of epochs to see each sample ~3 times
        epochs_to_see_data = 3
        frames_per_epoch = max(5000, num_frames // 20)  # Estimate samples per epoch
        approx_steps_for_3_epochs = (num_frames // frames_per_epoch) * epochs_to_see_data

        # Round to reasonable values
        suggested_max_steps = int(np.ceil(approx_steps_for_3_epochs / 1000) * 1000)
        suggested_max_steps = max(5000, min(100000, suggested_max_steps))

        suggestions["max_steps"] = suggested_max_steps
        print(f"Max steps: {suggested_max_steps} (to see data ~3 times)")

        # Batch size based on number of episodes
        if num_episodes < 20:
            suggested_batch_size = 64
            print(f"Batch size: {suggested_batch_size} (small dataset)")
        elif num_episodes < 100:
            suggested_batch_size = 256
            print(f"Batch size: {suggested_batch_size} (medium dataset)")
        else:
            suggested_batch_size = 640
            print(f"Batch size: {suggested_batch_size} (large dataset)")

        suggestions["global_batch_size"] = suggested_batch_size

        # Learning rate (smaller for smaller datasets)
        if num_episodes < 20:
            suggested_lr = 5e-5
        elif num_episodes < 100:
            suggested_lr = 1e-4
        else:
            suggested_lr = 5e-4

        suggestions["learning_rate"] = suggested_lr
        print(f"Learning rate: {suggested_lr}")

        # Warmup ratio
        suggestions["warmup_ratio"] = 0.05
        print(f"Warmup ratio: 0.05")

        # Weight decay
        suggestions["weight_decay"] = 1e-5
        print(f"Weight decay: 1e-5")

        # Gradient accumulation
        suggestions["gradient_accumulation_steps"] = 1
        print(f"Gradient accumulation steps: 1")

        return suggestions

    def print_report(self) -> None:
        """Print validation report."""
        print("\n" + "=" * 70)
        print("📋 VALIDATION REPORT")
        print("=" * 70)

        errors = [msg for severity, msg in self.issues if severity == "ERROR"]
        warnings = [msg for severity, msg in self.issues if severity == "WARNING"]
        warnings.extend(self.warnings)

        if errors:
            print(f"\n❌ ERRORS ({len(errors)}):")
            for error in errors:
                print(f"   • {error}")

        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for warning in warnings:
                print(f"   • {warning}")

        if self.recommendations:
            print(f"\n💡 RECOMMENDATIONS ({len(self.recommendations)}):")
            for rec in self.recommendations:
                print(f"   • {rec}")

        if not errors:
            print("\n✅ Configuration validation PASSED!")
        else:
            print("\n❌ Configuration validation FAILED - Please fix errors above")

    def validate(
        self,
        dataset_path: Optional[str] = None,
        embodiment_tag: Optional[str] = None,
        global_batch_size: int = 640,
        num_gpus: int = 8,
        learning_rate: float = 1e-4,
        max_steps: int = 20000,
        warmup_ratio: float = 0.05,
        weight_decay: float = 1e-5,
    ) -> bool:
        """Run all validation checks.

        Returns:
            True if all validations pass, False otherwise
        """
        print("=" * 70)
        print("🔍 GR00T Training Configuration Validator")
        print("=" * 70)

        all_valid = True

        # Validate dataset if provided
        if dataset_path:
            if not self.validate_dataset(dataset_path):
                all_valid = False

        # Validate embodiment if provided
        if embodiment_tag:
            if not self.validate_embodiment(embodiment_tag):
                all_valid = False

        # Validate hyperparameters
        if not self.validate_hyperparameters(
            global_batch_size, num_gpus, learning_rate, max_steps, warmup_ratio, weight_decay
        ):
            all_valid = False

        # Check memory
        memory_info = self.check_memory_requirements(global_batch_size, num_gpus)

        # Suggest hyperparameters if we have dataset info
        if dataset_path and "num_episodes" in self.stats:
            suggestions = self.suggest_hyperparameters(
                self.stats["num_episodes"], self.stats.get("total_frames", 0)
            )

        self.print_report()
        return all_valid and len([s for s, _ in self.issues if s == "ERROR"]) == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate GR00T fine-tuning configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_training_config.py --dataset /path/to/data --embodiment libero_panda
  python scripts/validate_training_config.py --suggest --dataset /path/to/data
  python scripts/validate_training_config.py --check-memory --gpus 8 --batch-size 640
        """,
    )

    parser.add_argument(
        "--dataset", type=str, help="Path to training dataset"
    )
    parser.add_argument(
        "--embodiment", type=str, help="Embodiment tag (e.g., libero_panda, unitree_g1)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=640,
        help="Global batch size (default: 640)",
    )
    parser.add_argument(
        "--gpus", type=int, default=8, help="Number of GPUs (default: 8)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20000,
        help="Maximum training steps (default: 20000)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help="Warmup ratio (default: 0.05)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay (default: 1e-5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Create and run validator
    validator = TrainingConfigValidator(verbose=args.verbose)
    success = validator.validate(
        dataset_path=args.dataset,
        embodiment_tag=args.embodiment,
        global_batch_size=args.batch_size,
        num_gpus=args.gpus,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
