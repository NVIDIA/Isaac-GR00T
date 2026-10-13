# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# The CLI intentionally bootstraps imports and prints its read-only report.
# ruff: noqa: E402, T201

"""Read-only preflight checks for a GR00T AutoResearch study."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_OSS_ROOT = Path(__file__).resolve().parents[3]
if str(_OSS_ROOT) not in sys.path:
    sys.path.insert(0, str(_OSS_ROOT))

from skills.autoresearch.cli import _controller, _validate, configure_logging
from skills.autoresearch.config import StudyConfig


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to autoresearch.yaml")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate one config and print its normalized execution contract."""
    args = _parser().parse_args(argv)
    configure_logging()
    try:
        config = StudyConfig.from_yaml(args.config)
        _validate(config)
        controller = _controller(config)
        controller.check_baseline_train_config()
        state_path = config.output_dir / "state.json"
        state = controller.load_state() if state_path.is_file() else None
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        print(f"[autoresearch-check] FAILED: {exc}", file=sys.stderr)
        return 1

    mode = "resume" if state is not None else "new"
    print("[autoresearch-check] PASSED")
    print(f"config={args.config.expanduser().resolve()}")
    print(f"study_id={config.study_id}")
    print(f"mode={mode}")
    print(f"workspace={config.workspace}")
    print(f"output_dir={config.output_dir}")
    print(f"base_branch={config.base_branch}")
    print(f"base_commit={config.base_commit}")
    print(
        f"metric={config.objective.result_path} "
        f"direction={config.objective.direction} "
        f"baseline={config.objective.baseline}"
    )
    print(
        f"budget=max_trials:{config.max_trials} "
        f"max_parallel_trials:{config.max_parallel_trials} "
        f"max_retry:{config.max_retry}"
    )
    print(f"train_script={config.train_script}")
    print(f"eval_script={config.eval_script}")
    lock = config.train_config_lock
    if lock is not None:
        print(f"locked_train_config={lock.path} fields={','.join(lock.fields)}")
    if state is not None:
        active_wave = state.active_wave_index if state.active_wave_index is not None else "none"
        best_trial = state.best_trial_id or "baseline"
        print(
            f"state=wave:{active_wave} trials:{len(state.trials)} "
            f"best_trial:{best_trial} best_score:{state.best_score}"
        )
    print("No train or eval script was executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
