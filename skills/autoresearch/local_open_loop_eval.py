# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""YAML-configured open-loop evaluation for AutoResearch."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path

import tyro
import yaml

from groot.infra.eval.oss_open_loop_eval_config import OssOpenLoopEvalConfig


def _load_config(path: Path) -> OssOpenLoopEvalConfig:
    """Load evaluator arguments and environment variables from YAML."""
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(values, dict):
        raise TypeError(f"Evaluation config must be a mapping: {path}")

    environment = values.pop("environment", {})
    if not isinstance(environment, dict):
        raise TypeError(f"Evaluation environment must be a mapping: {path}")
    for name, value in environment.items():
        os.environ[str(name)] = str(value)
    return OssOpenLoopEvalConfig(**values)


def _parse_config(args: Sequence[str] | None = None) -> OssOpenLoopEvalConfig:
    """Load YAML defaults, then apply command-line overrides."""
    loader = argparse.ArgumentParser(add_help=False)
    loader.add_argument("--load-config-path", type=Path)
    loader_args, remaining_args = loader.parse_known_args(args)

    config = OssOpenLoopEvalConfig()
    if loader_args.load_config_path is not None:
        config = _load_config(loader_args.load_config_path)
    return tyro.cli(OssOpenLoopEvalConfig, default=config, args=remaining_args)


def main(args: Sequence[str] | None = None) -> None:
    """Run open-loop evaluation."""
    config = _parse_config(args)
    # Import after loading YAML so environment settings take effect first.
    from groot.infra.eval.oss_open_loop_evaluator import OssOpenLoopEvaluator

    OssOpenLoopEvaluator(config).run()


if __name__ == "__main__":
    main()
