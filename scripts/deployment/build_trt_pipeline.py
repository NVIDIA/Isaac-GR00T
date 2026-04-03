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
Unified TensorRT pipeline: export ONNX, build engines, verify accuracy, and benchmark.

Wraps the 4 deployment steps into a single script with clean progress output.
Verbose logs from each step are written to a log file; the terminal shows only
progress headers, one-line results, and a final summary.

Usage:
    # Full pipeline (recommended)
    python scripts/deployment/build_trt_pipeline.py \
        --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
        --dataset-path demo_data/libero_demo

    # Export + build only
    python scripts/deployment/build_trt_pipeline.py \
        --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
        --dataset-path demo_data/libero_demo \
        --steps export,build

    # Skip benchmark
    python scripts/deployment/build_trt_pipeline.py \
        --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
        --dataset-path demo_data/libero_demo \
        --steps export,build,verify
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from enum import Enum
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback
from typing import IO, Literal, Optional

import tyro


# Ensure scripts/deployment/ is on sys.path for sibling module imports.
_DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
if _DEPLOY_DIR not in sys.path:
    sys.path.insert(0, _DEPLOY_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Step(str, Enum):
    """Pipeline steps. Values are the user-facing CLI tokens for --steps."""

    EXPORT = "export"
    BUILD = "build"
    VERIFY = "verify"
    BENCHMARK = "benchmark"


VALID_STEPS = tuple(Step)

# Mapping from export_mode -> (build mode, verify mode, benchmark trt_mode)
_MODE_MAP = {
    "full_pipeline": ("full_pipeline", "n17_full_pipeline", "n17_full_pipeline"),
    "action_head": ("full_pipeline", "action_head", "dit_only"),
    "dit_only": ("single", "action_head", "dit_only"),
}


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def _count_files(directory: str, suffix: str) -> int:
    if not os.path.isdir(directory):
        return 0
    return sum(1 for f in os.listdir(directory) if f.endswith(suffix))


def _print_header(step_num: int, total: int, msg: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"[Step {step_num}/{total}] {msg}")
    print(f"{'=' * 60}")


def _print_result(step_num: int, total: int, msg: str, elapsed: float) -> None:
    print(f"[Step {step_num}/{total}] {msg} ({_fmt_elapsed(elapsed)})")


class _TeeWriter:
    """Write to multiple streams simultaneously."""

    def __init__(self, *streams: IO[str]):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


@contextlib.contextmanager
def _redirect_to_log(log_file: IO[str], tee: bool = False):
    """Redirect stdout, stderr, and all logging to the log file.

    The caller's own progress prints should happen *outside* this context.
    If *tee* is True, output goes to both the log file and the real terminal.
    """
    old_stdout, old_stderr = sys.stdout, sys.stderr
    # Redirect streams
    if tee:
        tee_out = _TeeWriter(log_file, old_stdout)
        tee_err = _TeeWriter(log_file, old_stderr)
        sys.stdout = tee_out
        sys.stderr = tee_err
        log_stream = tee_out
    else:
        sys.stdout = log_file
        sys.stderr = log_file
        log_stream = log_file

    # Redirect all logging handlers to the log file
    root_logger = logging.getLogger()
    old_handlers = root_logger.handlers[:]
    root_logger.handlers.clear()
    file_handler = logging.StreamHandler(log_stream)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(file_handler)

    try:
        yield
    finally:
        log_file.flush()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        root_logger.handlers.clear()
        for h in old_handlers:
            root_logger.addHandler(h)


def _resolve_embodiment(model_path: str, embodiment_tag: Optional[str]):
    """Auto-detect embodiment tag from processor_config.json if not provided."""
    from gr00t.data.embodiment_tags import EmbodimentTag

    if embodiment_tag is not None:
        return EmbodimentTag.resolve(embodiment_tag)

    config_file = Path(model_path) / "processor_config.json"
    if not config_file.exists():
        raise ValueError(
            f"Cannot auto-detect embodiment_tag: {config_file} not found. "
            "Please provide --embodiment-tag explicitly."
        )
    with open(config_file, "r") as f:
        processor_config = json.load(f)
    modality_configs = processor_config.get("processor_kwargs", {}).get("modality_configs", {})
    if len(modality_configs) == 0:
        raise ValueError(
            "Cannot auto-detect embodiment_tag: no modality_configs found in processor_config.json. "
            "Please provide --embodiment-tag explicitly."
        )
    if len(modality_configs) == 1:
        embodiment_key = next(iter(modality_configs))
        tag = EmbodimentTag.resolve(embodiment_key)
        print(f"  Auto-detected embodiment: {tag} (from {embodiment_key})")
        return tag
    available = sorted(modality_configs.keys())
    raise ValueError(
        f"Multiple embodiments found in processor_config.json: {available}. "
        "Please provide --embodiment-tag explicitly."
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Unified TensorRT deployment pipeline configuration."""

    # -- Shared (required) --------------------------------------------------
    model_path: str = ""
    """Path to the model checkpoint (required)."""

    dataset_path: str = "demo_data/libero_demo"
    """Path to the dataset (LeRobot format)."""

    embodiment_tag: Optional[str] = None
    """Embodiment tag. Auto-detected from processor_config.json if not provided."""

    output_dir: str = "./gr00t_trt_deployment"
    """Root output directory. ONNX files go to <output_dir>/onnx/, engines to <output_dir>/engines/."""

    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    """Precision for ONNX export and TRT engine build."""

    batch_size: int = 1
    """Batch size baked into the exported ONNX/TRT models (default: 1)."""

    # -- Export options ------------------------------------------------------
    export_mode: Literal["dit_only", "action_head", "full_pipeline"] = "full_pipeline"
    """Export mode: 'dit_only', 'action_head', or 'full_pipeline' (recommended)."""

    video_backend: Literal["decord", "torchvision_av", "torchcodec"] = "torchcodec"
    """Video backend for dataset loading."""

    # -- Build options ------------------------------------------------------
    workspace: int = 8192
    """TRT builder workspace size in MB."""

    # -- Benchmark options --------------------------------------------------
    num_iterations: int = 20
    """Number of benchmark iterations."""

    warmup: int = 5
    """Number of warmup iterations."""

    skip_compile: bool = False
    """Skip torch.compile benchmark (slow JIT compilation)."""

    # -- Pipeline control ---------------------------------------------------
    steps: str = "all"
    """Steps to run: 'all' or comma-separated subset of 'export,build,verify,benchmark'."""

    log_file: Optional[str] = None
    """Log file path. Defaults to <output_dir>/pipeline.log."""


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------


def _run_export(cfg: PipelineConfig, onnx_dir: str, embodiment_tag, log_fp) -> None:
    from export_onnx_n1d7 import ExportConfig, main as export_main

    export_cfg = ExportConfig(
        model_path=cfg.model_path,
        dataset_path=cfg.dataset_path,
        embodiment_tag=embodiment_tag,
        output_dir=onnx_dir,
        export_mode=cfg.export_mode,
        video_backend=cfg.video_backend,
        precision=cfg.precision,
        batch_size=cfg.batch_size,
    )
    with _redirect_to_log(log_fp):
        export_main(export_cfg)


def _run_build(cfg: PipelineConfig, onnx_dir: str, engine_dir: str, log_fp) -> None:
    from build_tensorrt_engine import BuildConfig, main as build_main

    build_mode, _, _ = _MODE_MAP[cfg.export_mode]

    if build_mode == "single":
        # dit_only: single ONNX -> single engine
        build_cfg = BuildConfig(
            mode="single",
            onnx=os.path.join(onnx_dir, "dit_bf16.onnx"),
            engine=os.path.join(engine_dir, "dit_bf16.engine"),
            precision=cfg.precision,
            workspace=cfg.workspace,
        )
    else:
        build_cfg = BuildConfig(
            mode="full_pipeline",
            onnx_dir=onnx_dir,
            engine_dir=engine_dir,
            precision=cfg.precision,
            workspace=cfg.workspace,
        )
    with _redirect_to_log(log_fp):
        build_main(build_cfg)


def _run_verify(cfg: PipelineConfig, engine_dir: str, embodiment_tag, log_fp) -> float:
    from verify_n1d7_trt import VerifyConfig, main as verify_main

    _, verify_mode, _ = _MODE_MAP[cfg.export_mode]
    verify_cfg = VerifyConfig(
        model_path=cfg.model_path,
        dataset_path=cfg.dataset_path,
        engine_dir=engine_dir,
        mode=verify_mode,
        embodiment_tag=embodiment_tag,
        batch_size=cfg.batch_size,
    )
    with _redirect_to_log(log_fp, tee=True):
        cosine = verify_main(verify_cfg)
    return cosine


def _run_benchmark(cfg: PipelineConfig, engine_dir: str, embodiment_tag, log_fp) -> None:
    from benchmark_inference import BenchmarkConfig, main as benchmark_main

    _, _, trt_mode = _MODE_MAP[cfg.export_mode]

    # For dit_only, engine path is the single .engine file
    if cfg.export_mode == "dit_only":
        trt_engine_path = os.path.join(engine_dir, "dit_bf16.engine")
    else:
        trt_engine_path = engine_dir

    benchmark_cfg = BenchmarkConfig(
        model_path=cfg.model_path,
        dataset_path=cfg.dataset_path,
        embodiment_tag=embodiment_tag.value,
        trt_engine_path=trt_engine_path,
        trt_mode=trt_mode,
        num_iterations=cfg.num_iterations,
        warmup=cfg.warmup,
        skip_compile=cfg.skip_compile,
        batch_size=cfg.batch_size,
    )
    with _redirect_to_log(log_fp, tee=True):
        benchmark_main(benchmark_cfg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: PipelineConfig | None = None) -> None:
    if cfg is None:
        cfg = tyro.cli(PipelineConfig)

    if not cfg.model_path:
        raise ValueError("Please provide --model-path")

    # Parse steps
    valid_names = {s.value for s in Step}
    if cfg.steps == "all":
        steps = list(Step)
    else:
        steps = []
        for token in cfg.steps.split(","):
            token = token.strip()
            if token not in valid_names:
                raise ValueError(f"Unknown step '{token}'. Valid steps: {', '.join(valid_names)}")
            steps.append(Step(token))

    # Validate step dependencies
    if Step.BUILD in steps and Step.EXPORT not in steps:
        onnx_dir = os.path.join(cfg.output_dir, "onnx")
        if not os.path.isdir(onnx_dir) or _count_files(onnx_dir, ".onnx") == 0:
            raise ValueError(
                "Step 'build' requires ONNX models. Either include 'export' in --steps "
                f"or ensure ONNX files exist in {onnx_dir}"
            )
    if Step.VERIFY in steps and Step.BUILD not in steps:
        engine_dir = os.path.join(cfg.output_dir, "engines")
        if not os.path.isdir(engine_dir) or _count_files(engine_dir, ".engine") == 0:
            raise ValueError(
                "Step 'verify' requires TRT engines. Either include 'build' in --steps "
                f"or ensure engine files exist in {engine_dir}"
            )
    if Step.BENCHMARK in steps and Step.BUILD not in steps:
        engine_dir = os.path.join(cfg.output_dir, "engines")
        if not os.path.isdir(engine_dir) or _count_files(engine_dir, ".engine") == 0:
            raise ValueError(
                "Step 'benchmark' requires TRT engines. Either include 'build' in --steps "
                f"or ensure engine files exist in {engine_dir}"
            )

    # Derived paths
    onnx_dir = os.path.join(cfg.output_dir, "onnx")
    engine_dir = os.path.join(cfg.output_dir, "engines")
    os.makedirs(cfg.output_dir, exist_ok=True)

    log_path = cfg.log_file or os.path.join(cfg.output_dir, "pipeline.log")
    log_fp = open(log_path, "w")

    try:
        # Resolve embodiment tag once
        embodiment_tag = _resolve_embodiment(cfg.model_path, cfg.embodiment_tag)

        total = len(steps)
        results: dict[str, str] = {}
        cosine_val: float | None = None

        print("=" * 60)
        print("GR00T TensorRT Deployment Pipeline")
        print("=" * 60)
        print(f"  Model:        {cfg.model_path}")
        print(f"  Dataset:      {cfg.dataset_path}")
        print(f"  Embodiment:   {embodiment_tag}")
        print(f"  Export mode:  {cfg.export_mode}")
        print(f"  Batch size:   {cfg.batch_size}")
        print(f"  Precision:    {cfg.precision}")
        print(f"  Output:       {cfg.output_dir}")
        print(f"  Steps:        {', '.join(s.value for s in steps)}")
        print(f"  Log file:     {log_path}")

        for i, step in enumerate(steps, 1):
            t0 = time.time()  # fallback for error handler
            try:
                if step is Step.EXPORT:
                    _print_header(i, total, "Exporting ONNX models...")
                    t0 = time.time()
                    _run_export(cfg, onnx_dir, embodiment_tag, log_fp)
                    elapsed = time.time() - t0
                    n = _count_files(onnx_dir, ".onnx")
                    _print_result(
                        i, total, f"Export complete -- {n} ONNX files in {onnx_dir}", elapsed
                    )
                    results[step.value] = f"{n} ONNX files ({_fmt_elapsed(elapsed)})"

                elif step is Step.BUILD:
                    _print_header(i, total, "Building TensorRT engines...")
                    t0 = time.time()
                    _run_build(cfg, onnx_dir, engine_dir, log_fp)
                    elapsed = time.time() - t0
                    n = _count_files(engine_dir, ".engine")
                    _print_result(
                        i, total, f"Build complete -- {n} engines in {engine_dir}", elapsed
                    )
                    results[step.value] = f"{n} engines ({_fmt_elapsed(elapsed)})"

                elif step is Step.VERIFY:
                    _print_header(i, total, "Verifying TRT accuracy...")
                    t0 = time.time()
                    cosine_val = _run_verify(cfg, engine_dir, embodiment_tag, log_fp)
                    elapsed = time.time() - t0
                    status = "PASS" if cosine_val and cosine_val > 0.99 else "FAIL"
                    cos_str = f"{cosine_val:.6f}" if cosine_val is not None else "N/A"
                    _print_result(
                        i, total, f"Verify complete -- cosine={cos_str} {status}", elapsed
                    )
                    results[step.value] = f"cosine={cos_str} {status} ({_fmt_elapsed(elapsed)})"

                elif step is Step.BENCHMARK:
                    _print_header(i, total, "Running benchmark...")
                    t0 = time.time()
                    _run_benchmark(cfg, engine_dir, embodiment_tag, log_fp)
                    elapsed = time.time() - t0
                    _print_result(i, total, "Benchmark complete", elapsed)
                    results[step.value] = f"done ({_fmt_elapsed(elapsed)})"

            except Exception as e:
                elapsed = time.time() - t0
                print(f"\n[Step {i}/{total}] FAILED: {step.value} ({_fmt_elapsed(elapsed)})")
                print(f"  Error: {e}")
                print(f"  See full log: {log_path}")
                # Write traceback to log
                log_fp.write(f"\n{'=' * 60}\nSTEP FAILED: {step.value}\n{'=' * 60}\n")
                log_fp.write(traceback.format_exc())
                sys.exit(1)

        # Final summary
        print(f"\n{'=' * 60}")
        print("TRT Pipeline Complete!")
        print(f"{'=' * 60}")
        for step_name, result in results.items():
            print(f"  {step_name:12s}  {result}")
        print(f"  {'log':12s}  {log_path}")
        print(f"{'=' * 60}")
    finally:
        log_fp.close()


if __name__ == "__main__":
    main()
