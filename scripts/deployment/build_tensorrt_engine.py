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
Build TensorRT engines from exported ONNX models.

Supports two modes:
- single: Build engine for a single ONNX model
- full_pipeline: Build engines for all pipeline components
  (ViT, LLM, State Encoder, Action Encoder, DiT, Action Decoder)

Shape profiles are automatically derived from the ONNX models.

Usage:
    # Full pipeline:
    python scripts/deployment/build_tensorrt_engine.py \
        --mode full_pipeline \
        --onnx-dir ./gr00t_n1d7_onnx \
        --engine-dir ./gr00t_n1d7_engines \
        --precision bf16
"""

from dataclasses import dataclass
import json
import logging
import os
import time
from typing import Literal

from gr00t.deployment.modes import BuildEngineMode
import onnx
import tensorrt as trt
import tyro


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# STRONGLY_TYPED precision sanity check: TRT 10+ STRONGLY_TYPED reads
# precision from the ONNX tensor types and ignores --precision builder
# flags. Catch the silent mismatch (user asks fp16, ONNX is bf16, engine
# silently builds bf16) before burning build time. Indirected through
# dtype *names* so the helper can be unit-tested without TensorRT.
_PRECISION_TO_TRT_DTYPE_NAME: dict[str, str] = {
    "bf16": "BF16",
    "fp16": "HALF",
    "fp32": "FLOAT",
    "fp8": "FP8",
}


def _check_strongly_typed_precision_match(
    network_dtype_names: set[str], requested_precision: str
) -> None:
    """Raise if --precision cannot be honored by this STRONGLY_TYPED network."""
    expected = _PRECISION_TO_TRT_DTYPE_NAME.get(requested_precision)
    if expected is None:
        raise ValueError(
            f"Unknown precision: {requested_precision!r}. "
            f"Expected one of {sorted(_PRECISION_TO_TRT_DTYPE_NAME)}."
        )
    if expected not in network_dtype_names:
        raise ValueError(
            f"--precision={requested_precision} cannot be honored by this ONNX. "
            f"STRONGLY_TYPED (TRT 10+) reads precision from ONNX tensor types "
            f"and ignores builder flags. Network has tensor dtypes "
            f"{sorted(network_dtype_names)}; none of them are {expected}. "
            f"Either re-export the ONNX with the requested precision, or "
            f"pass --precision matching the existing ONNX dtypes."
        )
    # When fp32 is requested, the network must not contain any reduced-precision
    # tensors. STRONGLY_TYPED won't promote BF16/FP16/FP8 to FLOAT, so a mixed
    # BF16+FLOAT network silently runs at BF16 for those tensors despite the
    # caller asking for fp32.
    if requested_precision == "fp32":
        reduced = {"BF16", "HALF", "FP8"} & network_dtype_names
        if reduced:
            raise ValueError(
                f"--precision=fp32 cannot be honored: network also contains "
                f"reduced-precision tensors {sorted(reduced)}. STRONGLY_TYPED "
                f"won't promote them to FLOAT, so the engine would silently "
                f"run mixed precision. Re-export the ONNX as pure FP32, or "
                f"pass --precision matching the dominant reduced dtype."
            )


def _precision_from_onnx_path(onnx_path: str, default: str) -> str:
    """Return the precision tag suffixed in the ONNX filename (e.g.
    ``vit_fp32.onnx`` → ``"fp32"``), else ``default``. Used so the
    full-pipeline build mirrors the export's per-component dtype instead
    of forwarding the pipeline-wide ``--precision`` to a mismatched ONNX.
    """
    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    for tag in _PRECISION_TO_TRT_DTYPE_NAME:
        if stem.endswith(f"_{tag}"):
            return tag
    return default


# ============================================================
# Auto Shape Profile from ONNX
# ============================================================


def derive_shapes_from_onnx(onnx_path, max_batch=8):
    """Read an ONNX model and derive min/opt/max shape profiles.

    For each input:
    - Fixed dimensions (concrete values) are kept as-is across min/opt/max.
    - Dynamic batch dimension: min=1, opt=1, max=max_batch.
    - Dynamic sequence dimensions: min=1, opt=concrete_value, max=2*concrete_value.
      (concrete_value comes from the ONNX model's shape hints)

    Returns (min_shapes, opt_shapes, max_shapes) dicts.
    """
    model = onnx.load(onnx_path, load_external_data=False)

    min_shapes, opt_shapes, max_shapes = {}, {}, {}

    for inp in model.graph.input:
        name = inp.name
        dims = inp.type.tensor_type.shape.dim

        min_shape, opt_shape, max_shape = [], [], []
        for i, d in enumerate(dims):
            if d.dim_value > 0:
                # Fixed dimension — use as-is
                min_shape.append(d.dim_value)
                opt_shape.append(d.dim_value)
                max_shape.append(d.dim_value)
            else:
                # Dynamic dimension
                if i == 0:
                    # Batch dimension
                    min_shape.append(1)
                    opt_shape.append(1)
                    max_shape.append(max_batch)
                else:
                    # Sequence/spatial dimension — use generous range
                    # We don't know the "typical" value from ONNX alone,
                    # so use 1 / 1 / large_max. The builder will optimize for opt.
                    min_shape.append(1)
                    opt_shape.append(1)
                    max_shape.append(512)

        min_shapes[name] = tuple(min_shape)
        opt_shapes[name] = tuple(opt_shape)
        max_shapes[name] = tuple(max_shape)

    return min_shapes, opt_shapes, max_shapes


def derive_shapes_with_hint(onnx_path, opt_seq_lens=None, max_batch=8):
    """Derive shapes from ONNX, with optional sequence length hints.

    Args:
        onnx_path: Path to ONNX model
        opt_seq_lens: Dict mapping dynamic dim names to optimal sequence lengths.
                      e.g. {"sa_seq_len": 51, "vl_seq_len": 280, "sequence_length": 280}
        max_batch: Maximum batch size
    """
    model = onnx.load(onnx_path, load_external_data=False)
    opt_seq_lens = opt_seq_lens or {}

    min_shapes, opt_shapes, max_shapes = {}, {}, {}

    for inp in model.graph.input:
        name = inp.name
        dims = inp.type.tensor_type.shape.dim

        min_shape, opt_shape, max_shape = [], [], []
        for i, d in enumerate(dims):
            if d.dim_value > 0:
                # Fixed dimension
                min_shape.append(d.dim_value)
                opt_shape.append(d.dim_value)
                max_shape.append(d.dim_value)
            else:
                dim_name = d.dim_param if d.dim_param else f"dim_{i}"
                if dim_name == "batch_size":
                    # Batch dimension (at any index)
                    min_shape.append(1)
                    opt_shape.append(1)
                    max_shape.append(max_batch)
                elif dim_name in opt_seq_lens:
                    # Named dynamic dim with a hint
                    opt_val = opt_seq_lens[dim_name]
                    min_shape.append(1)
                    opt_shape.append(opt_val)
                    max_shape.append(max(opt_val * 2, opt_val + 64))
                else:
                    # Unknown dynamic dim — use wide range
                    min_shape.append(1)
                    opt_shape.append(256)
                    max_shape.append(512)

        min_shapes[name] = tuple(min_shape)
        opt_shapes[name] = tuple(opt_shape)
        max_shapes[name] = tuple(max_shape)

    return min_shapes, opt_shapes, max_shapes


# ============================================================
# Engine Builder
# ============================================================


def build_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = "bf16",
    workspace_mb: int = 8192,
    min_shapes: dict = None,
    opt_shapes: dict = None,
    max_shapes: dict = None,
    trt_severity=None,
):
    """Build TensorRT engine from ONNX model.

    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        precision: Precision mode ('fp32', 'fp16', 'bf16', 'fp8')
        workspace_mb: Workspace size in MB
        min_shapes: Minimum input shapes (dict: name -> shape tuple)
        opt_shapes: Optimal input shapes (dict: name -> shape tuple)
        max_shapes: Maximum input shapes (dict: name -> shape tuple)
    """
    logger.info("=" * 80)
    logger.info("TensorRT Engine Builder")
    logger.info("=" * 80)
    logger.info(f"ONNX model: {onnx_path}")
    logger.info(f"Engine output: {engine_path}")
    logger.info(f"Precision: {precision.upper()}")
    logger.info(f"Workspace: {workspace_mb} MB")
    logger.info("=" * 80)

    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE if trt_severity is None else trt_severity)

    # Create builder and network
    logger.info("\n[Step 1/5] Creating TensorRT builder...")
    builder = trt.Builder(TRT_LOGGER)

    # TRT 10.x prefers STRONGLY_TYPED; EXPLICIT_BATCH is the 9.x fallback.
    use_strongly_typed = hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED")
    if use_strongly_typed:
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        logger.info("Using STRONGLY_TYPED network (TRT 10.x+)")
    else:
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        logger.info("Using EXPLICIT_BATCH network (TRT 9.x fallback)")
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX model
    logger.info("\n[Step 2/5] Parsing ONNX model...")
    if not parser.parse_from_file(onnx_path):
        logger.error("Failed to parse ONNX file")
        for error in range(parser.num_errors):
            logger.error(parser.get_error(error))
        raise RuntimeError("ONNX parsing failed")

    logger.info(f"Network inputs: {network.num_inputs}")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        logger.info(f"  Input {i}: {inp.name} {inp.shape}")

    logger.info(f"Network outputs: {network.num_outputs}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        logger.info(f"  Output {i}: {out.name} {out.shape}")

    # Create builder config
    logger.info("\n[Step 3/5] Configuring builder...")
    config = builder.create_builder_config()

    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    logger.info("Enabled DETAILED profiling verbosity for engine inspection")

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1024**2))

    if use_strongly_typed:
        network_dtype_names: set[str] = set()
        for i in range(network.num_inputs):
            network_dtype_names.add(network.get_input(i).dtype.name)
        for i in range(network.num_outputs):
            network_dtype_names.add(network.get_output(i).dtype.name)
        _check_strongly_typed_precision_match(network_dtype_names, precision)
        logger.info(
            f"Precision '{precision}' matches ONNX tensor dtypes (STRONGLY_TYPED, "
            f"network has {sorted(network_dtype_names)})"
        )
    else:
        # Weak-typed fallback: explicitly set precision flags
        if precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("Enabled FP16 mode")
        elif precision == "bf16":
            config.set_flag(trt.BuilderFlag.BF16)
            logger.info("Enabled BF16 mode")
        elif precision == "fp8":
            config.set_flag(trt.BuilderFlag.FP8)
            config.set_flag(trt.BuilderFlag.BF16)
            logger.info("Enabled FP8 + BF16 mode")
        elif precision == "fp32":
            logger.info("Using FP32 (default precision)")
        else:
            raise ValueError(f"Unknown precision: {precision}")

    # Set optimization profiles for dynamic shapes
    if min_shapes and opt_shapes and max_shapes:
        logger.info("\n[Step 4/5] Setting optimization profiles...")
        profile = builder.create_optimization_profile()

        for i in range(network.num_inputs):
            inp = network.get_input(i)
            input_name = inp.name

            if input_name in min_shapes:
                min_shape = min_shapes[input_name]
                opt_shape = opt_shapes[input_name]
                max_shape = max_shapes[input_name]

                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                logger.info(f"  {input_name}:")
                logger.info(f"    min: {min_shape}")
                logger.info(f"    opt: {opt_shape}")
                logger.info(f"    max: {max_shape}")

        config.add_optimization_profile(profile)
    else:
        raise RuntimeError("Provide min/max and opt shapes for dynamic axes")

    # Build engine
    logger.info("\n[Step 5/5] Building TensorRT engine...")

    start_time = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    build_time = time.time() - start_time

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    logger.info(f"Engine built in {build_time:.1f} seconds ({build_time / 60:.1f} minutes)")

    # Save engine
    logger.info(f"\nSaving engine to {engine_path}...")
    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    engine_size_mb = os.path.getsize(engine_path) / (1024**2)
    logger.info(f"Engine saved! Size: {engine_size_mb:.2f} MB")

    logger.info("\n" + "=" * 80)
    logger.info("ENGINE BUILD COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Engine file: {engine_path}")
    logger.info(f"Size: {engine_size_mb:.2f} MB")
    logger.info(f"Build time: {build_time:.1f}s")
    logger.info(f"Precision: {precision.upper()}")
    logger.info("=" * 80)

    return engine_path


# ============================================================
# Full Pipeline Builder
# ============================================================


def build_full_pipeline(
    onnx_dir, engine_dir, precision="bf16", workspace_mb=8192, trt_severity=None
):
    """Build all TRT engines for the full pipeline.

    Shape profiles are automatically derived from the ONNX models.
    Dynamic sequence dimensions use hints based on typical inference shapes.

    Args:
        onnx_dir: Directory containing exported ONNX models
        engine_dir: Directory to save TRT engines
        precision: Precision mode
        workspace_mb: Workspace size in MB
    """
    os.makedirs(engine_dir, exist_ok=True)

    # Load sequence length hints from export metadata if available,
    # otherwise fall back to hardcoded defaults for GR1 single-view.
    metadata_path = os.path.join(onnx_dir, "export_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        seq_hints = {
            "sa_seq_len": metadata["sa_seq_len"],
            "vl_seq_len": metadata["vl_seq_len"],
            "sequence_length": metadata["llm_seq_len"],
            "seq_len": metadata["llm_seq_len"],  # N1.7 LLM dynamic dim name
            "num_patches": metadata.get("num_patches", 256),
            "num_merged_patches": metadata.get("num_merged_patches", 64),
            "num_vis_tokens": metadata.get("num_vis_tokens", 64),  # N1.7 deepstack
        }
        logger.info(f"Loaded shape hints from {metadata_path}: {seq_hints}")
    else:
        seq_hints = {
            "sa_seq_len": 51,  # 1 state + action_horizon
            "vl_seq_len": 280,  # typical backbone output seq_len
            "sequence_length": 280,  # LLM seq_len
        }
        logger.warning(
            f"No export_metadata.json found in {onnx_dir}, using default hints: {seq_hints}"
        )

    # Components: (name, onnx_file, engine_file)
    components = [
        # FP32 ViT preferred for accuracy; falls back to BF16 if only bf16 was
        # exported. Engine filename is precision-neutral (vit.engine) because
        # the input ONNX may be either FP32 or BF16; baking a precision tag
        # into the engine name was misleading whenever the FP32 ONNX path was
        # taken. The actual engine precision is recorded in
        # export_metadata.json and inspectable via TRT tooling.
        (
            "ViT",
            "vit_fp32.onnx"
            if os.path.exists(os.path.join(onnx_dir, "vit_fp32.onnx"))
            else "vit_bf16.onnx",
            "vit.engine",
        ),
        ("LLM", "llm_bf16.onnx", "llm_bf16.engine"),
        ("VL Self-Attention", "vl_self_attention.onnx", "vl_self_attention.engine"),
        ("State Encoder", "state_encoder.onnx", "state_encoder.engine"),
        ("Action Encoder", "action_encoder.onnx", "action_encoder.engine"),
        ("DiT", "dit_bf16.onnx", "dit_bf16.engine"),
        ("Action Decoder", "action_decoder.onnx", "action_decoder.engine"),
    ]

    results: list[tuple[str, str, str]] = []
    skipped: list[tuple[str, str]] = []  # (name, onnx_path) for components with no ONNX input

    for name, onnx_file, engine_file in components:
        onnx_path = os.path.join(onnx_dir, onnx_file)

        if not os.path.exists(onnx_path):
            logger.warning(f"Skipping {name}: ONNX file not found at {onnx_path}")
            skipped.append((name, onnx_path))
            continue

        logger.info(f"\n{'#' * 80}")
        logger.info(f"# Building {name} engine")
        logger.info(f"{'#' * 80}")

        engine_path = os.path.join(engine_dir, engine_file)
        # Pick the precision that actually matches this ONNX's tensor types.
        # The full_pipeline export is mixed-precision (ViT FP32, rest BF16),
        # so the pipeline-wide ``precision`` argument is the default but each
        # component uses what it was actually exported with.
        component_precision = _precision_from_onnx_path(onnx_path, default=precision)
        if component_precision != precision:
            logger.info(
                f"  Using precision={component_precision} for {name} (from ONNX filename); "
                f"pipeline default is {precision}"
            )

        try:
            # Derive shapes from the ONNX model itself
            min_shapes, opt_shapes, max_shapes = derive_shapes_with_hint(
                onnx_path, opt_seq_lens=seq_hints
            )

            logger.info(f"  Auto-derived shape profiles for {name}:")
            for input_name in opt_shapes:
                logger.info(
                    f"    {input_name}: min={min_shapes[input_name]} "
                    f"opt={opt_shapes[input_name]} max={max_shapes[input_name]}"
                )

            build_engine(
                onnx_path=onnx_path,
                engine_path=engine_path,
                precision=component_precision,
                workspace_mb=workspace_mb,
                min_shapes=min_shapes,
                opt_shapes=opt_shapes,
                max_shapes=max_shapes,
                trt_severity=trt_severity,
            )
            results.append((name, engine_path, "SUCCESS"))
        except Exception as e:
            logger.error(f"Failed to build {name} engine: {e}")
            results.append((name, engine_path, f"FAILED: {e}"))

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("FULL PIPELINE BUILD SUMMARY")
    logger.info("=" * 80)
    for name, path, status in results:
        logger.info(f"  {name:20s} -> {status}")
    logger.info("=" * 80)

    # Every component must build; missing ONNX inputs and failed builds are
    # equally fatal, otherwise an empty/half-built engine dir exits 0.
    failures = [(name, status) for name, _, status in results if status.startswith("FAILED")]
    if failures or skipped:
        parts = []
        if failures:
            parts.append(
                f"{len(failures)}/{len(components)} engine(s) failed: "
                + "; ".join(f"{name} ({status})" for name, status in failures)
            )
        if skipped:
            parts.append(
                f"{len(skipped)}/{len(components)} component(s) had no ONNX input: "
                + ", ".join(f"{name} ({path})" for name, path in skipped)
            )
        raise RuntimeError("Pipeline build incomplete — " + " | ".join(parts))


# ============================================================
# Main
# ============================================================


@dataclass
class BuildConfig:
    """Configuration for building TensorRT engines from ONNX models."""

    mode: BuildEngineMode = BuildEngineMode.single
    """Build mode: 'single' (one engine) or 'full_pipeline' (all engines)."""

    onnx: str | None = None
    """Path to ONNX model (single mode)."""

    engine: str | None = None
    """Path to save TensorRT engine (single mode)."""

    onnx_dir: str = "./gr00t_n1d7_onnx"
    """Directory with ONNX models (full_pipeline mode)."""

    engine_dir: str = "./gr00t_n1d7_engines"
    """Directory to save engines (full_pipeline mode)."""

    precision: Literal["fp32", "fp16", "bf16", "fp8"] = "bf16"
    """Precision mode (default: bf16)."""

    workspace: int = 8192
    """Workspace size in MB (default: 8192)."""


def main(args: BuildConfig | None = None, trt_severity=None):
    if args is None:
        args = tyro.cli(BuildConfig)

    if args.mode == "full_pipeline":
        build_full_pipeline(
            onnx_dir=args.onnx_dir,
            engine_dir=args.engine_dir,
            precision=args.precision,
            workspace_mb=args.workspace,
            trt_severity=trt_severity,
        )
    else:
        if not args.onnx or not args.engine:
            raise ValueError("--onnx and --engine are required in single mode")

        # Auto-derive shapes from the ONNX model
        min_shapes, opt_shapes, max_shapes = derive_shapes_with_hint(args.onnx)
        build_engine(
            onnx_path=args.onnx,
            engine_path=args.engine,
            precision=args.precision,
            workspace_mb=args.workspace,
            min_shapes=min_shapes,
            opt_shapes=opt_shapes,
            max_shapes=max_shapes,
            trt_severity=trt_severity,
        )


if __name__ == "__main__":
    config = tyro.cli(BuildConfig)
    main(config)
