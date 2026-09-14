#!/usr/bin/env python3
# ruff: noqa: T201
"""Build the complete fixed-shape N2 TensorRT bundle on its target GPU."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from groot.deployment.n2_trt.contract import TransformerPrecision
from groot.deployment.n2_trt.manifest import BUILD_FILENAME, RuntimeIdentity, sha256_file

TIMING_CACHE_FILENAME = "n2_trt_timing.cache"
TRANSFORMER_PRECISION = TransformerPrecision.BF16.value
ARTIFACTS = (
    (
        "transformer_joint",
        "n2_transformer_joint.onnx",
        "n2_transformer_joint.engine",
        TRANSFORMER_PRECISION,
    ),
    (
        "transformer_action_cache",
        "n2_transformer_action_cache.onnx",
        "n2_transformer_action_cache.engine",
        TRANSFORMER_PRECISION,
    ),
    ("vae_encoder", "n2_vae_encoder.onnx", "n2_vae_encoder.engine", "bf16"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--builder-optimization-level", type=int, default=3)
    return parser.parse_args()


def _strongly_typed_network_flags(trt: Any) -> int:
    strongly_typed = getattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None)
    if strongly_typed is None:
        raise RuntimeError("N2 TensorRT requires strongly typed network support")
    return (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) | (1 << int(strongly_typed))


def _tensor_spec(tensor: Any) -> dict[str, Any]:
    shape = tuple(int(dimension) for dimension in tensor.shape)
    if not shape or any(dimension <= 0 for dimension in shape):
        raise RuntimeError(f"N2 TensorRT requires fixed positive dimensions: {tensor.name}{shape}")
    dtype = str(tensor.dtype).rsplit(".", maxsplit=1)[-1].lower()
    return {"name": tensor.name, "shape": list(shape), "dtype": dtype}


def _engine_tensor_spec(engine: Any, name: str) -> dict[str, Any]:
    shape = tuple(int(dimension) for dimension in engine.get_tensor_shape(name))
    if not shape or any(dimension <= 0 for dimension in shape):
        raise RuntimeError(f"N2 TensorRT produced a dynamic tensor: {name}{shape}")
    dtype = str(engine.get_tensor_dtype(name)).rsplit(".", maxsplit=1)[-1].lower()
    return {"name": name, "shape": list(shape), "dtype": dtype}


def _onnx_members(onnx_path: Path, artifact_root: Path) -> list[dict[str, Any]]:
    """Return the ONNX protobuf and every external-data file it references."""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    physical_files = {onnx_path.resolve()}
    for initializer in model.graph.initializer:
        if initializer.data_location != onnx.TensorProto.EXTERNAL:
            continue
        external = {entry.key: entry.value for entry in initializer.external_data}
        location = external.get("location")
        if not location:
            raise RuntimeError(f"ONNX initializer {initializer.name!r} has no external location")
        candidate = (onnx_path.parent / location).resolve(strict=True)
        try:
            candidate.relative_to(artifact_root)
        except ValueError as error:
            raise RuntimeError(f"ONNX external data escapes artifact root: {candidate}") from error
        if not candidate.is_file() or candidate.stat().st_size <= 0:
            raise RuntimeError(f"ONNX external data is missing or empty: {candidate}")
        physical_files.add(candidate)
    return [
        {
            "relative_path": str(path.relative_to(artifact_root)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(physical_files, key=lambda item: str(item.relative_to(artifact_root)))
    ]


def _git_source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"component": "builder", "root": str(root), "revision": revision, "dirty": dirty}


def _build_engine(
    trt: Any,
    onnx_path: Path,
    engine_path: Path,
    *,
    artifact_root: Path,
    optimization_level: int,
    timing_cache_path: Path,
) -> dict[str, Any]:
    if engine_path.exists():
        raise FileExistsError(f"refusing to overwrite existing engine: {engine_path}")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(_strongly_typed_network_flags(trt))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(f"failed to parse {onnx_path}:\n{errors}")

    onnx_inputs = tuple(
        sorted(
            (_tensor_spec(network.get_input(index)) for index in range(network.num_inputs)),
            key=lambda spec: spec["name"],
        )
    )
    onnx_outputs = tuple(
        sorted(
            (_tensor_spec(network.get_output(index)) for index in range(network.num_outputs)),
            key=lambda spec: spec["name"],
        )
    )
    config = builder.create_builder_config()
    config.builder_optimization_level = optimization_level
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    cache_bytes = timing_cache_path.read_bytes() if timing_cache_path.is_file() else b""
    timing_cache = config.create_timing_cache(cache_bytes)
    config.set_timing_cache(timing_cache, ignore_mismatch=False)

    start = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build {onnx_path}")
    engine_path.write_bytes(bytes(serialized))
    timing_cache_path.write_bytes(config.get_timing_cache().serialize())
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError(f"TensorRT could not deserialize its new plan: {engine_path}")
    engine_inputs = []
    engine_outputs = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        spec = _engine_tensor_spec(engine, name)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            engine_inputs.append(spec)
        else:
            engine_outputs.append(spec)
    return {
        "onnx": onnx_path.name,
        "onnx_files": _onnx_members(onnx_path, artifact_root),
        "engine": engine_path.name,
        "engine_sha256": sha256_file(engine_path),
        "engine_size_bytes": engine_path.stat().st_size,
        "elapsed_s": time.perf_counter() - start,
        "onnx_inputs": onnx_inputs,
        "onnx_outputs": onnx_outputs,
        "engine_inputs": tuple(sorted(engine_inputs, key=lambda spec: spec["name"])),
        "engine_outputs": tuple(sorted(engine_outputs, key=lambda spec: spec["name"])),
    }


def main() -> int:
    args = _parse_args()
    root = args.artifact_root.expanduser().resolve(strict=True)
    if args.builder_optimization_level not in range(0, 6):
        raise ValueError("--builder-optimization-level must be between 0 and 5")
    build_path = root / BUILD_FILENAME
    if build_path.exists():
        raise FileExistsError(f"refusing to overwrite existing build receipt: {build_path}")
    required = [root / source for _role, source, _engine, _format in ARTIFACTS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"complete N2 ONNX bundle is required; missing {missing}")

    import tensorrt as trt

    timing_cache = root / TIMING_CACHE_FILENAME
    records = []
    for role, source_name, engine_name, numeric_format in ARTIFACTS:
        record = _build_engine(
            trt,
            root / source_name,
            root / engine_name,
            artifact_root=root,
            optimization_level=args.builder_optimization_level,
            timing_cache_path=timing_cache,
        )
        records.append({"role": role, "numeric_format": numeric_format, **record})
        print(f"[n2-trt-build] {engine_name}: {record['elapsed_s']:.1f}s")

    runtime = RuntimeIdentity.capture()
    payload = {
        "schema": "groot_n2_trt_build_v2",
        "quantize_mode": "none",
        "tensorrt": runtime.tensorrt,
        "source": _git_source_identity(),
        "runtime": asdict(runtime),
        "runtime_gpu_architecture": runtime.gpu_architecture,
        "settings": {
            "strongly_typed": True,
            "builder_optimization_level": args.builder_optimization_level,
            "transformer_precision": TRANSFORMER_PRECISION,
            "vae_precision": "bf16",
        },
        "timing_cache": timing_cache.name,
        "artifacts": records,
        # Stable consumer view used by the merged measurement contract. The
        # richer N2 records above remain authoritative for runtime validation.
        "engines": [
            {
                "engine": str((root / record["engine"]).resolve()),
                "size_bytes": record["engine_size_bytes"],
                "precision": record["numeric_format"],
            }
            for record in records
        ],
    }
    build_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[n2-trt-build] receipt={build_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
