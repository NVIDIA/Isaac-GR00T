# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Engine census sidecars: the verified per-layer output precisions of a built N2 engine.

Every admitted or benchmarked engine must sit on layer precisions that were READ from the
engine, never assumed from its recipe (a build once dropped every FP8 scale and passed for
"FP8" until it was inspected). This tool deserializes an engine, reads the per-layer output
dtypes through TensorRT's engine inspector (the build sets the detailed profiling verbosity
that makes them available) and writes ``<engine>.inspect.json`` beside the engine. The
``engine_sha256`` field joins the sidecar to the engine bytes: a mismatch means the engine
changed after inspection and the sidecar is stale. Dtype counts are a precision EXISTENCE
check ("does FP8 run at all?", "is the BF16 role FP8-free?"), never a per-layer time
attribution -- engines are Myelin-fused and layer names do not map to time.

Usage (on the host that built the engine):
    python -P -m groot.deployment.n2_trt.tools.inspect_engines ENGINE [ENGINE ...]
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import time
from typing import Any

from groot.deployment.n2_trt.manifest import sha256_file

INSPECTION_SUFFIX = ".inspect.json"
# The record keys every reader of a sidecar may rely on (the benches read ``engine_sha256``
# and ``layer_output_dtype_counts``; the rest is provenance).
INSPECTION_RECORD_KEYS = (
    "engine_path",
    "engine_sha256",
    "engine_size_bytes",
    "engine_mtime",
    "inspected_at",
    "host",
    "device",
    "tensorrt_version",
    "num_layers",
    "layer_output_dtype_counts",
    "layers_by_first_output_dtype",
    "layer_type_counts",
    "io_tensors",
    "sibling_json",
    "note",
)
INSPECTION_NOTE = (
    "dtype counts are a precision EXISTENCE check, never per-layer time attribution "
    "(engine is Myelin-fused)"
)
# Canonical dtype labels parsed from the inspector's "Format/Datatype" strings. Order
# matters: "BFloat16" contains "Float", so FP8 / BF16 / FP16 are matched before FP32.
_DTYPE_TOKENS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("FP8", ("fp8", "e4m3")),
    ("BF16", ("bf16", "bfloat16")),
    ("FP16", ("fp16", "half")),
    ("FP32", ("fp32", "float")),
    ("INT64", ("int64",)),
    ("INT32", ("int32",)),
    ("INT8", ("int8",)),
    ("BOOL", ("bool",)),
)


class EngineInspectionError(RuntimeError):
    """A missing or stale sidecar, or an engine whose census cannot be read."""


def canonical_dtype(format_string: str) -> str:
    lowered = format_string.lower()
    for label, tokens in _DTYPE_TOKENS:
        if any(token in lowered for token in tokens):
            return label
    return f"OTHER({format_string})"


def census_from_layers(layers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The dtype census of an inspector layer list (the pure core of an inspection)."""
    if not layers:
        raise EngineInspectionError("the engine inspector returned no layers")
    if not isinstance(layers[0], Mapping) or "Outputs" not in layers[0]:
        raise EngineInspectionError(
            "layer information carries no per-tensor dtypes: the engine was built without the "
            "detailed profiling verbosity, so its precisions cannot be verified"
        )
    output_counts: Counter[str] = Counter()
    first_output_counts: Counter[str] = Counter()
    layer_type_counts: Counter[str] = Counter()
    for layer in layers:
        layer_type_counts[str(layer.get("LayerType", "unknown"))] += 1
        for index, output in enumerate(layer.get("Outputs") or []):
            dtype = canonical_dtype(str(output.get("Format/Datatype", "")))
            output_counts[dtype] += 1
            if index == 0:
                first_output_counts[dtype] += 1
    return {
        "num_layers": len(layers),
        "layer_output_dtype_counts": dict(output_counts.most_common()),
        "layers_by_first_output_dtype": dict(first_output_counts.most_common()),
        "layer_type_counts": dict(layer_type_counts.most_common()),
    }


def sidecar_path(engine_path: str | Path) -> Path:
    engine = Path(engine_path)
    return engine.with_name(engine.name + INSPECTION_SUFFIX)


def inspect_engine(engine_path: str | Path, *, full_layers: bool = False) -> dict[str, Any]:
    """Read one engine's census with TensorRT (requires the runtime that built it)."""
    import tensorrt as trt

    engine_file = Path(engine_path)
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_file.read_bytes())
    if engine is None:
        raise EngineInspectionError(
            f"failed to deserialize {engine_file} (built for another SM or TensorRT version?)"
        )
    inspector = engine.create_engine_inspector()
    info = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    census = census_from_layers(info.get("Layers") or [])
    io_tensors = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        io_tensors.append(
            {
                "name": name,
                "mode": str(engine.get_tensor_mode(name)).rsplit(".", 1)[-1],
                "dtype": str(engine.get_tensor_dtype(name)).rsplit(".", 1)[-1],
                "shape": list(engine.get_tensor_shape(name)),
            }
        )
    device_name = None
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name()
    except Exception:  # noqa: BLE001 - provenance only; the census does not depend on it
        device_name = None
    stat = engine_file.stat()
    record: dict[str, Any] = {
        "engine_path": str(engine_file.resolve()),
        "engine_sha256": sha256_file(engine_file),
        "engine_size_bytes": stat.st_size,
        "engine_mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime)),
        "inspected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": os.uname().nodename,
        "device": device_name,
        "tensorrt_version": trt.__version__,
        **census,
        "io_tensors": io_tensors,
        "sibling_json": sorted(path.name for path in engine_file.parent.glob("*.json")),
        "note": INSPECTION_NOTE,
    }
    if full_layers:
        layers_path = engine_file.with_name(engine_file.name + ".layers.json")
        layers_path.write_text(json.dumps(info, indent=2) + "\n")
        record["full_layers_json"] = str(layers_path)
    del inspector, engine, runtime
    return record


def write_sidecar(engine_path: str | Path, record: Mapping[str, Any] | None = None) -> Path:
    """Inspect ``engine_path`` (or take a ready record) and write its sidecar beside it."""
    payload = dict(record) if record is not None else inspect_engine(engine_path)
    target = sidecar_path(engine_path)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return target


def read_sidecar(engine_path: str | Path) -> dict[str, Any]:
    path = sidecar_path(engine_path)
    if not path.is_file():
        raise EngineInspectionError(
            f"{Path(engine_path).name} has no {path.name}: inspect it first"
        )
    payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping):
        raise EngineInspectionError(f"{path} is not an inspection record")
    return dict(payload)


def require_current_sidecar(engine_path: str | Path) -> dict[str, Any]:
    """The engine's sidecar, refused when missing or written for other engine bytes."""
    record = read_sidecar(engine_path)
    actual = sha256_file(engine_path)
    if record.get("engine_sha256") != actual:
        raise EngineInspectionError(
            f"{sidecar_path(engine_path).name} is stale: its engine_sha256 does not match the "
            "engine file (the engine changed after inspection)"
        )
    counts = record.get("layer_output_dtype_counts")
    if not isinstance(counts, Mapping) or not counts:
        raise EngineInspectionError(
            f"{sidecar_path(engine_path).name} records no layer_output_dtype_counts"
        )
    return record


def census_check(
    counts: Mapping[str, Any], *, numeric_format: str, max_fp32_layer_outputs: int
) -> dict[str, Any]:
    """Judge one engine's census against its role: FP8 present iff FP8, FP32 within the cap."""
    fp32 = int(counts.get("FP32") or 0)
    fp8 = int(counts.get("FP8") or 0)
    failures: list[str] = []
    if numeric_format == "fp8" and fp8 <= 0:
        failures.append("an fp8 engine has no FP8 layer outputs")
    if numeric_format != "fp8" and fp8 > 0:
        failures.append(f"a {numeric_format} engine has {fp8} FP8 layer outputs")
    if fp32 > int(max_fp32_layer_outputs):
        failures.append(f"{fp32} FP32 layer outputs exceed the cap {max_fp32_layer_outputs}")
    return {
        "numeric_format": numeric_format,
        "fp32": fp32,
        "fp8": fp8,
        "max_fp32_layer_outputs": int(max_fp32_layer_outputs),
        "passed": not failures,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engines", nargs="+", type=Path, help="engine file(s) to inspect")
    parser.add_argument(
        "--full-layers",
        action="store_true",
        help="also dump the raw per-layer inspector JSON to <engine>.layers.json",
    )
    args = parser.parse_args()
    for engine in args.engines:
        if not engine.is_file():
            raise SystemExit(f"[n2-inspect] engine not found: {engine}")
        record = inspect_engine(engine, full_layers=args.full_layers)
        target = write_sidecar(engine, record)
        print(  # noqa: T201
            f"[n2-inspect] wrote {target}: {record['num_layers']} layers, "
            f"layer outputs by dtype = {record['layer_output_dtype_counts']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
