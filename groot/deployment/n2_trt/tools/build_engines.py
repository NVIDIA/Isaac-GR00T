#!/usr/bin/env python3
# ruff: noqa: T201
"""Build the complete fixed-shape N2 TensorRT bundle on its target GPU."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
from typing import Any

from groot.deployment.n2_trt.contract import (
    ACTION_CONTEXT_MODES,
    VAE_PRECISIONS,
    TransformerPrecision,
)
from groot.deployment.n2_trt.manifest import (
    BUILD_FILENAME,
    BUILD_SCHEMA,
    COND_CACHED_ENGINE_ROLE,
    RuntimeIdentity,
    SourceRevision,
    sha256_file,
    validation_sample_paths,
)

TIMING_CACHE_FILENAME = "n2_trt_timing.cache"
EXPORT_MANIFEST_FILENAME = "n2_transformer_export.json"
VAE_EXPORT_MANIFEST_FILENAME = "n2_vae_export_manifest.json"
ARTIFACTS = (
    ("transformer_joint", "n2_transformer_joint.onnx", "n2_transformer_joint.engine"),
    (
        "transformer_action_cache",
        "n2_transformer_action_cache.onnx",
        "n2_transformer_action_cache.engine",
    ),
    ("vae_encoder", "n2_vae_encoder.onnx", "n2_vae_encoder.engine"),
)
# The fourth engine a cond_kv_freeze export adds (its manifest records cond_kv_freeze=true).
COND_CACHED_ARTIFACT = (
    COND_CACHED_ENGINE_ROLE,
    "n2_transformer_joint_condcached.onnx",
    "n2_transformer_joint_condcached.engine",
)
TRANSFORMER_ROLES = ("transformer_joint", "transformer_action_cache")
# The export manifest's per-graph precision keys (the quantized export arm writes one
# value per graph; the stock exporter writes a single string for both).
EXPORT_MANIFEST_GRAPH_KEYS = {
    "transformer_joint": "joint",
    "transformer_action_cache": "action_cache",
    COND_CACHED_ENGINE_ROLE: "joint_condcached",
}


def _bundle_artifacts(cond_kv_freeze: bool) -> tuple[tuple[str, str, str], ...]:
    """(role, onnx, engine) per engine this bundle carries."""
    return ARTIFACTS + ((COND_CACHED_ARTIFACT,) if cond_kv_freeze else ())


def _transformer_roles(cond_kv_freeze: bool) -> tuple[str, ...]:
    return TRANSFORMER_ROLES + ((COND_CACHED_ENGINE_ROLE,) if cond_kv_freeze else ())


# The stock VAE exporter's precision (its manifest predates the field).
VAE_PRECISION = "bf16"
UNQUANTIZED_MODE = "none"
# Quantize/dequantize operators a ModelOpt export leaves in a graph.
QDQ_OP_TYPES = frozenset(
    {"TRT_FP8QuantizeLinear", "TRT_FP8DequantizeLinear", "QuantizeLinear", "DequantizeLinear"}
)
# onnx.TensorProto FLOAT8E4M3FN / FLOAT8E5M2: the element types standard Q/DQ dequantizes from.
FP8_ELEMENT_TYPES = frozenset({17, 18})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--builder-optimization-level", type=int, default=3)
    parser.add_argument(
        "--source-revision",
        help="Source revision to record when Git metadata is unavailable.",
    )
    parser.add_argument(
        "--reuse-engine-from",
        action="append",
        default=[],
        metavar="ROLE=ARTIFACT_ROOT",
        help=(
            "Copy one engine role from a validated bundle instead of building it, "
            "only when this bundle's ONNX bytes for that role are identical and the "
            "reference was built on this GPU/software stack (A/B byte-reuse of "
            "non-target engines). Repeatable; any mismatch fails closed."
        ),
    )
    return parser.parse_args()


def _parse_reuse_map(
    values: list[str], artifacts: tuple[tuple[str, str, str], ...] = ARTIFACTS
) -> dict[str, Path]:
    roles = {role for role, _source, _engine in artifacts}
    reuse: dict[str, Path] = {}
    for value in values:
        role, separator, root = value.partition("=")
        if not separator or role not in roles or not root:
            raise ValueError(
                f"--reuse-engine-from expects ROLE=ARTIFACT_ROOT with ROLE in {sorted(roles)}; got {value!r}"
            )
        if role in reuse:
            raise ValueError(f"--reuse-engine-from names {role!r} twice")
        reuse[role] = Path(root).expanduser().resolve(strict=True)
    return reuse


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
    tensors = [(f"initializer {init.name!r}", init) for init in model.graph.initializer]
    # Node-attribute tensors (Constant values) count as members too: a plan is only
    # byte-reusable when every file TensorRT read at build time is identical.
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                tensors.append((f"attribute of {node.name!r}", attribute.t))
            elif attribute.type == onnx.AttributeProto.TENSORS:
                tensors.extend((f"attribute of {node.name!r}", t) for t in attribute.tensors)
    for label, tensor in tensors:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        external = {entry.key: entry.value for entry in tensor.external_data}
        location = external.get("location")
        if not location:
            raise RuntimeError(f"ONNX {label} has no external location")
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


def _declared_precisions(root: Path, cond_kv_freeze: bool = False) -> tuple[dict[str, str], str]:
    """Per-role transformer precision and quantize mode the export manifest declares.

    The stock exporter writes a single BF16 literal and no quantize mode; the
    quantized export arm writes one precision per graph plus its quantize mode. An
    absent manifest is the BF16 declaration. The graph census must agree with the
    declaration before anything is receipted.
    """
    roles = _transformer_roles(cond_kv_freeze)
    path = root / EXPORT_MANIFEST_FILENAME
    if not path.is_file():
        return dict.fromkeys(roles, TransformerPrecision.BF16.value), UNQUANTIZED_MODE
    manifest = json.loads(path.read_text())
    declared = manifest.get("transformer_precision", TransformerPrecision.BF16.value)
    if isinstance(declared, str):
        precisions = dict.fromkeys(roles, declared)
    elif isinstance(declared, dict):
        precisions = {role: declared.get(EXPORT_MANIFEST_GRAPH_KEYS[role]) for role in roles}
    else:
        raise TypeError(
            f"export manifest transformer_precision must be a string or a per-graph object: {declared!r}"
        )
    for role, value in precisions.items():
        if value not in {member.value for member in TransformerPrecision}:
            raise ValueError(f"export manifest declares an unsupported {role} precision: {value!r}")
    quantize_mode = manifest.get("quantize_mode") or UNQUANTIZED_MODE
    if not isinstance(quantize_mode, str):
        raise TypeError(f"export manifest quantize_mode must be a string: {quantize_mode!r}")
    quantized = any(value != TransformerPrecision.BF16.value for value in precisions.values())
    if quantized == (quantize_mode == UNQUANTIZED_MODE):
        raise ValueError(
            f"export manifest quantize_mode {quantize_mode!r} contradicts its transformer precisions {precisions}"
        )
    return precisions, quantize_mode


def _declared_vae_precision(root: Path) -> tuple[str, str]:
    """The VAE encoder precision and quantize mode the VAE export manifest declares.

    The stock VAE exporter writes neither field (a BF16 transcription); the quantized
    VAE export arm declares ``vae_precision`` and its quantize mode. The graph census
    must agree with the declaration before anything is receipted.
    """
    path = root / VAE_EXPORT_MANIFEST_FILENAME
    if not path.is_file():
        return VAE_PRECISION, UNQUANTIZED_MODE
    manifest = json.loads(path.read_text())
    precision = manifest.get("vae_precision", VAE_PRECISION)
    if precision not in VAE_PRECISIONS:
        raise ValueError(f"VAE export manifest declares an unsupported precision: {precision!r}")
    quantize_mode = manifest.get("quantize_mode") or UNQUANTIZED_MODE
    if not isinstance(quantize_mode, str):
        raise TypeError(f"VAE export manifest quantize_mode must be a string: {quantize_mode!r}")
    if (precision != VAE_PRECISION) == (quantize_mode == UNQUANTIZED_MODE):
        raise ValueError(
            f"VAE export manifest quantize_mode {quantize_mode!r} contradicts its vae_precision {precision!r}"
        )
    return precision, quantize_mode


def _bundle_quantize_mode(transformer_mode: str, vae_mode: str) -> str:
    """One quantize mode for the receipt: the quantized roles' mode, ``none`` for a BF16 bundle."""
    if transformer_mode == UNQUANTIZED_MODE:
        return vae_mode
    if vae_mode not in (UNQUANTIZED_MODE, transformer_mode):
        raise ValueError(
            f"transformer export quantize_mode {transformer_mode!r} and VAE export quantize_mode "
            f"{vae_mode!r} disagree"
        )
    return transformer_mode


def _graph_precision_census(onnx_path: Path) -> str:
    """The transformer precision the ONNX graph itself carries (a Q/DQ census)."""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    qdq_nodes = [node for node in model.graph.node if node.op_type in QDQ_OP_TYPES]
    if not qdq_nodes:
        return TransformerPrecision.BF16.value
    if any(node.op_type.startswith("TRT_FP8") for node in qdq_nodes):
        return TransformerPrecision.FP8.value
    initializer_types = {
        initializer.name: initializer.data_type for initializer in model.graph.initializer
    }
    dequantized_types = {
        initializer_types[node.input[0]]
        for node in qdq_nodes
        if node.op_type == "DequantizeLinear" and node.input and node.input[0] in initializer_types
    }
    if dequantized_types and dequantized_types <= FP8_ELEMENT_TYPES:
        return TransformerPrecision.FP8.value
    raise RuntimeError(
        f"{onnx_path.name}: unsupported quantized graph ({len(qdq_nodes)} Q/DQ nodes, "
        f"dequantized element types {sorted(dequantized_types)})"
    )


def _source_identity(source_revision: str | None = None) -> dict[str, Any]:
    """Describe the builder source without requiring a Git checkout."""
    root = Path(__file__).resolve().parents[3]
    source = SourceRevision.capture("builder", __file__, revision=source_revision)
    return {"root": str(root), **asdict(source)}


def _exported_validation_sample_digests(root: Path, cond_kv_freeze: bool = False) -> dict[str, str]:
    """Load the sample identities written at the two export boundaries."""
    manifests = (
        root / "n2_transformer_export.json",
        root / "n2_vae_export_manifest.json",
    )
    digests: dict[str, str] = {}
    for path in manifests:
        payload = json.loads(path.read_text())
        exported = payload.get("validation_sample_sha256")
        if not isinstance(exported, dict):
            raise TypeError(f"export manifest omits validation sample digests: {path}")
        overlap = set(digests) & set(exported)
        if overlap:
            raise ValueError(f"duplicate validation sample digests: {sorted(overlap)}")
        digests.update(exported)
    expected = validation_sample_paths(cond_kv_freeze)
    if set(digests) != set(expected):
        raise ValueError(
            "export manifests must identify every validation sample: "
            f"expected {sorted(expected)}, got {sorted(digests)}"
        )
    return digests


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


def _reuse_engine(
    role: str,
    source_name: str,
    engine_name: str,
    numeric_format: str,
    *,
    artifact_root: Path,
    reference_root: Path,
    runtime: RuntimeIdentity,
) -> dict[str, Any]:
    """Copy a target-local engine whose ONNX bytes are identical to this bundle's.

    Byte-reuse keeps non-target engines fixed across an A/B. The reference bundle's
    build receipt must carry this GPU/software identity, its ONNX members for the
    role must match this bundle's members exactly, and the copied plan must hash to
    the recorded digest. Any mismatch fails closed instead of rebuilding.
    """
    engine_path = artifact_root / engine_name
    if engine_path.exists():
        raise FileExistsError(f"refusing to overwrite existing engine: {engine_path}")
    receipt = json.loads((reference_root / BUILD_FILENAME).read_text())
    if receipt.get("schema") != BUILD_SCHEMA:
        raise ValueError(f"reference bundle has an unsupported build receipt: {reference_root}")
    reference_runtime = dict(receipt["runtime"])
    reference_runtime["compute_capability"] = tuple(reference_runtime["compute_capability"])
    mismatches = RuntimeIdentity(**reference_runtime).compatibility_mismatches(runtime)
    if mismatches:
        raise RuntimeError(
            f"reference bundle for {role!r} was built on a different GPU/software stack: "
            + json.dumps(mismatches, sort_keys=True, default=str)
        )
    matches = [record for record in receipt.get("artifacts", []) if record.get("role") == role]
    if len(matches) != 1:
        raise ValueError(f"reference bundle does not record exactly one {role!r} engine")
    reference = matches[0]
    if reference.get("numeric_format") != numeric_format:
        raise ValueError(f"reference {role!r} engine is {reference.get('numeric_format')!r}")
    members = _onnx_members(artifact_root / source_name, artifact_root)
    if members != reference.get("onnx_files"):
        raise RuntimeError(
            f"ONNX for {role!r} differs from the reference bundle; byte-reuse refused"
        )
    reference_engine = (reference_root / reference["engine"]).resolve(strict=True)
    if sha256_file(reference_engine) != reference["engine_sha256"]:
        raise RuntimeError(f"reference engine changed after its build: {reference_engine}")
    shutil.copyfile(reference_engine, engine_path)
    digest = sha256_file(engine_path)
    if digest != reference["engine_sha256"]:
        raise RuntimeError(f"copied engine does not match the reference digest: {engine_path}")
    return {
        "onnx": source_name,
        "onnx_files": members,
        "engine": engine_name,
        "engine_sha256": digest,
        "engine_size_bytes": engine_path.stat().st_size,
        "elapsed_s": 0.0,
        "onnx_inputs": reference["onnx_inputs"],
        "onnx_outputs": reference["onnx_outputs"],
        "engine_inputs": reference["engine_inputs"],
        "engine_outputs": reference["engine_outputs"],
        "reused_from": str(reference_root),
    }


def main() -> int:
    args = _parse_args()
    root = args.artifact_root.expanduser().resolve(strict=True)
    if args.builder_optimization_level not in range(0, 6):
        raise ValueError("--builder-optimization-level must be between 0 and 5")
    build_path = root / BUILD_FILENAME
    if build_path.exists():
        raise FileExistsError(f"refusing to overwrite existing build receipt: {build_path}")
    required = [root / source for _role, source, _engine in ARTIFACTS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"complete N2 ONNX bundle is required; missing {missing}")
    transformer_export = json.loads((root / "n2_transformer_export.json").read_text())
    # Exports written before the field existed carry no conditioning-cached joint graph.
    cond_kv_freeze = transformer_export.get("cond_kv_freeze", False)
    if not isinstance(cond_kv_freeze, bool):
        raise TypeError(f"transformer export cond_kv_freeze must be a boolean: {cond_kv_freeze!r}")
    artifacts = _bundle_artifacts(cond_kv_freeze)
    required = [root / source for _role, source, _engine in artifacts]
    missing = [str(path) for path in required if not path.is_file()]
    sample_paths = {
        role: root / path for role, path in validation_sample_paths(cond_kv_freeze).items()
    }
    missing.extend(str(path) for path in sample_paths.values() if not path.is_file())
    if missing:
        raise FileNotFoundError(f"complete N2 ONNX bundle is required; missing {missing}")
    # The receipt tells the truth about each graph: the export manifests' declarations
    # (per transformer role, and the VAE's), cross-checked against each graph's own
    # Q/DQ census.
    declared, transformer_quantize_mode = _declared_precisions(root, cond_kv_freeze)
    vae_precision, vae_quantize_mode = _declared_vae_precision(root)
    quantize_mode = _bundle_quantize_mode(transformer_quantize_mode, vae_quantize_mode)
    numeric_formats = {**declared, "vae_encoder": vae_precision}
    for role, source_name, _engine_name in artifacts:
        census = _graph_precision_census(root / source_name)
        if census != numeric_formats[role]:
            raise RuntimeError(
                f"{source_name}: export manifest declares {numeric_formats[role]!r} but the graph "
                f"carries {census!r}"
            )
    exported_sample_digests = _exported_validation_sample_digests(root, cond_kv_freeze)
    actual_sample_digests = {role: sha256_file(path) for role, path in sample_paths.items()}
    if exported_sample_digests != actual_sample_digests:
        raise RuntimeError("validation samples changed after export")
    codec = transformer_export.get("codec_roundtrip")
    if codec not in {"h264", "mp4v", "none"}:
        raise ValueError(f"transformer export has invalid serving codec: {codec!r}")
    # Exports written before the field existed are raw-context exports.
    action_context_mode = transformer_export.get("action_context_mode", "raw")
    if action_context_mode not in ACTION_CONTEXT_MODES:
        raise ValueError(
            f"transformer export has invalid action context mode: {action_context_mode!r}"
        )
    reuse = _parse_reuse_map(args.reuse_engine_from, artifacts)

    import tensorrt as trt

    runtime = RuntimeIdentity.capture()
    timing_cache = root / TIMING_CACHE_FILENAME
    records = []
    for role, source_name, engine_name in artifacts:
        if role in reuse:
            record = _reuse_engine(
                role,
                source_name,
                engine_name,
                numeric_formats[role],
                artifact_root=root,
                reference_root=reuse[role],
                runtime=runtime,
            )
            records.append({"role": role, "numeric_format": numeric_formats[role], **record})
            print(f"[n2-trt-build] {engine_name}: reused byte-identical plan from {reuse[role]}")
            continue
        record = _build_engine(
            trt,
            root / source_name,
            root / engine_name,
            artifact_root=root,
            optimization_level=args.builder_optimization_level,
            timing_cache_path=timing_cache,
        )
        records.append({"role": role, "numeric_format": numeric_formats[role], **record})
        print(f"[n2-trt-build] {engine_name}: {record['elapsed_s']:.1f}s")

    payload = {
        "schema": BUILD_SCHEMA,
        "quantize_mode": quantize_mode,
        "tensorrt": runtime.tensorrt,
        "source": _source_identity(args.source_revision),
        "runtime": asdict(runtime),
        "runtime_gpu_architecture": runtime.gpu_architecture,
        "settings": {
            "strongly_typed": True,
            "builder_optimization_level": args.builder_optimization_level,
            "transformer_precision": declared["transformer_joint"],
            "vae_precision": vae_precision,
            "codec": codec,
            "action_context_mode": action_context_mode,
            "cond_kv_freeze": cond_kv_freeze,
        },
        "timing_cache": timing_cache.name,
        "validation_samples": exported_sample_digests,
        "artifacts": records,
        # Stable consumer view used by the merged measurement contract. The
        # richer N2 records above remain authoritative for runtime validation.
        "engines": [
            {
                "engine": record["engine"],
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
