# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Post-build eager-sample versus target-local TensorRT parity qualification."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import torch

from groot.deployment.n2_trt.artifacts import resolve_checkpoint_vae_weights
from groot.deployment.n2_trt.contract import N2TensorRTContract
from groot.deployment.n2_trt.manifest import (
    BUILD_FILENAME,
    BUILD_SCHEMA,
    MANIFEST_FILENAME,
    TRANSFORMER_EAGER_ORACLE,
    TRANSFORMER_EXPORT_ACTION_ORACLE,
    TRANSFORMER_EXPORT_JOINT_ORACLE,
    VAE_EAGER_ORACLE,
    VALIDATION_FILENAME,
    VALIDATION_POLICY,
    VALIDATION_POLICY_SHA256,
    VALIDATION_PROFILE_THRESHOLDS,
    VALIDATION_SAMPLE_PATHS,
    VALIDATION_SCHEMA,
    ArtifactRecord,
    BuildSettings,
    EngineManifest,
    RuntimeIdentity,
    SourceRevision,
    TensorSpec,
    canonical_sha256,
    checkpoint_file_hashes,
    sha256_file,
)
from groot.deployment.n2_trt.runtime import N2TensorTrtRunner

PROFILES = VALIDATION_PROFILE_THRESHOLDS


class ParityValidationError(RuntimeError):
    """Fail-closed parity result whose receipt has already been retained."""


def _compare(actual: torch.Tensor, expected: torch.Tensor, *, profile: str) -> dict[str, Any]:
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"parity shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    actual_dtype = actual.dtype
    expected_dtype = expected.dtype
    dtype_match = actual_dtype == expected_dtype
    actual = actual.float()
    expected = expected.to(device=actual.device).float()
    thresholds = PROFILES[profile]
    actual_finite = bool(torch.isfinite(actual).all().item())
    expected_finite = bool(torch.isfinite(expected).all().item())
    finite = actual_finite and expected_finite
    max_abs = None
    relative_rmse = None
    cosine = None
    if finite:
        difference = actual - expected
        epsilon = torch.finfo(torch.float32).eps
        max_abs = float(difference.abs().max().item())
        relative_rmse = float(
            (
                difference.square().mean().sqrt()
                / expected.square().mean().sqrt().clamp_min(epsilon)
            ).item()
        )
        cosine = float(
            torch.nn.functional.cosine_similarity(
                actual.flatten(), expected.flatten(), dim=0, eps=epsilon
            ).item()
        )
    gating_metrics = ["shape", "dtype", "finite", "relative_rmse"]
    if thresholds["min_cosine"] is not None:
        gating_metrics.append("cosine")
    record = {
        "shape": list(actual.shape),
        "actual_dtype": str(actual_dtype),
        "expected_dtype": str(expected_dtype),
        "dtype_match": dtype_match,
        "actual_finite": actual_finite,
        "expected_finite": expected_finite,
        "finite": finite,
        "max_abs": max_abs,
        "relative_rmse": relative_rmse,
        "cosine": cosine,
        "profile": profile,
        "thresholds": thresholds,
        "gating_metrics": gating_metrics,
        "diagnostic_metrics": ["max_abs"],
    }
    record["passed"] = (
        record["finite"]
        and record["dtype_match"]
        and record["relative_rmse"] is not None
        and record["relative_rmse"] <= thresholds["max_relative_rmse"]
        and (
            thresholds["min_cosine"] is None
            or (record["cosine"] is not None and record["cosine"] >= thresholds["min_cosine"])
        )
    )
    return record


def _load_sample(
    path: Path,
    *,
    output_name: str | None = None,
) -> dict[str, Any]:
    sample = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(sample, dict):
        raise TypeError(f"validation sample must be a tensor mapping: {path}")
    if output_name is not None and output_name not in sample:
        # Production-native captures historically used the transformer's
        # generic ``output`` name for both roles. Normalize that retained
        # representation once at the load boundary; export-source samples
        # already carry their exact engine binding names.
        if (
            output_name == "output1"
            and sample.get("oracle") == TRANSFORMER_EAGER_ORACLE
            and "output" in sample
        ):
            sample = {**sample, output_name: sample["output"]}
        else:
            raise KeyError(f"validation sample is missing required output {output_name!r}: {path}")
    return sample


def _require_oracle(sample: dict[str, Any], expected: str) -> None:
    actual = sample.get("oracle")
    if actual != expected:
        raise RuntimeError(f"validation sample oracle {actual!r} != {expected!r}")


def _runner_inputs(runner: N2TensorTrtRunner, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
    missing = set(runner.inputs) - set(sample)
    if missing:
        raise KeyError(f"validation sample is missing engine inputs: {sorted(missing)}")
    return {
        name: sample[name].to(device="cuda", dtype=metadata["dtype"]).contiguous()
        for name, metadata in runner.inputs.items()
    }


def _run(runner: N2TensorTrtRunner, inputs: dict[str, torch.Tensor]) -> None:
    for _ in range(2):
        runner(inputs)
    torch.cuda.synchronize()


def _joint_gates(
    engine: Path,
    sample_path: Path,
    *,
    expected_oracle: str = TRANSFORMER_EXPORT_JOINT_ORACLE,
    runner: N2TensorTrtRunner | None = None,
) -> tuple[dict[str, Any], dict[str, Any], N2TensorTrtRunner, dict[str, Any]]:
    sample = _load_sample(sample_path)
    _require_oracle(sample, expected_oracle)
    runner = runner or N2TensorTrtRunner(str(engine))
    _run(runner, _runner_inputs(runner, sample))
    joint = _compare(
        runner.outputs["output"].clone(),
        sample["output"],
        profile="joint_hidden",
    )

    if {"output2", "output3"} <= set(runner.outputs):
        pairs = (
            (
                "output2",
                "output2" if "output2" in sample else "kv_cache_key",
                "kv_cache_key",
            ),
            (
                "output3",
                "output3" if "output3" in sample else "kv_cache_value",
                "kv_cache_value",
            ),
        )
    else:
        pairs = tuple(
            (name, name, name)
            for name in (
                "kv_cache_key_fine",
                "kv_cache_value_fine",
                "kv_cache_key_coarse",
                "kv_cache_value_coarse",
            )
        )
    full_components = {
        component_name: _compare(
            runner.outputs[output_name].clone(),
            sample[sample_name],
            profile="bf16",
        )
        for output_name, sample_name, component_name in pairs
    }
    static_components = {}
    for output_name, sample_name, component_name in pairs:
        coarse = component_name.endswith("_coarse")
        indices = _static_indices(sample, coarse=coarse)
        actual = runner.outputs[output_name]
        static_components[component_name] = _compare(
            actual.index_select(3, indices.to(device=actual.device)).clone(),
            sample[sample_name].index_select(3, indices),
            profile="bf16",
        )
    kv_diagnostics = {
        "full": {
            "passed": all(record["passed"] for record in full_components.values()),
            "components": full_components,
        },
        "static": {
            "passed": all(record["passed"] for record in static_components.values()),
            "components": static_components,
        },
    }
    return joint, kv_diagnostics, runner, sample


def _static_indices(sample: dict[str, Any], *, coarse: bool) -> torch.Tensor:
    modality_ids = sample["modality_ids"][0]
    if coarse:
        modality_ids = modality_ids.index_select(0, sample["coarse_token_idx"].long())
    return torch.nonzero(modality_ids != 3).flatten()


def _action_gate(
    engine: Path,
    sample_path: Path,
    *,
    expected_oracle: str = TRANSFORMER_EXPORT_ACTION_ORACLE,
    runner: N2TensorTrtRunner | None = None,
) -> tuple[dict[str, Any], N2TensorTrtRunner, dict[str, Any]]:
    sample = _load_sample(sample_path, output_name="output1")
    _require_oracle(sample, expected_oracle)
    runner = runner or N2TensorTrtRunner(str(engine))
    _run(runner, _runner_inputs(runner, sample))
    return (
        _compare(
            runner.outputs["output1"].clone(),
            sample["output1"],
            profile="cached_action",
        ),
        runner,
        sample,
    )


def _composed_action_gate(
    *,
    joint_runner: N2TensorTrtRunner,
    joint_sample: dict[str, Any],
    action_runner: N2TensorTrtRunner,
    action_sample: dict[str, Any],
) -> dict[str, Any]:
    inputs = _runner_inputs(action_runner, action_sample)
    if "kv_cache_key" in action_runner.inputs:
        mappings = (
            ("kv_cache_key", "output2", False),
            ("kv_cache_value", "output3", False),
        )
    else:
        mappings = tuple(
            (name, name, name.endswith("_coarse"))
            for name in (
                "kv_cache_key_fine",
                "kv_cache_value_fine",
                "kv_cache_key_coarse",
                "kv_cache_value_coarse",
            )
        )
    for action_name, joint_name, coarse in mappings:
        indices = _static_indices(joint_sample, coarse=coarse)
        joint_sample_name = joint_name if joint_name in joint_sample else action_name
        source_static = joint_sample[joint_sample_name].index_select(3, indices)
        if not torch.equal(source_static, action_sample[action_name]):
            raise RuntimeError(
                f"joint/action samples are not one matched cache pair: {action_name}"
            )
        joint_output = joint_runner.outputs[joint_name]
        inputs[action_name] = joint_output.index_select(
            3,
            indices.to(device=joint_output.device),
        ).clone()
    _run(action_runner, inputs)
    return _compare(
        action_runner.outputs["output1"].clone(),
        action_sample["output1"],
        profile="composed_action",
    )


def _vae_gate(engine: Path, sample_path: Path) -> dict[str, Any]:
    sample = _load_sample(sample_path)
    _require_oracle(sample, VAE_EAGER_ORACLE)
    runner = N2TensorTrtRunner(str(engine))
    _run(runner, _runner_inputs(runner, sample))
    return _compare(runner.outputs["latents"].clone(), sample["latents"], profile="bf16")


def _tensor_specs(records: list[dict[str, Any]]) -> tuple[TensorSpec, ...]:
    return tuple(
        TensorSpec(name=record["name"], shape=tuple(record["shape"]), dtype=record["dtype"])
        for record in records
    )


def _artifact_records(
    root: Path, build: dict[str, Any]
) -> tuple[tuple[ArtifactRecord, ...], tuple[ArtifactRecord, ...]]:
    sources = []
    engines = []
    for record in build["artifacts"]:
        onnx_files = record.get("onnx_files")
        if not isinstance(onnx_files, list) or not onnx_files:
            raise ValueError(f"build receipt omits ONNX members for {record['role']}")
        expected_members = {
            member["relative_path"]: (member["sha256"], member["size_bytes"])
            for member in onnx_files
        }
        for relative_path, (expected_sha256, expected_size) in expected_members.items():
            path = (root / relative_path).resolve(strict=True)
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError(f"build receipt path escapes artifact root: {path}") from error
            if path.stat().st_size != expected_size or sha256_file(path) != expected_sha256:
                raise RuntimeError(f"ONNX changed after TensorRT build: {relative_path}")
        engine_path = (root / record["engine"]).resolve(strict=True)
        if (
            engine_path.stat().st_size != record["engine_size_bytes"]
            or sha256_file(engine_path) != record["engine_sha256"]
        ):
            raise RuntimeError(f"engine changed after TensorRT build: {record['engine']}")

        onnx_inputs = _tensor_specs(record["onnx_inputs"])
        onnx_outputs = _tensor_specs(record["onnx_outputs"])
        engine_inputs = _tensor_specs(record["engine_inputs"])
        engine_outputs = _tensor_specs(record["engine_outputs"])
        onnx_path = root / record["onnx"]
        auxiliary_files = [
            root / member["relative_path"]
            for member in onnx_files
            if member["relative_path"] != str(onnx_path.relative_to(root))
        ]
        sources.append(
            ArtifactRecord.from_files(
                onnx_path,
                auxiliary_files=auxiliary_files,
                role=f"{record['role']}_onnx",
                root=root,
                kind="onnx",
                numeric_format=record["numeric_format"],
                inputs=onnx_inputs,
                outputs=onnx_outputs,
            )
        )
        engines.append(
            ArtifactRecord.from_file(
                engine_path,
                role=record["role"],
                root=root,
                kind="engine",
                numeric_format=record["numeric_format"],
                inputs=engine_inputs,
                outputs=engine_outputs,
            )
        )
    return tuple(sources), tuple(engines)


def _source_revisions(
    root: Path, build: dict[str, Any], source_revision: str | None
) -> tuple[SourceRevision, ...]:
    transformer_export = json.loads((root / "n2_transformer_export.json").read_text())
    vae_export = json.loads((root / "n2_vae_export_manifest.json").read_text())
    transformer_source = transformer_export["source"]
    vae_source = vae_export["source"]
    if transformer_source != vae_source:
        raise RuntimeError("transformer and VAE were exported from different source revisions")
    builder_source = build["source"]
    return (
        SourceRevision(**transformer_source),
        SourceRevision(
            component="builder",
            revision=builder_source["revision"],
            dirty=builder_source["dirty"],
        ),
        SourceRevision.capture("validator", __file__, revision=source_revision),
    )


def _validated_runtime_identity(build: dict[str, Any]) -> RuntimeIdentity:
    """Authenticate the GPU/software stack that executes parity validation."""
    build_runtime_payload = dict(build["runtime"])
    build_runtime_payload["compute_capability"] = tuple(build_runtime_payload["compute_capability"])
    build_runtime = RuntimeIdentity(**build_runtime_payload)
    validator_runtime = RuntimeIdentity.capture()
    runtime_mismatch = build_runtime.compatibility_mismatches(validator_runtime)
    if runtime_mismatch:
        raise RuntimeError(
            "N2 TensorRT validation must run on the build GPU and software stack: "
            + json.dumps(runtime_mismatch, sort_keys=True, default=str)
        )
    return validator_runtime


def validate_bundle(
    *,
    artifact_root: Path,
    checkpoint: Path,
    embodiment: str,
    source_revision: str | None = None,
) -> dict[str, Any]:
    root = artifact_root.expanduser().resolve(strict=True)
    contract = N2TensorRTContract.from_checkpoint(
        checkpoint,
        embodiment=embodiment,
        video_denoising_steps=2,
        action_denoising_steps=4,
        joint_denoise=True,
        conditional_kv_cache=True,
    )
    build = json.loads((root / BUILD_FILENAME).read_text())
    if build.get("schema") != BUILD_SCHEMA:
        raise ValueError(f"unsupported build receipt schema: {build.get('schema')!r}")
    sample_paths = {
        role: (root / relative_path).resolve(strict=True)
        for role, relative_path in VALIDATION_SAMPLE_PATHS.items()
    }
    expected_samples = build.get("validation_samples")
    actual_samples = {role: sha256_file(path) for role, path in sample_paths.items()}
    if expected_samples != actual_samples:
        raise RuntimeError("validation samples changed after the TensorRT engine build")
    vae_export = json.loads((root / "n2_vae_export_manifest.json").read_text())
    exported_vae_weights = vae_export.get("vae_weights")
    if not isinstance(exported_vae_weights, str) or not exported_vae_weights:
        raise RuntimeError("VAE export manifest omits the resolved weights path")
    vae_weights = resolve_checkpoint_vae_weights(checkpoint, exported_vae_weights)
    vae_weights_sha256 = sha256_file(vae_weights)
    if vae_export.get("vae_weights_sha256") != vae_weights_sha256:
        raise RuntimeError("VAE weights changed after export")
    validator_runtime = _validated_runtime_identity(build)
    sources, engines = _artifact_records(root, build)
    engine_paths = {record.role: root / record.relative_path for record in engines}

    joint_export_path = sample_paths["transformer_joint_export"]
    action_export_path = sample_paths["transformer_action_export"]
    joint_production_path = sample_paths["transformer_joint_production"]
    action_production_path = sample_paths["transformer_action_production"]
    vae_sample_path = sample_paths["vae_eager"]
    joint_export, joint_export_kv, joint_runner, joint_export_sample = _joint_gates(
        engine_paths["transformer_joint"], joint_export_path
    )
    action_export, action_runner, action_export_sample = _action_gate(
        engine_paths["transformer_action_cache"], action_export_path
    )
    composed_export = _composed_action_gate(
        joint_runner=joint_runner,
        joint_sample=joint_export_sample,
        action_runner=action_runner,
        action_sample=action_export_sample,
    )
    joint_production, joint_production_kv, _, joint_production_sample = _joint_gates(
        engine_paths["transformer_joint"],
        joint_production_path,
        expected_oracle=TRANSFORMER_EAGER_ORACLE,
        runner=joint_runner,
    )
    action_production, _, action_production_sample = _action_gate(
        engine_paths["transformer_action_cache"],
        action_production_path,
        expected_oracle=TRANSFORMER_EAGER_ORACLE,
        runner=action_runner,
    )
    composed_production = _composed_action_gate(
        joint_runner=joint_runner,
        joint_sample=joint_production_sample,
        action_runner=action_runner,
        action_sample=action_production_sample,
    )
    gates = {
        "transformer_joint_export": joint_export,
        "transformer_action_cache_export": action_export,
        "transformer_composed_action_export": composed_export,
        "transformer_joint_production": joint_production,
        "transformer_action_cache_production": action_production,
        "transformer_composed_action_production": composed_production,
        "vae_encoder": _vae_gate(engine_paths["vae_encoder"], vae_sample_path),
    }
    checkpoint_files = checkpoint_file_hashes(contract.checkpoint)
    payload = {
        "schema": VALIDATION_SCHEMA,
        "validation_policy": VALIDATION_POLICY,
        "validation_policy_sha256": VALIDATION_POLICY_SHA256,
        "contract_sha256": canonical_sha256(contract.to_dict()),
        "checkpoint_sha256": canonical_sha256(checkpoint_files),
        "vae_weights_path": str(vae_weights),
        "vae_weights_sha256": vae_weights_sha256,
        "artifacts": {record.role: record.sha256 for record in (*sources, *engines)},
        "oracles": actual_samples,
        "gates": gates,
        "diagnostics": {
            "transformer_kv_export": joint_export_kv,
            "transformer_kv_production": joint_production_kv,
        },
        "validator_runtime": asdict(validator_runtime),
        "passed": all(record["passed"] for record in gates.values()),
    }
    receipt_path = root / VALIDATION_FILENAME
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if not payload["passed"]:
        raise ParityValidationError(f"N2 TensorRT parity failed; receipt={receipt_path}")

    settings = BuildSettings(**build["settings"])
    validation = ArtifactRecord.from_file(
        receipt_path,
        role="engine_validation",
        root=root,
        kind="validation",
        numeric_format="json",
    )
    build_record = ArtifactRecord.from_file(
        root / BUILD_FILENAME,
        role="engine_build",
        root=root,
        kind="build",
        numeric_format="json",
    )
    manifest = EngineManifest.create(
        contract=contract,
        runtime=validator_runtime,
        build_settings=settings,
        tool_sources=_source_revisions(root, build, source_revision),
        sources=sources,
        engines=engines,
        build=build_record,
        validation=validation,
        vae_weights_path=vae_weights,
        vae_weights_sha256=vae_weights_sha256,
    )
    manifest.write(root / MANIFEST_FILENAME)
    return {**payload, "manifest": asdict(manifest)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--embodiment", required=True)
    parser.add_argument("--source-revision")
    args = parser.parse_args()
    payload = validate_bundle(
        artifact_root=args.artifact_root,
        checkpoint=args.checkpoint,
        embodiment=args.embodiment,
        source_revision=args.source_revision,
    )
    print(json.dumps({"passed": payload["passed"]}, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
