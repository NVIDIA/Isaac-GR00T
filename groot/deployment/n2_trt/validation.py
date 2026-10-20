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
import os
from pathlib import Path
from typing import Any

import torch

from groot.deployment.n2_trt.artifacts import (
    resolve_checkpoint_vae_weights,
    resolve_recorded_vae_weights,
)
from groot.deployment.n2_trt.contract import N2TensorRTContract
from groot.deployment.n2_trt.gate_evidence import (
    GATE_EVIDENCE_FILENAME,
    GateEvidence,
    GateEvidenceError,
    gate_evidence_policy_for,
)
from groot.deployment.n2_trt.golden import (
    EQUIVALENCE_POLICY,
    GOLDEN_OUTPUTS_MANIFEST,
    GOLDEN_OUTPUTS_SCHEMA,
    GOLDEN_SIDECAR_FILENAME,
    GoldenRecord,
    checkpoint_identity,
    checkpoint_identity_matches,
    entry_distances,
    envelope_provenance_matches,
    judge_entry,
    load_record,
    package_sha256,
    read_golden_sidecar,
    runtime_envelope_identity,
    verify_package_files,
)
from groot.deployment.n2_trt.manifest import (
    ADMISSION_GOLDEN,
    BUILD_FILENAME,
    BUILD_SCHEMA,
    COND_CACHED_ENGINE_ROLE,
    MANIFEST_FILENAME,
    TRANSFORMER_EAGER_ORACLE,
    TRANSFORMER_EXPORT_ACTION_ORACLE,
    TRANSFORMER_EXPORT_JOINT_COND_ORACLE,
    TRANSFORMER_EXPORT_JOINT_ORACLE,
    VAE_EAGER_ORACLE,
    VALIDATION_FILENAME,
    VALIDATION_POLICY,
    VALIDATION_PROFILE_THRESHOLDS,
    ArtifactRecord,
    BuildSettings,
    EngineManifest,
    RuntimeIdentity,
    SourceRevision,
    TensorSpec,
    canonical_sha256,
    checkpoint_file_hashes,
    gating_parity_gates,
    golden_validation_policy_for,
    normalize_build_settings_payload,
    required_engine_roles,
    required_parity_gates,
    sha256_file,
    transformer_export_action_oracle,
    validation_policy_for,
    validation_sample_paths,
)
from groot.deployment.n2_trt.runtime import N2TensorTrtRunner
from groot.deployment.n2_trt.tools.inspect_engines import (
    EngineInspectionError,
    census_check,
    require_current_sidecar,
)

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


def _runner_inputs(
    runner: N2TensorTrtRunner,
    sample: dict[str, Any],
    *,
    skip: tuple[str, ...] = (),
) -> dict[str, torch.Tensor]:
    missing = set(runner.inputs) - set(sample)
    if missing:
        raise KeyError(f"validation sample is missing engine inputs: {sorted(missing)}")
    return {
        name: sample[name].to(device="cuda", dtype=metadata["dtype"]).contiguous()
        for name, metadata in runner.inputs.items()
        if name not in skip
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
    keep = modality_ids != 3
    valid_mask = sample.get("token_valid_mask")
    if valid_mask is not None:
        keep &= valid_mask[0, : keep.numel()]
    if coarse:
        keep = keep.index_select(0, sample["coarse_token_idx"].long())
    return torch.nonzero(keep).flatten()


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
    # These reference caches are replaced by joint-engine outputs below.
    inputs = _runner_inputs(
        action_runner, action_sample, skip=tuple(name for name, _, _ in mappings)
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


def _conditioning_query_positions(sample: dict[str, Any]) -> torch.Tensor:
    """Positions the conditioning-cached engine runs as queries: every non-frozen token."""
    keep = torch.ones(int(sample["modality_ids"].shape[1]), dtype=torch.bool)
    keep[sample["cond_index"].long()] = False
    return torch.nonzero(keep).flatten()


def _reduce_conditioning_production_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Bind a full-layout eager capture of a frozen-prefix joint step to the engine's ABI.

    Production captures keep the model's full layout; the engine sees only the query
    tokens, takes the frozen prefix as inputs, and returns the query tokens' fresh K/V.
    """
    for name in ("cond_index", "cond_key", "cond_value", "kv_cache_key", "kv_cache_value"):
        if name not in sample:
            raise KeyError(f"conditioning-cached production sample is missing {name!r}")
    query = _conditioning_query_positions(sample)
    return {
        **sample,
        "latents": sample["latents"].index_select(1, query),
        "timesteps": sample["timesteps"].index_select(1, query),
        "rope_cos": sample["rope_cos"].index_select(0, query),
        "rope_sin": sample["rope_sin"].index_select(0, query),
        "modality_ids": sample["modality_ids"].index_select(1, query),
        "output": sample["output"].index_select(1, query),
        "kv_key": sample["kv_cache_key"].index_select(3, query),
        "kv_value": sample["kv_cache_value"].index_select(3, query),
    }


def _joint_cond_gates(
    engine: Path,
    sample_path: Path,
    *,
    expected_oracle: str = TRANSFORMER_EXPORT_JOINT_COND_ORACLE,
    runner: N2TensorTrtRunner | None = None,
) -> tuple[dict[str, Any], dict[str, Any], N2TensorTrtRunner, dict[str, Any]]:
    sample = _load_sample(sample_path)
    _require_oracle(sample, expected_oracle)
    if expected_oracle == TRANSFORMER_EAGER_ORACLE:
        sample = _reduce_conditioning_production_sample(sample)
    runner = runner or N2TensorTrtRunner(str(engine))
    _run(runner, _runner_inputs(runner, sample))
    hidden = _compare(runner.outputs["output"].clone(), sample["output"], profile="joint_hidden")
    components = {
        name: _compare(runner.outputs[name].clone(), sample[name], profile="bf16")
        for name in ("kv_key", "kv_value")
    }
    kv_diagnostics = {
        "fresh": {
            "passed": all(record["passed"] for record in components.values()),
            "components": components,
        }
    }
    return hidden, kv_diagnostics, runner, sample


def _composed_cond_gate(
    *,
    joint_runner: N2TensorTrtRunner,
    joint_sample: dict[str, Any],
    cond_runner: N2TensorTrtRunner,
    cond_sample: dict[str, Any],
) -> dict[str, Any]:
    """Run the conditioning-cached engine on the joint engine's own conditioning K/V."""
    inputs = _runner_inputs(cond_runner, cond_sample)
    cond_index = cond_sample["cond_index"].long()
    for cond_name, joint_name, native_name in (
        ("cond_key", "output2", "kv_cache_key"),
        ("cond_value", "output3", "kv_cache_value"),
    ):
        joint_sample_name = joint_name if joint_name in joint_sample else native_name
        source = joint_sample[joint_sample_name].index_select(3, cond_index)
        if not torch.equal(source, cond_sample[cond_name]):
            raise RuntimeError(
                f"joint/conditioning-cached samples are not one matched prefix pair: {cond_name}"
            )
        joint_output = joint_runner.outputs[joint_name]
        inputs[cond_name] = joint_output.index_select(
            3, cond_index.to(device=joint_output.device)
        ).clone()
    _run(cond_runner, inputs)
    return _compare(
        cond_runner.outputs["output"].clone(), cond_sample["output"], profile="joint_hidden"
    )


def _composed_action_gate_or_diagnostic(*, gating: bool, **kwargs: Any) -> dict[str, Any]:
    """Run the composed gate; its sample-pairing precondition aborts only when the gate gates.

    A mixed bundle exports its BF16 action graph (and that graph's export sample) before
    calibration from the BF16 joint's KV output, and its quantized joint graph after, so the
    two EXPORT samples are not one matched pair by construction. Over a quantized joint the
    composed gate is a diagnostic, and the receipt records it as unavailable instead of
    aborting the admission; a BF16 contract keeps the hard failure.
    """
    try:
        return _composed_action_gate(**kwargs)
    except RuntimeError as error:
        if gating:
            raise
        return {
            "shape": None,
            "actual_dtype": None,
            "expected_dtype": None,
            "dtype_match": None,
            "actual_finite": None,
            "expected_finite": None,
            "finite": None,
            "max_abs": None,
            "relative_rmse": None,
            "cosine": None,
            "profile": "composed_action",
            "thresholds": PROFILES["composed_action"],
            "gating_metrics": VALIDATION_POLICY["gating_metrics"]["composed_action"],
            "diagnostic_metrics": ["max_abs"],
            "unavailable": str(error),
            "passed": False,
        }


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
        resolved_members = {}
        for relative_path, expected in expected_members.items():
            path = (root / relative_path).resolve(strict=True)
            try:
                relative = path.relative_to(root)
            except ValueError as error:
                raise ValueError(f"build receipt path escapes artifact root: {path}") from error
            resolved_members[str(relative)] = expected
        engine_path = (root / record["engine"]).resolve(strict=True)

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
        actual_members = {
            member.relative_path: (member.sha256, member.size_bytes)
            for member in sources[-1].members
        }
        for relative_path, expected in resolved_members.items():
            if actual_members[relative_path] != expected:
                raise RuntimeError(f"ONNX changed after TensorRT build: {relative_path}")
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
        if (
            engines[-1].size_bytes != record["engine_size_bytes"]
            or engines[-1].sha256 != record["engine_sha256"]
        ):
            raise RuntimeError(f"engine changed after TensorRT build: {record['engine']}")
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
    # Build/validation provenance still belongs to one GPU; only serving is portable.
    if build_runtime.gpu_uuid != validator_runtime.gpu_uuid:
        runtime_mismatch["gpu_uuid"] = {
            "expected": build_runtime.gpu_uuid,
            "actual": validator_runtime.gpu_uuid,
        }
    if runtime_mismatch:
        raise RuntimeError(
            "N2 TensorRT validation must run on the build GPU and software stack: "
            + json.dumps(runtime_mismatch, sort_keys=True, default=str)
        )
    return validator_runtime


def _cpu_outputs(runner: N2TensorTrtRunner) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone().cpu() for name, tensor in runner.outputs.items()}


def _action_inputs_from_kv(
    action_runner: N2TensorTrtRunner,
    action_sample: dict[str, Any],
    key: torch.Tensor,
    value: torch.Tensor,
    static: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """The action engine's inputs with its cached K/V taken from a joint step's full layout."""
    inputs = _runner_inputs(action_runner, action_sample)
    if "kv_cache_key" not in action_runner.inputs:
        raise RuntimeError("golden equivalence covers dense (non-token-merge) action engines only")
    inputs["kv_cache_key"] = key.index_select(3, static.to(device=key.device)).clone()
    inputs["kv_cache_value"] = value.index_select(3, static.to(device=value.device)).clone()
    return inputs


def golden_entry_outputs(
    engine_paths: dict[str, Path], sample_paths: dict[str, Path], *, cond_kv_freeze: bool
) -> dict[str, dict[str, torch.Tensor]]:
    """Run every role and composition of a bundle on the package's export samples.

    The same function produces a package's canonical outputs when it is published and a
    rebuilt bundle's outputs when it is admitted, so equivalence compares like with like:
    each engine on its export-source sample, the joint's K/V fed into the action engine
    (``joint_to_action``) and, for a conditioning-cached bundle, the joint's conditioning K/V
    fed into the conditioning-cached engine whose fresh K/V rebuild the full layout the action
    engine then consumes (``joint_cond_to_action``) -- the served step order.
    """
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    joint_sample = _load_sample(sample_paths["transformer_joint_export"])
    joint_runner = N2TensorTrtRunner(str(engine_paths["transformer_joint"]))
    _run(joint_runner, _runner_inputs(joint_runner, joint_sample))
    outputs["transformer_joint"] = _cpu_outputs(joint_runner)
    joint_key = joint_runner.outputs["output2"].clone()
    joint_value = joint_runner.outputs["output3"].clone()
    action_sample = _load_sample(sample_paths["transformer_action_export"], output_name="output1")
    action_runner = N2TensorTrtRunner(str(engine_paths["transformer_action_cache"]))
    _run(action_runner, _runner_inputs(action_runner, action_sample))
    outputs["transformer_action_cache"] = _cpu_outputs(action_runner)
    static = _static_indices(joint_sample, coarse=False)
    _run(
        action_runner,
        _action_inputs_from_kv(action_runner, action_sample, joint_key, joint_value, static),
    )
    outputs["joint_to_action"] = _cpu_outputs(action_runner)
    if cond_kv_freeze:
        cond_sample = _load_sample(sample_paths["transformer_joint_condcached_export"])
        cond_runner = N2TensorTrtRunner(str(engine_paths[COND_CACHED_ENGINE_ROLE]))
        cond_inputs = _runner_inputs(cond_runner, cond_sample)
        frozen = cond_sample["cond_index"].long()
        for cond_name, source in (("cond_key", joint_key), ("cond_value", joint_value)):
            cond_inputs[cond_name] = source.index_select(3, frozen.to(device=source.device)).clone()
        _run(cond_runner, cond_inputs)
        outputs[COND_CACHED_ENGINE_ROLE] = _cpu_outputs(cond_runner)
        total_tokens = int(joint_sample["modality_ids"].shape[1])
        keep = torch.ones(total_tokens, dtype=torch.bool)
        keep[frozen] = False
        query = torch.nonzero(keep).flatten().to(device=joint_key.device)
        full_key = joint_key.clone()
        full_value = joint_value.clone()
        full_key.index_copy_(3, query, cond_runner.outputs["kv_key"].to(dtype=full_key.dtype))
        full_value.index_copy_(3, query, cond_runner.outputs["kv_value"].to(dtype=full_value.dtype))
        _run(
            action_runner,
            _action_inputs_from_kv(action_runner, action_sample, full_key, full_value, static),
        )
        outputs["joint_cond_to_action"] = _cpu_outputs(action_runner)
    vae_sample = _load_sample(sample_paths["vae_eager"])
    vae_runner = N2TensorTrtRunner(str(engine_paths["vae_encoder"]))
    _run(vae_runner, _runner_inputs(vae_runner, vae_sample))
    outputs["vae_encoder"] = _cpu_outputs(vae_runner)
    return outputs


def _golden_route_inputs(
    root: Path, *, checkpoint: Path, build: dict[str, Any], settings_payload: dict[str, Any]
) -> dict[str, Any] | None:
    """The golden route's inputs when the root carries a golden sidecar (None otherwise).

    Everything that needs no engine execution is verified here, fail-closed: exactly one
    admission route present, the sidecar and its record (by package digest), the export-half
    files and every role's ONNX and sample digests, the checkpoint identity, the build settings.
    """
    if not (root / GOLDEN_SIDECAR_FILENAME).is_file():
        return None
    if (root / GATE_EVIDENCE_FILENAME).is_file():
        raise RuntimeError(
            f"{root} carries both {GOLDEN_SIDECAR_FILENAME} and {GATE_EVIDENCE_FILENAME}: a golden "
            "package root is admitted by its record alone and never receives gate evidence"
        )
    sidecar = read_golden_sidecar(root)
    record = load_record(sidecar["package_id"])
    digest = package_sha256(record)
    if digest != sidecar["package_sha256"]:
        raise RuntimeError(
            "golden sidecar package digest does not match the record "
            f"(sidecar {sidecar['package_sha256'][:12]}, record {digest[:12]})"
        )
    files = verify_package_files(root, record.package, allow_extra=True)
    if not files["passed"]:
        raise RuntimeError(
            "golden package files do not match the record: " + "; ".join(files["failures"][:8])
        )
    roles = record.roles
    for artifact in build.get("artifacts") or []:
        role = artifact.get("role")
        if role not in roles:
            raise RuntimeError(f"golden record {record.package_id!r} has no role {role!r}")
        onnx_digest = sha256_file(root / artifact["onnx"])
        if onnx_digest != roles[role]["onnx_sha256"]:
            raise RuntimeError(f"{role}: ONNX digest does not match the golden record")
        for relative, expected in roles[role]["sample_sha256s"].items():
            if sha256_file(root / relative) != expected:
                raise RuntimeError(f"{role}: sample {relative} does not match the golden record")
    differences = checkpoint_identity_matches(
        record.package["checkpoint"], checkpoint_identity(checkpoint)
    )
    if differences:
        raise RuntimeError(
            "checkpoint identity does not match the golden record: " + "; ".join(differences)
        )
    if settings_payload != record.package["build_settings"]:
        raise RuntimeError(
            "build settings do not match the golden record's build_settings "
            f"(built {settings_payload}, record {record.package['build_settings']})"
        )
    return {
        "sidecar": sidecar,
        "sidecar_sha256": sha256_file(root / GOLDEN_SIDECAR_FILENAME),
        "record": record,
        "package_sha256": digest,
        "files_checked": files["checked"],
    }


def _golden_admission(
    root: Path,
    golden: dict[str, Any],
    *,
    contract: N2TensorRTContract,
    engines: tuple[ArtifactRecord, ...],
    engine_paths: dict[str, Path],
    sample_paths: dict[str, Path],
    validator_runtime: RuntimeIdentity,
    cond_kv_freeze: bool,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """The engine-dependent half of the golden route: platform row, census, equivalence."""
    record: GoldenRecord = golden["record"]
    sidecar = golden["sidecar"]
    gates = record.gate_policy.effective()
    failures: list[str] = []
    identity = runtime_envelope_identity(validator_runtime)
    row = None
    if sidecar["mode"] == "serve":
        row = record.admission_row_for(identity)
        if row is None:
            failures.append(
                f"no admission row of {record.package_id!r} covers this runtime envelope {identity}"
            )
        elif row["row_sha256"] != sidecar["platform_row_sha256"]:
            failures.append(
                "the sidecar's platform row digest does not match the record's row for this envelope"
            )
            row = None
    inspection: dict[str, Any] = {}
    for engine in engines:
        role = engine.role
        try:
            sidecar_record = require_current_sidecar(root / engine.relative_path)
        except EngineInspectionError as error:
            failures.append(f"{role}: {error}")
            inspection[role] = {"passed": False, "failures": [str(error)]}
            continue
        check = census_check(
            sidecar_record["layer_output_dtype_counts"],
            numeric_format=contract.engine_precisions[role],
            max_fp32_layer_outputs=gates["max_fp32_layer_outputs"],
        )
        inspection[role] = {"engine_sha256": sidecar_record["engine_sha256"], **check}
        failures.extend(f"{role}: {reason}" for reason in check["failures"])
    equivalence: dict[str, Any] = {
        "policy": EQUIVALENCE_POLICY,
        "gating": row is not None,
        "entries": {},
    }
    oracle_path = root / GOLDEN_OUTPUTS_MANIFEST
    if not oracle_path.is_file():
        failures.append(f"the package carries no {GOLDEN_OUTPUTS_MANIFEST}")
    elif sha256_file(oracle_path) != record.package["oracle_manifest_sha256"]:
        failures.append("the canonical output manifest does not match the golden record")
    else:
        oracle = json.loads(oracle_path.read_text())
        entries = oracle.get("entries") if isinstance(oracle, dict) else None
        if oracle.get("schema") != GOLDEN_OUTPUTS_SCHEMA or not isinstance(entries, dict):
            failures.append(f"unsupported canonical output manifest: {oracle.get('schema')!r}")
        else:
            outputs = golden_entry_outputs(
                engine_paths, sample_paths, cond_kv_freeze=cond_kv_freeze
            )
            for name, entry in entries.items():
                if name not in outputs:
                    failures.append(f"equivalence {name}: the bundle produces no such entry")
                    equivalence["entries"][name] = {"passed": False}
                    continue
                expected = torch.load(root / entry["output"], map_location="cpu", weights_only=True)
                dtypes = {key: spec["dtype"] for key, spec in (entry.get("tensors") or {}).items()}
                distances = entry_distances(outputs[name], expected, oracle_dtypes=dtypes)
                result: dict[str, Any] = {
                    key: distances[key] for key in ("rel_rmse", "rel_maxabs", "cosine")
                }
                if row is not None:
                    bounds = (row.get("entries") or {}).get(name)
                    if bounds is None:
                        judged = {
                            "passed": False,
                            "failures": ["the admitting row carries no bounds for this entry"],
                        }
                    else:
                        judged = judge_entry(distances, bounds)
                        result["bounds"] = {
                            key: bounds[key] for key in ("envelope", "maxabs_bound", "cosine_floor")
                        }
                    result["passed"] = judged["passed"]
                    result["failures"] = judged["failures"]
                else:
                    # Certify mode: distances are diagnostics; only a structural mismatch fails.
                    result["passed"] = None
                    result["failures"] = list(distances["failures"])
                if result["failures"]:
                    failures.append(f"equivalence {name}: " + "; ".join(result["failures"]))
                equivalence["entries"][name] = result
    schema, policy = golden_validation_policy_for(contract, record, row)
    payload = {
        "package_id": record.package_id,
        "package_sha256": golden["package_sha256"],
        "mode": sidecar["mode"],
        "platform_row_sha256": sidecar.get("platform_row_sha256"),
        "sidecar_sha256": golden["sidecar_sha256"],
        "certification_level": record.certification.level,
        "runtime_envelope": identity,
        "speed_certified_on_this_stack": bool(row)
        and envelope_provenance_matches(row["envelope"], identity),
        "files_verified": golden["files_checked"],
        "checkpoint_identity": "match",
        "build_settings": "match",
        "inspection": inspection,
        "equivalence": equivalence,
        "failures": failures,
        "passed": not failures,
    }
    return schema, policy, payload


def validate_bundle(
    *,
    artifact_root: Path,
    checkpoint: Path,
    embodiment: str,
    source_revision: str | None = None,
    num_video_steps: int = 2,
    num_action_steps: int = 4,
) -> dict[str, Any]:
    root = artifact_root.expanduser().resolve(strict=True)
    build = json.loads((root / BUILD_FILENAME).read_text())
    if build.get("schema") != BUILD_SCHEMA:
        raise ValueError(f"unsupported build receipt schema: {build.get('schema')!r}")
    # The transformer precisions are declared by the build receipt (per role); the
    # contract is resolved for them so the manifest describes the bundle as built.
    receipt_formats = {
        record.get("role"): record.get("numeric_format") for record in build.get("artifacts") or []
    }
    settings_payload = normalize_build_settings_payload(build.get("settings"))
    cond_kv_freeze = bool(settings_payload["cond_kv_freeze"])
    expected_roles = required_engine_roles(cond_kv_freeze)
    missing_roles = sorted(expected_roles - set(receipt_formats))
    if missing_roles:
        raise ValueError(f"build receipt omits engine roles {missing_roles}")
    unexpected_roles = sorted(set(receipt_formats) - expected_roles)
    if unexpected_roles:
        raise ValueError(f"build receipt carries undeclared engine roles {unexpected_roles}")
    contract = N2TensorRTContract.from_checkpoint(
        checkpoint,
        embodiment=embodiment,
        video_denoising_steps=num_video_steps,
        action_denoising_steps=num_action_steps,
        joint_denoise=True,
        conditional_kv_cache=True,
        transformer_precision=receipt_formats["transformer_joint"],
        action_cache_precision=receipt_formats["transformer_action_cache"],
        vae_precision=receipt_formats["vae_encoder"],
        cond_kv_freeze=cond_kv_freeze,
    )
    if (
        cond_kv_freeze
        and receipt_formats[COND_CACHED_ENGINE_ROLE] != receipt_formats["transformer_joint"]
    ):
        raise ValueError(
            "build receipt conditioning-cached engine precision "
            f"{receipt_formats[COND_CACHED_ENGINE_ROLE]!r} != joint "
            f"{receipt_formats['transformer_joint']!r}"
        )
    golden = _golden_route_inputs(
        root, checkpoint=checkpoint, build=build, settings_payload=settings_payload
    )
    action_export_oracle = transformer_export_action_oracle(settings_payload["action_context_mode"])
    sample_paths = {
        role: (root / relative_path).resolve(strict=True)
        for role, relative_path in validation_sample_paths(cond_kv_freeze).items()
    }
    expected_samples = build.get("validation_samples")
    actual_samples = {role: sha256_file(path) for role, path in sample_paths.items()}
    if expected_samples != actual_samples:
        raise RuntimeError("validation samples changed after the TensorRT engine build")
    vae_export = json.loads((root / "n2_vae_export_manifest.json").read_text())
    exported_vae_weights = vae_export.get("vae_weights")
    if not isinstance(exported_vae_weights, str) or not exported_vae_weights:
        raise RuntimeError("VAE export manifest omits the resolved weights path")
    exported_path = Path(exported_vae_weights).expanduser()
    if (
        golden is not None
        and not os.environ.get("GROOT_N2_MODEL_DEPS_ROOT")
        and exported_path.is_absolute()
        and not exported_path.is_file()
    ):
        # A package exported on another host names that host's weights path: resolve the
        # weights from the local checkpoint and bind them by digest instead.
        vae_weights = resolve_checkpoint_vae_weights(checkpoint)
    else:
        # Explicit package mode stays offline, including for golden bundles whose
        # export manifest still names another host's absolute cache path.
        vae_weights = resolve_recorded_vae_weights(checkpoint, exported_vae_weights)
    vae_weights_sha256 = sha256_file(vae_weights)
    if vae_export.get("vae_weights_sha256") != vae_weights_sha256:
        raise RuntimeError("VAE weights changed after export")
    if golden is not None and golden["record"].package["vae_weights_sha256"] != vae_weights_sha256:
        raise RuntimeError("VAE weights do not match the golden record")
    validator_runtime = _validated_runtime_identity(build)
    gating = gating_parity_gates(contract)
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
        engine_paths["transformer_action_cache"],
        action_export_path,
        expected_oracle=action_export_oracle,
    )
    composed_export = _composed_action_gate_or_diagnostic(
        gating="transformer_composed_action_export" in gating,
        joint_runner=joint_runner,
        joint_sample=joint_export_sample,
        action_runner=action_runner,
        action_sample=action_export_sample,
    )
    cond_gates: dict[str, Any] = {}
    cond_diagnostics: dict[str, Any] = {}
    if cond_kv_freeze:
        # Both cond gates that pair with the joint engine run while its outputs still hold
        # the export sample's result.
        cond_engine = engine_paths[COND_CACHED_ENGINE_ROLE]
        cond_export, cond_export_kv, cond_runner, cond_export_sample = _joint_cond_gates(
            cond_engine, sample_paths["transformer_joint_condcached_export"]
        )
        cond_gates["transformer_joint_condcached_export"] = cond_export
        cond_gates["transformer_composed_condcached_export"] = _composed_cond_gate(
            joint_runner=joint_runner,
            joint_sample=joint_export_sample,
            cond_runner=cond_runner,
            cond_sample=cond_export_sample,
        )
        cond_production, cond_production_kv, _, _ = _joint_cond_gates(
            cond_engine,
            sample_paths["transformer_joint_condcached_production"],
            expected_oracle=TRANSFORMER_EAGER_ORACLE,
            runner=cond_runner,
        )
        cond_gates["transformer_joint_condcached_production"] = cond_production
        cond_diagnostics = {
            "transformer_condcached_kv_export": cond_export_kv,
            "transformer_condcached_kv_production": cond_production_kv,
        }
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
    composed_production = _composed_action_gate_or_diagnostic(
        gating="transformer_composed_action_production" in gating,
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
        **cond_gates,
    }
    checkpoint_files = checkpoint_file_hashes(contract.checkpoint)
    checkpoint_sha256 = canonical_sha256(checkpoint_files)
    contract_sha256 = canonical_sha256(contract.to_dict())
    if golden is not None:
        schema, policy, golden_payload = _golden_admission(
            root,
            golden,
            contract=contract,
            engines=engines,
            engine_paths=engine_paths,
            sample_paths=sample_paths,
            validator_runtime=validator_runtime,
            cond_kv_freeze=cond_kv_freeze,
        )
    else:
        schema, policy = validation_policy_for(contract)
    payload = {
        "schema": schema,
        "validation_policy": policy,
        "validation_policy_sha256": canonical_sha256(policy),
        "contract_sha256": contract_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "vae_weights_path": str(vae_weights),
        "vae_weights_sha256": vae_weights_sha256,
        "artifacts": {record.role: record.sha256 for record in (*sources, *engines)},
        "oracles": actual_samples,
        "gates": gates,
        "diagnostics": {
            "transformer_kv_export": joint_export_kv,
            "transformer_kv_production": joint_production_kv,
            **cond_diagnostics,
        },
        "validator_runtime": asdict(validator_runtime),
        "passed": all(gates[name]["passed"] for name in gating),
    }
    if golden is not None:
        # Quantized roles are admitted by equivalence against the golden package's canonical
        # outputs within the admitting platform row (serve) or recorded as diagnostics (certify).
        payload["gating_parity_gates"] = sorted(gating)
        payload["diagnostic_parity_gates"] = sorted(required_parity_gates(cond_kv_freeze) - gating)
        payload["golden"] = golden_payload
        payload["passed"] = payload["passed"] and golden_payload["passed"]
    elif contract.is_quantized:
        # Quantized roles are admitted by the loop's gate evidence, recomputed here
        # from its stored metrics and bound to these exact engines and this checkpoint.
        evidence_path = root / GATE_EVIDENCE_FILENAME
        if not evidence_path.is_file():
            raise FileNotFoundError(
                f"quantized bundle admission requires {GATE_EVIDENCE_FILENAME} in {root}"
            )
        evidence_schema, evidence_policy_sha256 = gate_evidence_policy_for(cond_kv_freeze)
        evidence = GateEvidence.read(evidence_path)
        if evidence.schema != evidence_schema:
            raise GateEvidenceError(
                f"gate evidence schema {evidence.schema!r} does not match the bundle contract "
                f"(cond_kv_freeze={cond_kv_freeze}: expected {evidence_schema!r})"
            )
        recomputed = evidence.recompute(
            engine_sha256={record.role: record.sha256 for record in engines},
            checkpoint_sha256=checkpoint_sha256,
            contract_sha256=contract_sha256,
        )
        payload["gating_parity_gates"] = sorted(gating)
        payload["diagnostic_parity_gates"] = sorted(required_parity_gates(cond_kv_freeze) - gating)
        payload["gate_evidence"] = {
            "relative_path": GATE_EVIDENCE_FILENAME,
            "sha256": sha256_file(evidence_path),
            "policy_sha256": evidence_policy_sha256,
            "recomputed": recomputed,
        }
        payload["passed"] = payload["passed"] and recomputed["passed"]
    receipt_path = root / VALIDATION_FILENAME
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if not payload["passed"]:
        raise ParityValidationError(f"N2 TensorRT admission failed; receipt={receipt_path}")

    settings = BuildSettings(**settings_payload)
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
        **(
            {}
            if golden is None
            else {
                "admission": ADMISSION_GOLDEN,
                "certification_level": golden["record"].certification.level,
                "package_id": golden["record"].package_id,
                "golden_sidecar_sha256": golden["sidecar_sha256"],
            }
        ),
    )
    manifest.write(root / MANIFEST_FILENAME)
    return {**payload, "manifest": asdict(manifest)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--embodiment", required=True)
    parser.add_argument("--source-revision")
    parser.add_argument("--num-video-steps", type=int, default=2)
    parser.add_argument("--num-action-steps", type=int, default=4)
    args = parser.parse_args()
    payload = validate_bundle(
        artifact_root=args.artifact_root,
        checkpoint=args.checkpoint,
        embodiment=args.embodiment,
        source_revision=args.source_revision,
        num_video_steps=args.num_video_steps,
        num_action_steps=args.num_action_steps,
    )
    print(json.dumps({"passed": payload["passed"]}, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
