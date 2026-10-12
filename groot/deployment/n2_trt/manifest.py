# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Versioned, content-addressed N2 TensorRT engine manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

from groot.core.utils.storage.checkpoint import checkpoint_source_path
from groot.deployment.n2_trt.artifacts import resolve_checkpoint_vae_weights
from groot.deployment.n2_trt.contract import N2TensorRTContract

MANIFEST_SCHEMA = "groot_n2_tensorrt_engine_manifest_v4"
MANIFEST_FILENAME = "n2_trt_manifest.json"
BUILD_FILENAME = "n2_trt_build.json"
BUILD_SCHEMA = "groot_n2_trt_build_v4"
VALIDATION_FILENAME = "n2_trt_validation.json"
REQUIRED_SERVING_RECEIPTS = (
    BUILD_FILENAME,
    VALIDATION_FILENAME,
    MANIFEST_FILENAME,
)
VALIDATION_SCHEMA = "groot_n2_tensorrt_engine_parity_v8"
TRANSFORMER_EAGER_ORACLE = "production-eager-native-attention-v2"
TRANSFORMER_EXPORT_JOINT_ORACLE = "export-segmented-sdpa-joint-v1"
TRANSFORMER_EXPORT_ACTION_ORACLE = "export-sdpa-action-cache-v1"
VAE_EAGER_ORACLE = "unchanged-eager-chunked-encoder-v1"
CHECKPOINT_METADATA = ("config.json", "processor_config.json", "statistics.json")
REQUIRED_ENGINE_ROLES = frozenset({"transformer_joint", "transformer_action_cache", "vae_encoder"})
REQUIRED_SOURCE_ROLES = frozenset(f"{role}_onnx" for role in REQUIRED_ENGINE_ROLES)
REQUIRED_TOOL_COMPONENTS = frozenset({"exporter", "builder", "validator"})
REQUIRED_PARITY_GATES = frozenset(
    {
        "transformer_joint_export",
        "transformer_action_cache_export",
        "transformer_composed_action_export",
        "transformer_joint_production",
        "transformer_action_cache_production",
        "transformer_composed_action_production",
        "vae_encoder",
    }
)
VALIDATION_SAMPLE_PATHS = {
    "transformer_joint_export": "samples/joint_export_source.pt",
    "transformer_action_export": "samples/action_export_source.pt",
    "transformer_joint_production": "samples/joint_full_capture.pt",
    "transformer_action_production": "samples/action_cached.pt",
    "vae_eager": "sample.pt",
}
VALIDATION_PROFILE_THRESHOLDS = {
    "bf16": {"max_relative_rmse": 0.02, "min_cosine": None},
    "joint_hidden": {"max_relative_rmse": 0.1, "min_cosine": 0.995},
    "cached_action": {"max_relative_rmse": 0.025, "min_cosine": 0.9995},
    # The composed gate contains the accepted numerical error of both BF16
    # transformer engines. Target-local G1 calibration for this policy is
    # retained in groot/core/docs/n2/TENSORRT_RUNTIME.md.
    "composed_action": {"max_relative_rmse": 0.03, "min_cosine": 0.9995},
}
VALIDATION_GATE_PROFILES = {
    "transformer_joint_export": "joint_hidden",
    "transformer_action_cache_export": "cached_action",
    "transformer_composed_action_export": "composed_action",
    "transformer_joint_production": "joint_hidden",
    "transformer_action_cache_production": "cached_action",
    "transformer_composed_action_production": "composed_action",
    "vae_encoder": "bf16",
}
RUNTIME_COMPATIBILITY_FIELDS = (
    "machine",
    "gpu_name",
    "gpu_uuid",
    "compute_capability",
    "driver",
    "cuda",
    "tensorrt",
    "torch",
    "python",
    "platform",
)


def required_serving_files(manifest: EngineManifest) -> tuple[str, ...]:
    """Return manifest-owned engine paths plus required serving receipts."""
    return tuple(record.relative_path for record in manifest.engines) + REQUIRED_SERVING_RECEIPTS


def sha256_file(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


VALIDATION_POLICY = {
    "schema": VALIDATION_SCHEMA,
    "gate_profiles": VALIDATION_GATE_PROFILES,
    "profile_thresholds": VALIDATION_PROFILE_THRESHOLDS,
    "gating_metrics": {
        profile: [
            "shape",
            "dtype",
            "finite",
            "relative_rmse",
            *(["cosine"] if thresholds["min_cosine"] is not None else []),
        ]
        for profile, thresholds in VALIDATION_PROFILE_THRESHOLDS.items()
    },
}
VALIDATION_POLICY_SHA256 = canonical_sha256(VALIDATION_POLICY)


def checkpoint_file_hashes(checkpoint: str | Path) -> dict[str, str]:
    """Hash required metadata and every index-referenced inference weight shard."""
    root = checkpoint_source_path(Path(checkpoint).expanduser()).resolve(strict=True)
    paths: set[Path] = set()
    for name in CHECKPOINT_METADATA:
        candidate = root / name
        if candidate.is_file():
            paths.add(candidate)
    missing_metadata = [name for name in CHECKPOINT_METADATA if not (root / name).is_file()]
    if missing_metadata:
        raise ValueError(f"checkpoint identity is missing required metadata: {missing_metadata}")

    indexes = sorted(root.glob("*.index.json"))
    for index in indexes:
        payload = json.loads(index.read_text())
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            continue
        paths.add(index)
        for relative_name in set(weight_map.values()):
            if not isinstance(relative_name, str):
                raise TypeError(f"checkpoint index contains a non-string shard: {index}")
            relative = Path(relative_name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"checkpoint index shard escapes checkpoint: {relative_name}")
            shard = root / relative
            if not shard.is_file():
                raise FileNotFoundError(f"checkpoint weight shard is missing: {shard}")
            paths.add(shard)

    if not any(path.name.endswith(".index.json") for path in paths):
        paths.update(path for path in root.glob("*.safetensors") if path.is_file())
        pytorch_bin = root / "pytorch_model.bin"
        if pytorch_bin.is_file():
            paths.add(pytorch_bin)
    weight_paths = [
        path
        for path in paths
        if path.suffix in {".safetensors", ".bin"} or path.name.endswith(".index.json")
    ]
    if not weight_paths:
        raise ValueError("checkpoint identity found no inference weight files or weight index")
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(paths, key=lambda candidate: str(candidate.relative_to(root)))
    }


@dataclass(frozen=True)
class RuntimeIdentity:
    hostname: str
    machine: str
    gpu_name: str
    gpu_uuid: str
    compute_capability: tuple[int, int]
    driver: str
    cuda: str
    tensorrt: str
    torch: str
    python: str
    platform: str

    @property
    def gpu_architecture(self) -> str:
        major, minor = self.compute_capability
        return f"sm{major}{minor}"

    def compatibility_mismatches(self, other: RuntimeIdentity) -> dict[str, dict[str, Any]]:
        """Return target-local runtime fields that differ from ``other``."""
        return {
            name: {"expected": getattr(self, name), "actual": getattr(other, name)}
            for name in RUNTIME_COMPATIBILITY_FIELDS
            if getattr(self, name) != getattr(other, name)
        }

    @classmethod
    def capture(cls, device: int | None = None) -> RuntimeIdentity:
        """Capture the software and GPU identity that owns local plan bytes."""
        try:
            import tensorrt
            import torch
        except ImportError as error:
            raise RuntimeError(
                "TensorRT runtime is not installed; eager fallback is disabled"
            ) from error
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; eager fallback is disabled")
        if device is None:
            device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        gpu_uuid = getattr(properties, "uuid", None) or _nvidia_smi_field(device, "uuid")
        return cls(
            hostname=platform.node(),
            machine=platform.machine(),
            gpu_name=torch.cuda.get_device_name(device),
            gpu_uuid=str(gpu_uuid),
            compute_capability=tuple(torch.cuda.get_device_capability(device)),
            driver=_nvidia_smi_field(device, "driver_version"),
            cuda=str(torch.version.cuda),
            tensorrt=str(tensorrt.__version__),
            torch=str(torch.__version__),
            python=platform.python_version(),
            platform=platform.platform(),
        )


@dataclass(frozen=True)
class TensorSpec:
    """One fixed engine or ONNX tensor ABI entry."""

    name: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not self.name or not self.dtype or any(dimension <= 0 for dimension in self.shape):
            raise ValueError(f"invalid tensor specification: {self}")


@dataclass(frozen=True)
class SourceRevision:
    """Pinned implementation revision used to produce or validate artifacts."""

    component: str
    revision: str
    dirty: bool = False

    @classmethod
    def capture(
        cls,
        component: str,
        source_path: str | Path,
        revision: str | None = None,
    ) -> SourceRevision:
        """Capture trusted Git or installed-package source provenance."""
        source = Path(source_path).resolve(strict=True)
        try:
            root = subprocess.run(
                ["git", "-C", str(source.parent), "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            captured_revision = subprocess.run(
                ["git", "-C", root, "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            dirty = bool(
                subprocess.run(
                    ["git", "-C", root, "status", "--porcelain"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            captured_revision = None
        else:
            if revision is not None and revision.strip() != captured_revision:
                raise RuntimeError(
                    "supplied source revision does not match the current Git checkout: "
                    f"{revision.strip()!r} != {captured_revision!r}"
                )
            return cls(component=component, revision=captured_revision, dirty=dirty)

        if revision is not None:
            supplied_revision = revision.strip()
            if not supplied_revision:
                raise ValueError("source revision must not be empty")
            return cls(component=component, revision=supplied_revision, dirty=False)

        package_revision = _installed_package_revision(source)
        if package_revision is not None:
            return cls(component=component, revision=package_revision, dirty=False)
        raise RuntimeError(
            "source provenance is unavailable; automatic TensorRT builds require either "
            "a Git checkout or an installed non-editable package with an immutable RECORD"
        )


def _installed_package_revision(source: Path) -> str | None:
    """Return a content-addressed identity for a non-editable wheel install."""
    try:
        distribution = metadata.distribution("groot")
        package_root = Path(distribution.locate_file("groot")).resolve(strict=True)
        source.relative_to(package_root)
        direct_url_text = distribution.read_text("direct_url.json")
        if direct_url_text:
            direct_url = json.loads(direct_url_text)
            if not isinstance(direct_url, dict):
                return None
            directory_info = direct_url.get("dir_info", {})
            if isinstance(directory_info, dict) and directory_info.get("editable", False):
                return None
        record = distribution.read_text("RECORD")
    except (FileNotFoundError, json.JSONDecodeError, metadata.PackageNotFoundError, ValueError):
        return None
    if not record:
        return None
    return f"wheel-record-sha256:{hashlib.sha256(record.encode()).hexdigest()}"


@dataclass(frozen=True)
class BuildSettings:
    """Strongly typed TensorRT build settings admitted by this runtime."""

    strongly_typed: bool
    builder_optimization_level: int
    transformer_precision: str
    vae_precision: str
    codec: str

    def validate(self, contract: N2TensorRTContract) -> None:
        if not self.strongly_typed:
            raise ValueError("N2 TensorRT engines must use a strongly typed network")
        if self.builder_optimization_level not in range(0, 6):
            raise ValueError("TensorRT builder optimization level must be between 0 and 5")
        if self.transformer_precision != contract.transformer_precision.value:
            raise ValueError("transformer build precision does not match the runtime contract")
        if self.vae_precision != contract.vae_precision:
            raise ValueError("VAE build precision does not match the runtime contract")
        if self.codec not in {"h264", "mp4v", "none"}:
            raise ValueError(f"unsupported N2 TensorRT serving codec: {self.codec!r}")


@dataclass(frozen=True)
class ArtifactMember:
    """One physical file belonging to a logical runtime artifact."""

    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ArtifactRecord:
    role: str
    relative_path: str
    sha256: str
    size_bytes: int
    kind: str
    numeric_format: str
    inputs: tuple[TensorSpec, ...] = ()
    outputs: tuple[TensorSpec, ...] = ()
    members: tuple[ArtifactMember, ...] = ()

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        role: str,
        root: str | Path,
        kind: str,
        numeric_format: str,
        inputs: tuple[TensorSpec, ...] | list[TensorSpec] = (),
        outputs: tuple[TensorSpec, ...] | list[TensorSpec] = (),
    ) -> ArtifactRecord:
        return cls.from_files(
            path,
            auxiliary_files=(),
            role=role,
            root=root,
            kind=kind,
            numeric_format=numeric_format,
            inputs=inputs,
            outputs=outputs,
        )

    @classmethod
    def from_files(
        cls,
        path: str | Path,
        *,
        auxiliary_files: tuple[str | Path, ...] | list[str | Path],
        role: str,
        root: str | Path,
        kind: str,
        numeric_format: str,
        inputs: tuple[TensorSpec, ...] | list[TensorSpec] = (),
        outputs: tuple[TensorSpec, ...] | list[TensorSpec] = (),
    ) -> ArtifactRecord:
        """Bind a logical artifact and all physical files carrying its bytes."""
        artifact = Path(path).resolve(strict=True)
        artifact_root = Path(root).resolve(strict=True)
        physical_files = (artifact, *(Path(item).resolve(strict=True) for item in auxiliary_files))
        if len(set(physical_files)) != len(physical_files):
            raise ValueError(f"artifact contains duplicate physical files: {physical_files}")
        members = []
        for physical in physical_files:
            try:
                relative = physical.relative_to(artifact_root)
            except ValueError as error:
                raise ValueError(f"artifact is outside manifest root: {physical}") from error
            if not physical.is_file() or physical.stat().st_size <= 0:
                raise ValueError(f"artifact member is missing or empty: {physical}")
            members.append(
                ArtifactMember(
                    relative_path=str(relative),
                    sha256=sha256_file(physical),
                    size_bytes=physical.stat().st_size,
                )
            )
        members = sorted(members, key=lambda member: member.relative_path)
        primary_relative = str(artifact.relative_to(artifact_root))
        aggregate_sha256 = (
            members[0].sha256
            if len(members) == 1
            else canonical_sha256([asdict(member) for member in members])
        )
        return cls(
            role=role,
            relative_path=primary_relative,
            sha256=aggregate_sha256,
            size_bytes=sum(member.size_bytes for member in members),
            kind=kind,
            numeric_format=numeric_format,
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            members=tuple(members),
        )


@dataclass(frozen=True)
class EngineManifest:
    schema: str
    contract: dict[str, Any]
    contract_sha256: str
    checkpoint_files: dict[str, str]
    checkpoint_sha256: str
    vae_weights_path: str
    vae_weights_sha256: str
    runtime: RuntimeIdentity
    build_settings: BuildSettings
    tool_sources: tuple[SourceRevision, ...]
    sources: tuple[ArtifactRecord, ...]
    engines: tuple[ArtifactRecord, ...]
    build: ArtifactRecord
    validation: ArtifactRecord

    @classmethod
    def create(
        cls,
        *,
        contract: N2TensorRTContract,
        runtime: RuntimeIdentity,
        build_settings: BuildSettings,
        tool_sources: tuple[SourceRevision, ...] | list[SourceRevision],
        sources: tuple[ArtifactRecord, ...] | list[ArtifactRecord],
        engines: tuple[ArtifactRecord, ...] | list[ArtifactRecord],
        build: ArtifactRecord,
        validation: ArtifactRecord,
        vae_weights_path: str | Path,
        vae_weights_sha256: str,
    ) -> EngineManifest:
        contract.validate()
        build_settings.validate(contract)
        checkpoint_files = checkpoint_file_hashes(contract.checkpoint)
        source_records = tuple(sources)
        engine_records = tuple(engines)
        tool_records = tuple(tool_sources)
        _require_unique_roles(source_records, "source")
        _require_unique_roles(engine_records, "engine")
        _require_artifact_io(source_records, "source")
        _require_artifact_io(engine_records, "engine")
        source_roles = {record.role for record in source_records}
        if source_roles != REQUIRED_SOURCE_ROLES:
            raise ValueError(
                "source manifest requires exactly "
                f"{sorted(REQUIRED_SOURCE_ROLES)}, got {sorted(source_roles)}"
            )
        tool_components = {record.component for record in tool_records}
        if tool_components != REQUIRED_TOOL_COMPONENTS:
            raise ValueError(
                "manifest requires exact tool revisions "
                f"{sorted(REQUIRED_TOOL_COMPONENTS)}, got {sorted(tool_components)}"
            )
        if any(not record.revision or record.dirty for record in tool_records):
            raise ValueError("manifest tool revisions must be non-empty and from clean sources")
        engine_roles = {record.role for record in engine_records}
        if engine_roles != REQUIRED_ENGINE_ROLES:
            raise ValueError(
                "engine manifest requires exactly "
                f"{sorted(REQUIRED_ENGINE_ROLES)}, got {sorted(engine_roles)}"
            )
        expected_formats = {
            "transformer_joint": contract.transformer_precision.value,
            "transformer_action_cache": contract.transformer_precision.value,
            "vae_encoder": contract.vae_precision,
        }
        format_mismatches = {
            record.role: {
                "actual": record.numeric_format,
                "expected": expected_formats[record.role],
            }
            for record in engine_records
            if record.numeric_format != expected_formats[record.role]
        }
        if format_mismatches:
            raise ValueError(f"engine numeric formats do not match contract: {format_mismatches}")
        source_by_role = {record.role.removesuffix("_onnx"): record for record in source_records}
        schema_mismatches = {
            engine.role: "ONNX and engine input/output schemas differ"
            for engine in engine_records
            if (
                source_by_role[engine.role].inputs != engine.inputs
                or source_by_role[engine.role].outputs != engine.outputs
            )
        }
        if schema_mismatches:
            raise ValueError(f"artifact I/O schemas do not match: {schema_mismatches}")
        if validation.kind != "validation" or validation.role != "engine_validation":
            raise ValueError("manifest validation record must be the engine_validation receipt")
        if build.kind != "build" or build.role != "engine_build":
            raise ValueError("manifest build record must be the engine_build receipt")
        resolved_vae_weights = Path(vae_weights_path).expanduser().resolve(strict=True)
        if not resolved_vae_weights.is_file():
            raise ValueError(f"VAE weights path must identify a file: {resolved_vae_weights}")
        if len(vae_weights_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in vae_weights_sha256
        ):
            raise ValueError("VAE weights digest must be a lowercase SHA-256 hex string")
        if sha256_file(resolved_vae_weights) != vae_weights_sha256:
            raise ValueError("VAE weights digest does not match the resolved file")
        contract_payload = contract.to_dict()
        return cls(
            schema=MANIFEST_SCHEMA,
            contract=contract_payload,
            contract_sha256=canonical_sha256(contract_payload),
            checkpoint_files=checkpoint_files,
            checkpoint_sha256=canonical_sha256(checkpoint_files),
            vae_weights_path=str(resolved_vae_weights),
            vae_weights_sha256=vae_weights_sha256,
            runtime=runtime,
            build_settings=build_settings,
            tool_sources=tool_records,
            sources=source_records,
            engines=engine_records,
            build=build,
            validation=validation,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["runtime"]["compute_capability"] = list(self.runtime.compute_capability)
        payload["runtime"]["gpu_architecture"] = self.runtime.gpu_architecture
        return payload

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        os.replace(temporary, target)

    @classmethod
    def read(cls, path: str | Path) -> EngineManifest:
        payload = json.loads(Path(path).read_text())
        if payload.get("schema") != MANIFEST_SCHEMA:
            raise ValueError(f"unsupported engine manifest schema: {payload.get('schema')!r}")
        runtime_payload = dict(payload["runtime"])
        runtime_payload.pop("gpu_architecture", None)
        runtime_payload["compute_capability"] = tuple(runtime_payload["compute_capability"])
        build_payload = dict(payload["build_settings"])
        for collection in (payload["sources"], payload["engines"]):
            for record in collection:
                record["inputs"] = tuple(
                    TensorSpec(
                        name=entry["name"], shape=tuple(entry["shape"]), dtype=entry["dtype"]
                    )
                    for entry in record.get("inputs", [])
                )
                record["outputs"] = tuple(
                    TensorSpec(
                        name=entry["name"], shape=tuple(entry["shape"]), dtype=entry["dtype"]
                    )
                    for entry in record.get("outputs", [])
                )
                record["members"] = tuple(
                    ArtifactMember(**entry) for entry in record.get("members", [])
                )
        return cls(
            schema=payload["schema"],
            contract=dict(payload["contract"]),
            contract_sha256=payload["contract_sha256"],
            checkpoint_files=dict(payload["checkpoint_files"]),
            checkpoint_sha256=payload["checkpoint_sha256"],
            vae_weights_path=payload["vae_weights_path"],
            vae_weights_sha256=payload["vae_weights_sha256"],
            runtime=RuntimeIdentity(**runtime_payload),
            build_settings=BuildSettings(**build_payload),
            tool_sources=tuple(SourceRevision(**record) for record in payload["tool_sources"]),
            sources=tuple(ArtifactRecord(**record) for record in payload["sources"]),
            engines=tuple(ArtifactRecord(**record) for record in payload["engines"]),
            build=_artifact_record_from_payload(payload["build"]),
            validation=_artifact_record_from_payload(payload["validation"]),
        )

    def validate(
        self,
        *,
        artifact_root: str | Path,
        contract: N2TensorRTContract,
        runtime: RuntimeIdentity,
    ) -> None:
        """Reject changed checkpoints, foreign plans, or edited receipts."""
        failures: dict[str, Any] = {}
        contract_payload = contract.to_dict()
        if self.contract != contract_payload or self.contract_sha256 != canonical_sha256(
            contract_payload
        ):
            failures["contract"] = "manifest does not match resolved checkpoint contract"
        checkpoint_files = checkpoint_file_hashes(contract.checkpoint)
        if checkpoint_files != self.checkpoint_files or self.checkpoint_sha256 != canonical_sha256(
            checkpoint_files
        ):
            failures["checkpoint"] = "checkpoint metadata changed after engine build"
        try:
            vae_weights = resolve_checkpoint_vae_weights(
                contract.checkpoint,
                self.vae_weights_path,
            )
            vae_weights_sha256 = sha256_file(vae_weights)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            failures["vae_weights"] = f"cannot resolve current VAE weights: {error}"
        else:
            if vae_weights_sha256 != self.vae_weights_sha256:
                failures["vae_weights"] = "VAE weights changed after engine build"
        runtime_mismatch = self.runtime.compatibility_mismatches(runtime)
        if runtime_mismatch:
            failures["runtime"] = runtime_mismatch
        root = Path(artifact_root).resolve(strict=True)
        artifact_failures = _validate_records(
            root,
            (self.build, *self.engines, self.validation),
        )
        if artifact_failures:
            failures["artifacts"] = artifact_failures
        else:
            build_failure = _build_receipt_failure(root / self.build.relative_path, self)
            build_receipt = None
            if build_failure is not None:
                failures["build"] = build_failure
            else:
                build_receipt = json.loads((root / self.build.relative_path).read_text())
            receipt_path = root / self.validation.relative_path
            try:
                receipt = json.loads(receipt_path.read_text())
            except (json.JSONDecodeError, OSError) as error:
                failures["validation"] = f"unreadable receipt: {error}"
            else:
                if receipt.get("schema") != VALIDATION_SCHEMA:
                    failures["validation"] = (
                        f"unsupported receipt schema: {receipt.get('schema')!r}"
                    )
                elif receipt.get("validation_policy") != VALIDATION_POLICY:
                    failures["validation"] = "receipt validation policy does not match runtime"
                elif receipt.get("validation_policy_sha256") != VALIDATION_POLICY_SHA256:
                    failures["validation"] = (
                        "receipt validation policy digest does not match runtime"
                    )
                elif receipt.get("passed") is not True:
                    failures["validation"] = "engine parity gates did not pass"
                elif receipt.get("contract_sha256") != self.contract_sha256:
                    failures["validation"] = "receipt contract digest does not match manifest"
                elif receipt.get("checkpoint_sha256") != self.checkpoint_sha256:
                    failures["validation"] = "receipt checkpoint digest does not match manifest"
                elif receipt.get("vae_weights_path") != self.vae_weights_path:
                    failures["validation"] = "receipt VAE weights path does not match manifest"
                elif receipt.get("vae_weights_sha256") != self.vae_weights_sha256:
                    failures["validation"] = "receipt VAE weights digest does not match manifest"
                elif receipt.get("artifacts") != {
                    record.role: record.sha256 for record in (*self.sources, *self.engines)
                }:
                    failures["validation"] = "receipt artifact digests do not match manifest"
                elif build_receipt is None:
                    failures["validation"] = "build receipt is invalid"
                elif receipt.get("oracles") != build_receipt.get("validation_samples"):
                    failures["validation"] = (
                        "receipt validation sample digests do not match the build receipt"
                    )
                elif not isinstance(receipt.get("validator_runtime"), dict):
                    failures["validation"] = "receipt omits the validator runtime identity"
                else:
                    validator_runtime_payload = dict(receipt["validator_runtime"])
                    validator_runtime_payload["compute_capability"] = tuple(
                        validator_runtime_payload["compute_capability"]
                    )
                    validator_runtime = RuntimeIdentity(**validator_runtime_payload)
                    if validator_runtime != self.runtime:
                        failures["validation"] = (
                            "receipt validator runtime identity does not match manifest"
                        )
                        validator_runtime = None
                    gates = receipt.get("gates")
                    gate_names = set(gates) if isinstance(gates, dict) else set()
                    passed_gates = {
                        name
                        for name, result in (gates.items() if isinstance(gates, dict) else ())
                        if isinstance(result, dict) and result.get("passed") is True
                    }
                    if validator_runtime is not None and (
                        gate_names != REQUIRED_PARITY_GATES or passed_gates != REQUIRED_PARITY_GATES
                    ):
                        failures["validation"] = (
                            "receipt must pass the required parity gates: "
                            f"{sorted(REQUIRED_PARITY_GATES)}"
                        )
                    elif validator_runtime is not None:
                        semantic_failures = _validation_gate_semantic_failures(gates)
                        if semantic_failures:
                            failures["validation"] = {
                                "gate_semantics": semantic_failures,
                            }
        if failures:
            raise RuntimeError(
                "N2 TensorRT manifest validation failed: " + json.dumps(failures, default=str)
            )


def _artifact_record_from_payload(payload: dict[str, Any]) -> ArtifactRecord:
    record = dict(payload)
    record["inputs"] = tuple(
        TensorSpec(name=entry["name"], shape=tuple(entry["shape"]), dtype=entry["dtype"])
        for entry in record.get("inputs", [])
    )
    record["outputs"] = tuple(
        TensorSpec(name=entry["name"], shape=tuple(entry["shape"]), dtype=entry["dtype"])
        for entry in record.get("outputs", [])
    )
    record["members"] = tuple(ArtifactMember(**entry) for entry in record.get("members", []))
    return ArtifactRecord(**record)


def _tensor_specs_payload(specs: tuple[TensorSpec, ...]) -> list[dict[str, Any]]:
    return [{"name": spec.name, "shape": list(spec.shape), "dtype": spec.dtype} for spec in specs]


def _build_receipt_failure(path: Path, manifest: EngineManifest) -> str | dict[str, Any] | None:
    try:
        receipt = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        return f"unreadable receipt: {error}"
    if not isinstance(receipt, dict):
        return "receipt must be an object"
    if receipt.get("schema") != BUILD_SCHEMA:
        return f"unsupported receipt schema: {receipt.get('schema')!r}"
    if receipt.get("quantize_mode") != "none":
        return "quantize_mode must be 'none'"
    if receipt.get("tensorrt") != manifest.runtime.tensorrt:
        return "TensorRT version does not match manifest runtime"
    expected_runtime = asdict(manifest.runtime)
    expected_runtime["compute_capability"] = list(manifest.runtime.compute_capability)
    if receipt.get("runtime") != expected_runtime:
        return "build runtime identity does not match manifest"
    if receipt.get("settings") != asdict(manifest.build_settings):
        return "build settings do not match manifest"
    validation_samples = receipt.get("validation_samples")
    if not isinstance(validation_samples, dict) or set(validation_samples) != set(
        VALIDATION_SAMPLE_PATHS
    ):
        return "validation_samples must contain every required validation sample"
    if any(
        not isinstance(digest, str) or len(digest) != 64 for digest in validation_samples.values()
    ):
        return "validation sample digests must be SHA-256 hex strings"

    expected_artifacts = {
        record.role: {
            "engine": record.relative_path,
            "engine_sha256": record.sha256,
            "engine_size_bytes": record.size_bytes,
            "numeric_format": record.numeric_format,
            "engine_inputs": _tensor_specs_payload(record.inputs),
            "engine_outputs": _tensor_specs_payload(record.outputs),
        }
        for record in manifest.engines
    }
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list):
        return "artifacts must be a list"
    actual_artifacts: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("role"), str):
            return "each artifact must have a string role"
        role = artifact["role"]
        if role in actual_artifacts:
            return f"duplicate build artifact role: {role!r}"
        actual_artifacts[role] = {
            key: artifact.get(key) for key in next(iter(expected_artifacts.values()))
        }
    if actual_artifacts != expected_artifacts:
        return {
            "artifact_records": "build receipt engines do not match manifest",
            "expected_roles": sorted(expected_artifacts),
            "actual_roles": sorted(actual_artifacts),
        }

    consumer_engines = receipt.get("engines")
    if not isinstance(consumer_engines, list) or len(consumer_engines) != len(manifest.engines):
        return "consumer engine records do not match manifest"
    for consumer, record in zip(consumer_engines, manifest.engines, strict=True):
        if (
            not isinstance(consumer, dict)
            or not isinstance(consumer.get("engine"), str)
            or Path(consumer["engine"]).is_absolute()
            or consumer["engine"] != record.relative_path
            or consumer.get("size_bytes") != record.size_bytes
            or consumer.get("precision") != record.numeric_format
        ):
            return "consumer engine records do not match manifest"
    return None


def _validation_gate_semantic_failures(gates: Any) -> dict[str, str]:
    if not isinstance(gates, dict):
        return {"gates": "must be an object"}
    failures: dict[str, str] = {}
    for gate, profile in VALIDATION_GATE_PROFILES.items():
        result = gates.get(gate)
        if not isinstance(result, dict):
            failures[gate] = "must be an object"
            continue
        if result.get("profile") != profile:
            failures[gate] = f"profile must be {profile!r}"
        elif result.get("thresholds") != VALIDATION_PROFILE_THRESHOLDS[profile]:
            failures[gate] = "thresholds do not match the admitted validation policy"
        elif result.get("gating_metrics") != VALIDATION_POLICY["gating_metrics"][profile]:
            failures[gate] = "gating metrics do not match the admitted validation policy"
    return failures


def _require_unique_roles(records: tuple[ArtifactRecord, ...], kind: str) -> None:
    if not records:
        raise ValueError(f"manifest cannot omit {kind} records")
    roles = [record.role for record in records]
    if len(roles) != len(set(roles)):
        raise ValueError(f"{kind} roles must be unique: {roles}")


def _require_artifact_io(records: tuple[ArtifactRecord, ...], kind: str) -> None:
    missing = [record.role for record in records if not record.inputs or not record.outputs]
    if missing:
        raise ValueError(f"{kind} records must include fixed input/output schemas: {missing}")


def _validate_records(root: Path, records: tuple[ArtifactRecord, ...]) -> dict[str, str]:
    failures: dict[str, str] = {}
    for record in records:
        members = record.members or (
            ArtifactMember(record.relative_path, record.sha256, record.size_bytes),
        )
        primary_path = (root / record.relative_path).resolve()
        try:
            primary_path.relative_to(root)
        except ValueError:
            failures[record.role] = "primary path escapes artifact root"
            continue
        if record.relative_path not in {member.relative_path for member in members}:
            failures[record.role] = "primary path is not an authenticated artifact member"
            continue
        for member in members:
            path = (root / member.relative_path).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                failures[record.role] = "path escapes artifact root"
                break
            if not path.is_file():
                failures[record.role] = f"missing member: {member.relative_path}"
                break
            if path.stat().st_size != member.size_bytes or sha256_file(path) != member.sha256:
                failures[record.role] = f"hash or size mismatch: {member.relative_path}"
                break
        else:
            aggregate_sha256 = (
                members[0].sha256
                if len(members) == 1
                else canonical_sha256([asdict(member) for member in members])
            )
            if (
                aggregate_sha256 != record.sha256
                or sum(member.size_bytes for member in members) != record.size_bytes
            ):
                failures[record.role] = "aggregate hash or size mismatch"
    return failures


def _nvidia_smi_field(device: int, field: str) -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={device}",
            f"--query-gpu={field}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if not value:
        raise RuntimeError(f"nvidia-smi returned no {field} for GPU {device}")
    return value
