# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Golden package records: the certificate a turn-key quantized N2 bundle is admitted against.

A golden package is a calibrated FP8 ONNX set (weights, export manifests, parity samples and
one canonical output set) published once from a certified research run. Its record is
repository data: it names the package content by digest, the checkpoint the package was
exported from, the gate policy its rows were judged by, the certification evidence (the main,
confirm and cond-less twin runs, byte identity of the roles cond-freeze does not touch, graph
replay bit identity, the outcome) and, per platform, an immutable admission row holding the
runtime envelope and the equivalence bounds an operator's fresh build must reproduce.

Nothing in a record is trusted as stored: every verdict is recomputed from the record's own
metrics against its effective gate policy, the package digest pins everything admission
consumes, and each admission row is pinned by its own digest so appending another platform's
row never invalidates an admitted bundle while a change to its own row does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import yaml

from groot.deployment.n2_trt.gate_evidence import (
    COND_CACHED_EVIDENCE_ENGINE_ROLES,
    EVIDENCE_ENGINE_ROLES,
    _at_least,
    _at_most,
    _canonical_sha256,
    _number,
)

if TYPE_CHECKING:
    import torch

GOLDEN_RECORD_SCHEMA = "groot_n2_golden_record_v1"
# The oracle manifest shipped in a package beside its canonical outputs.
GOLDEN_OUTPUTS_SCHEMA = "groot_n2_golden_outputs_v1"
GOLDEN_OUTPUTS_DIR = "golden_outputs"
GOLDEN_OUTPUTS_MANIFEST = f"{GOLDEN_OUTPUTS_DIR}/manifest.json"
# The sidecar a golden build writes beside its engines; it selects the golden admission route.
GOLDEN_SIDECAR_FILENAME = "n2_trt_golden.json"
GOLDEN_SIDECAR_MODES = ("serve", "certify")
GOLDEN_SIDECAR_SCHEMA = "groot_n2_golden_sidecar_v1"
RECORDS_DIR = Path(__file__).resolve().parent / "golden"

# The ratified gates every golden row is judged by (the research loop's defaults, pinned there
# by its own test). ``floor_worst_hint`` is a run-time hint, not a gate, and lives in the
# record's resolved facts instead.
RATIFIED_GATES_VERSION = "n2_trt_ratified_gates_v1"
RATIFIED_GATES: Mapping[str, Any] = MappingProxyType(
    {
        "paired_floor": 0.05,
        "increment_floor_pp": 2.0,
        "p90_floor": 0.03,
        "worst_ratio_floor": 2.0,
        "worst_abs_floor": 0.06,
        "max_fp32_layer_outputs": 130,
        "replay_jerk_postlpf_floor": 1.3,
        "replay_reversal_floor": 1.5,
        "replay_mismatch_vs_seed_base": True,
        "reversal_min_reference_events": 50,
    }
)
# The paired amplitude gates in the model's normalized output unit: per-episode paired relative
# RMSE in % of the action half-range (q99 - q01) / 2 over the paired action keys. The arms are
# the increment over the same-run BF16 TensorRT floor (its mean over the evaluation episodes and
# the upper end of that mean's bootstrap CI95) and the absolute tail (the episode p90 and its
# exact one-sided 95 % upper bound), at the robot-validated G1 open-door golden's own readings;
# the census and replay gates are unchanged. A record names the version its rows were judged by.
RATIFIED_GATES_VERSION_V2 = "n2_trt_ratified_gates_v2"
RATIFIED_GATES_V2: Mapping[str, Any] = MappingProxyType(
    {
        "rs_increment_mean_floor": 0.70,
        "rs_increment_ci95_upper_floor": 0.76,
        "rs_abs_p90_floor": 1.32,
        "rs_abs_p90_exact_upper_floor": 1.54,
        "max_fp32_layer_outputs": 130,
        "replay_jerk_postlpf_floor": 1.3,
        "replay_reversal_floor": 1.5,
        "replay_mismatch_vs_seed_base": True,
        "reversal_min_reference_events": 50,
    }
)
RATIFIED_GATES_BY_VERSION: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {RATIFIED_GATES_VERSION: RATIFIED_GATES, RATIFIED_GATES_VERSION_V2: RATIFIED_GATES_V2}
)
# The unit label a v2 amplitude block carries and its gated readings (block key, gate key).
RS_UNIT = "range_scaled_pct_half_range"
RS_AMPLITUDE_GATES: tuple[tuple[str, str], ...] = (
    ("rs_increment_mean", "rs_increment_mean_floor"),
    ("rs_increment_mean_ci95_upper", "rs_increment_ci95_upper_floor"),
    ("rs_abs_p90", "rs_abs_p90_floor"),
    ("rs_abs_p90_exact_upper", "rs_abs_p90_exact_upper_floor"),
)

# Equivalence policy: how a rebuilt engine's outputs are compared with the package's canonical
# outputs, and how a platform's bounds are set from two independent builds.
EQUIVALENCE_POLICY = "groot_n2_equivalence_v1"
EQUIVALENCE_EPSILON = 1e-6
REL_RMSE_FLOOR = 0.005
REL_MAXABS_FLOOR = 0.02
COSINE_SLACK_FLOOR = 1e-4
ENVELOPE_ARM = 3.0

# The only root entries of a published package directory outside the record's file list.
PACKAGE_CONTROL_FILES = ("SHA256SUMS", "PACKAGE.json")
REQUIRED_ROLES = EVIDENCE_ENGINE_ROLES
COND_CACHED_ROLE = "transformer_joint_condcached"
KNOWN_ROLES = COND_CACHED_EVIDENCE_ENGINE_ROLES
NUMERIC_FORMATS = ("bf16", "fp8")
BUILD_SETTINGS_FIELDS = (
    "strongly_typed",
    "builder_optimization_level",
    "transformer_precision",
    "vae_precision",
    "codec",
    "action_context_mode",
    "cond_kv_freeze",
)
CHECKPOINT_METADATA_FILES = ("config.json", "processor_config.json", "statistics.json")
# The dependency-path fields a checkpoint audit repoints per host; they are blanked before the
# metadata digests are taken so the identity survives relocation and nothing else.
CHECKPOINT_PATH_FIELDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "config.json": (
            "diffusion_model_cfg.diffusion_model_pretrained_path",
            "text_encoder_cfg.text_encoder_pretrained_path",
            "tokenizer_path",
            "vae_cfg.vae_pretrained_path",
        ),
        "processor_config.json": ("processor_kwargs.tokenizer_path",),
        "statistics.json": (),
    }
)
CERTIFICATION_LEVELS = ("amplitude", "temporal")
CERTIFIED_RUNS = ("main", "confirm", "twin")
BLOCK_NAMES = ("cond_on", "cond_off", "plain")
GRAPH_LEGS = ("cond_on", "cond_off")
SERVED_MODES = GRAPH_LEGS
ADMITTED_MODES_FIELD = "admitted_modes"
# Outcome labels a record may carry and the outcome-row flags each requires to be true.
OUTCOME_REQUIRED_PASSES: Mapping[str, tuple[tuple[str, str], ...]] = MappingProxyType(
    {
        "all_on_passed_default_on": (
            ("floor", "on_pass"),
            ("main", "on_pass"),
            ("main", "off_pass"),
            ("confirm", "on_pass"),
            ("confirm", "off_pass"),
            ("twin", "on_pass"),
        ),
        "all_on_passed_default_off": (
            ("floor", "on_pass"),
            ("main", "on_pass"),
            ("main", "off_pass"),
            ("confirm", "on_pass"),
            ("confirm", "off_pass"),
            ("twin", "on_pass"),
        ),
        "integrated_cond_disabled": (
            ("floor", "on_pass"),
            ("main", "off_pass"),
            ("confirm", "off_pass"),
            ("twin", "on_pass"),
        ),
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PACKAGE_ID = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_MAJOR_MINOR = re.compile(r"^[0-9]+\.[0-9]+$")


class GoldenRecordError(RuntimeError):
    """A malformed record, or stored evidence that contradicts its own verdicts."""


# ── digests ──────────────────────────────────────────────────────────────────────────────


def canonical_sha256(payload: Any) -> str:
    """sha256 of the canonical JSON form (sorted keys, no whitespace) of ``payload``."""
    return _canonical_sha256(payload)


def sha256_file(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _plain(value: Any) -> Any:
    """JSON-ready copy: mappings to dicts, sequences to lists, dataclasses through ``to_dict``."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


# ── the record ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GatePolicy:
    """The effective gates a record's rows were judged by: ratified defaults plus reviewed overrides.

    ``resolved`` holds run-time facts (the floor's worst episode and how the worst-episode
    ratio was resolved); they are kept for the reader and are never gates.
    """

    ratified: dict[str, Any]
    overrides: dict[str, Any]
    resolved: dict[str, Any]

    @classmethod
    def init_from_dict(cls, payload: Mapping[str, Any]) -> GatePolicy:
        _require_mapping(payload, "gate_policy")
        return cls(
            ratified=dict(_require_mapping(payload.get("ratified"), "gate_policy.ratified")),
            overrides=dict(_require_mapping(payload.get("overrides", {}), "gate_policy.overrides")),
            resolved=dict(_require_mapping(payload.get("resolved", {}), "gate_policy.resolved")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ratified": _plain(self.ratified),
            "overrides": _plain(self.overrides),
            "resolved": _plain(self.resolved),
        }

    def effective(self) -> dict[str, Any]:
        """The ratified values with the reviewed overrides applied."""
        gates = dict(self.ratified.get("values") or {})
        for key, override in self.overrides.items():
            gates[key] = override.get("value") if isinstance(override, Mapping) else override
        return gates


@dataclass(frozen=True)
class Certification:
    """The evidence a package was certified on, self-contained and free of host paths."""

    level: str
    runs: dict[str, dict[str, Any]]
    byte_identity: dict[str, Any]
    graph_replay: dict[str, dict[str, Any]]
    outcome: dict[str, Any]

    @classmethod
    def init_from_dict(cls, payload: Mapping[str, Any]) -> Certification:
        _require_mapping(payload, "certification")
        runs = _require_mapping(payload.get("runs"), "certification.runs")
        legs = _require_mapping(payload.get("graph_replay"), "certification.graph_replay")
        return cls(
            level=str(payload.get("level")),
            runs={
                str(name): dict(_require_mapping(run, f"run {name!r}"))
                for name, run in runs.items()
            },
            byte_identity=dict(
                _require_mapping(payload.get("byte_identity"), "certification.byte_identity")
            ),
            graph_replay={
                str(name): dict(_require_mapping(leg, f"graph leg {name!r}"))
                for name, leg in legs.items()
            },
            outcome=dict(_require_mapping(payload.get("outcome"), "certification.outcome")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "runs": _plain(self.runs),
            "byte_identity": _plain(self.byte_identity),
            "graph_replay": _plain(self.graph_replay),
            "outcome": _plain(self.outcome),
        }


@dataclass(frozen=True)
class GoldenRecord:
    """One golden package's record (see the module docstring for what each section binds)."""

    schema: str
    package: dict[str, Any]
    gate_policy: GatePolicy
    certification: Certification
    admission_rows: tuple[dict[str, Any], ...]
    serving_measurements: tuple[dict[str, Any], ...]

    @classmethod
    def init_from_dict(cls, payload: Mapping[str, Any]) -> GoldenRecord:
        _require_mapping(payload, "record")
        rows = payload.get("admission_rows", [])
        serving = payload.get("serving_measurements", [])
        if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
            raise GoldenRecordError("admission_rows must be a list")
        if not isinstance(serving, Sequence) or isinstance(serving, str | bytes):
            raise GoldenRecordError("serving_measurements must be a list")
        return cls(
            schema=str(payload.get("schema")),
            package=dict(_require_mapping(payload.get("package"), "package")),
            gate_policy=GatePolicy.init_from_dict(payload.get("gate_policy")),
            certification=Certification.init_from_dict(payload.get("certification")),
            admission_rows=tuple(dict(_require_mapping(row, "admission row")) for row in rows),
            serving_measurements=tuple(
                dict(_require_mapping(row, "serving measurement")) for row in serving
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "package": _plain(self.package),
            "gate_policy": self.gate_policy.to_dict(),
            "certification": self.certification.to_dict(),
            "admission_rows": _plain(self.admission_rows),
            "serving_measurements": _plain(self.serving_measurements),
        }

    @property
    def package_id(self) -> str:
        return str(self.package["package_id"])

    @property
    def roles(self) -> dict[str, dict[str, Any]]:
        return dict(self.package["roles"])

    @property
    def cond_kv_freeze(self) -> bool:
        return COND_CACHED_ROLE in self.package["roles"]

    def admission_row_for(self, identity: Mapping[str, Any]) -> dict[str, Any] | None:
        """The admission row for this hardware, or None.

        ``identity`` carries ``gpu_name``, ``sm_arch``, ``tensorrt``, ``cuda`` and ``driver`` as
        the runtime reports them; a row matches on the GPU-name fragment and the SM architecture
        (``envelope_matches``). When several rows cover the same hardware, the one certified on
        this software stack (``envelope_provenance_matches``) is preferred, then the first.
        """
        matches = [
            row for row in self.admission_rows if envelope_matches(row["envelope"], identity)
        ]
        for row in matches:
            if envelope_provenance_matches(row["envelope"], identity):
                return row
        return matches[0] if matches else None


def package_sha256(record: GoldenRecord) -> str:
    """The digest sidecars pin: everything admission consumes, never the platform rows."""
    payload = record.to_dict()
    return canonical_sha256(
        {
            "schema": payload["schema"],
            "package": payload["package"],
            "gate_policy": payload["gate_policy"],
            "certification": payload["certification"],
        }
    )


def row_sha256(row: Mapping[str, Any]) -> str:
    """The digest of one admission row over everything but its own ``row_sha256`` field."""
    payload = {key: value for key, value in _plain(row).items() if key != "row_sha256"}
    return canonical_sha256(payload)


def served_mode_name(cond_kv_freeze: bool) -> str:
    """The certified block (and graph leg) a served conditioning-cache mode maps to."""
    return "cond_on" if cond_kv_freeze else "cond_off"


def admitted_served_modes(row: Mapping[str, Any]) -> tuple[str, ...]:
    """The served modes a platform row admits.

    A row names them in ``admitted_modes`` — a passed subset of its certified blocks, because a
    package can pass one served mode's gates and fail the other's on a platform; a row minted
    before the field existed admits every certified block it carries.
    """
    explicit = row.get(ADMITTED_MODES_FIELD)
    if explicit is not None:
        return tuple(str(mode) for mode in explicit)
    return tuple(mode for mode in SERVED_MODES if mode in (row.get("certified") or {}))


def envelope_matches(envelope: Mapping[str, Any], identity: Mapping[str, Any]) -> bool:
    """Whether a runtime identity falls inside an admission row's runtime envelope.

    A row is keyed on the HARDWARE it was certified on: the GPU-name fragment and the SM
    architecture. The TensorRT and CUDA ``major.minor`` and the driver major it also records are
    the provenance of its certified numbers (``envelope_provenance_matches``), never admission
    keys: the package is version-free, so a bundle rebuilt on the same GPU class under another
    software stack is admitted by reproducing the canonical outputs within the row's equivalence
    bounds, and the receipt says whether the certified speed was measured on this stack.
    """
    gpu_name = str(identity.get("gpu_name") or "")
    return (
        str(envelope.get("gpu_name_contains") or "") in gpu_name
        and bool(envelope.get("gpu_name_contains"))
        and envelope.get("sm_arch") == identity.get("sm_arch")
    )


def envelope_provenance_matches(envelope: Mapping[str, Any], identity: Mapping[str, Any]) -> bool:
    """Whether the runtime's software stack is the one the row's numbers were certified on.

    Compares the TensorRT ``major.minor`` and the driver major, the two stack facts the engines
    depend on. The row's ``cuda`` is recorded but not compared: the runtime reports torch's CUDA
    build, which the TensorRT engines do not depend on.
    """
    return envelope.get("tensorrt") == _major_minor(identity.get("tensorrt")) and envelope.get(
        "driver_major"
    ) == _major(identity.get("driver"))


def _major_minor(version: Any) -> str | None:
    parts = str(version or "").split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
        return None
    return f"{parts[0]}.{parts[1]}"


def _major(version: Any) -> int | None:
    head = str(version or "").split(".")[0]
    return int(head) if head.isdigit() else None


# ── the golden sidecar and the runtime envelope ──────────────────────────────────────────


def write_golden_sidecar(
    root: str | Path,
    *,
    package_id: str,
    package_sha256: str,
    mode: str,
    platform_row_sha256: str | None = None,
    files_verified: bool = True,
) -> Path:
    """Write the sidecar that selects the golden admission route for a build root.

    ``serve`` mode names the admitting platform row by digest; ``certify`` mode carries no row
    (the platform is not certified yet) and mints a candidate receipt.
    """
    if mode not in GOLDEN_SIDECAR_MODES:
        raise GoldenRecordError(f"sidecar mode must be one of {list(GOLDEN_SIDECAR_MODES)}")
    if (platform_row_sha256 is None) != (mode == "certify"):
        raise GoldenRecordError(
            "serve mode needs the admitting row digest; certify mode carries none"
        )
    payload: dict[str, Any] = {
        "schema": GOLDEN_SIDECAR_SCHEMA,
        "package_id": package_id,
        "package_sha256": package_sha256,
        "mode": mode,
        "files_verified": bool(files_verified),
    }
    if platform_row_sha256 is not None:
        payload["platform_row_sha256"] = platform_row_sha256
    target = Path(root) / GOLDEN_SIDECAR_FILENAME
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, target)
    return target


def read_golden_sidecar(root: str | Path) -> dict[str, Any]:
    """Read and validate a build root's golden sidecar."""
    path = Path(root) / GOLDEN_SIDECAR_FILENAME
    if not path.is_file():
        raise GoldenRecordError(f"{path} does not exist")
    payload = json.loads(path.read_text())
    _require_mapping(payload, "golden sidecar")
    if payload.get("schema") != GOLDEN_SIDECAR_SCHEMA:
        raise GoldenRecordError(f"unsupported golden sidecar schema {payload.get('schema')!r}")
    _require_str(payload.get("package_id"), "sidecar package_id")
    _require_sha256(payload.get("package_sha256"), "sidecar package_sha256")
    if payload.get("mode") not in GOLDEN_SIDECAR_MODES:
        raise GoldenRecordError(f"sidecar mode must be one of {list(GOLDEN_SIDECAR_MODES)}")
    if payload.get("files_verified") is not True:
        raise GoldenRecordError("the sidecar must record that the package files were verified")
    row = payload.get("platform_row_sha256")
    if payload["mode"] == "serve":
        _require_sha256(row, "sidecar platform_row_sha256")
    elif row is not None:
        raise GoldenRecordError("a certify-mode sidecar carries no platform row digest")
    return dict(payload)


def runtime_envelope_identity(runtime: Any) -> dict[str, Any]:
    """The envelope-facing view of a runtime identity (what ``envelope_matches`` reads)."""
    return {
        "gpu_name": getattr(runtime, "gpu_name", None),
        "sm_arch": getattr(runtime, "gpu_architecture", None),
        "tensorrt": getattr(runtime, "tensorrt", None),
        "cuda": getattr(runtime, "cuda", None),
        "driver": getattr(runtime, "driver", None),
    }


# ── reading, writing, listing ────────────────────────────────────────────────────────────


def read_record(path: str | Path) -> GoldenRecord:
    """Load and validate a record file (yaml); refuses anything the validator refuses."""
    payload = yaml.safe_load(Path(path).read_text())
    return validate_record(GoldenRecord.init_from_dict(payload))


def write_record(record: GoldenRecord, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(record.to_dict(), sort_keys=False))


def list_records(records_dir: str | Path | None = None) -> list[str]:
    directory = Path(records_dir) if records_dir is not None else RECORDS_DIR
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.yaml"))


def load_record(package_id: str, records_dir: str | Path | None = None) -> GoldenRecord:
    directory = Path(records_dir) if records_dir is not None else RECORDS_DIR
    path = directory / f"{package_id}.yaml"
    if not path.is_file():
        raise GoldenRecordError(f"unknown golden package {package_id!r}: no record at {path}")
    record = read_record(path)
    if record.package_id != package_id:
        raise GoldenRecordError(
            f"record {path} names package {record.package_id!r}, not {package_id!r}"
        )
    return record


# ── validation ───────────────────────────────────────────────────────────────────────────


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GoldenRecordError(f"{name} must be an object")
    return value


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise GoldenRecordError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.match(value):
        raise GoldenRecordError(f"{name} must be a sha256 hex digest")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise GoldenRecordError(f"{name} must be a boolean")
    return value


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GoldenRecordError(f"{name} must be an integer >= {minimum}")
    return value


def _require_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise GoldenRecordError(f"{name} must be a finite number")
    return float(value)


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise GoldenRecordError(f"{name} must be a list")
    return list(value)


def _validate_package(package: Mapping[str, Any]) -> None:
    package_id = _require_str(package.get("package_id"), "package.package_id")
    if not _PACKAGE_ID.match(package_id):
        raise GoldenRecordError(f"package_id {package_id!r} must be lower-case [a-z0-9_]")
    _require_str(package.get("mirror_key"), "package.mirror_key")
    files = _require_mapping(package.get("files"), "package.files")
    if not files:
        raise GoldenRecordError("package.files must list the package content")
    for relative, entry in files.items():
        _validate_relative_path(relative)
        if relative in PACKAGE_CONTROL_FILES:
            raise GoldenRecordError(f"{relative} is a package control file, not package content")
        body = _require_mapping(entry, f"package.files[{relative!r}]")
        _require_sha256(body.get("sha256"), f"package.files[{relative!r}].sha256")
        _require_int(body.get("bytes"), f"package.files[{relative!r}].bytes")
    _require_sha256(package.get("sha256sums_sha256"), "package.sha256sums_sha256")
    roles = _require_mapping(package.get("roles"), "package.roles")
    unknown = sorted(set(roles) - set(KNOWN_ROLES))
    if unknown:
        raise GoldenRecordError(f"unknown engine roles in package.roles: {unknown}")
    missing = sorted(set(REQUIRED_ROLES) - set(roles))
    if missing:
        raise GoldenRecordError(f"package.roles is missing required roles: {missing}")
    for role, content in roles.items():
        body = _require_mapping(content, f"package.roles[{role!r}]")
        _require_sha256(body.get("onnx_sha256"), f"package.roles[{role!r}].onnx_sha256")
        samples = _require_mapping(
            body.get("sample_sha256s"), f"package.roles[{role!r}].sample_sha256s"
        )
        for sample, digest in samples.items():
            _validate_relative_path(str(sample))
            _require_sha256(digest, f"package.roles[{role!r}].sample_sha256s[{sample!r}]")
        if body.get("numeric_format") not in NUMERIC_FORMATS:
            raise GoldenRecordError(
                f"package.roles[{role!r}].numeric_format must be one of {list(NUMERIC_FORMATS)}"
            )
    _require_sha256(package.get("oracle_manifest_sha256"), "package.oracle_manifest_sha256")
    _require_sha256(package.get("vae_weights_sha256"), "package.vae_weights_sha256")
    settings = _require_mapping(package.get("build_settings"), "package.build_settings")
    if tuple(sorted(settings)) != tuple(sorted(BUILD_SETTINGS_FIELDS)):
        raise GoldenRecordError(
            f"package.build_settings must carry exactly {list(BUILD_SETTINGS_FIELDS)}, got {sorted(settings)}"
        )
    _require_bool(settings.get("strongly_typed"), "build_settings.strongly_typed")
    _require_int(
        settings.get("builder_optimization_level"), "build_settings.builder_optimization_level"
    )
    _require_bool(settings.get("cond_kv_freeze"), "build_settings.cond_kv_freeze")
    if settings["cond_kv_freeze"] != (COND_CACHED_ROLE in roles):
        raise GoldenRecordError(
            "build_settings.cond_kv_freeze must agree with the presence of the conditioning-cached role"
        )
    if settings.get("transformer_precision") != roles["transformer_joint"]["numeric_format"]:
        raise GoldenRecordError(
            "build_settings.transformer_precision must equal the joint role's format"
        )
    if settings.get("vae_precision") != roles["vae_encoder"]["numeric_format"]:
        raise GoldenRecordError("build_settings.vae_precision must equal the VAE role's format")
    if COND_CACHED_ROLE in roles and (
        roles[COND_CACHED_ROLE]["numeric_format"] != roles["transformer_joint"]["numeric_format"]
    ):
        raise GoldenRecordError("the conditioning-cached role's format must equal the joint's")
    _validate_checkpoint_identity(_require_mapping(package.get("checkpoint"), "package.checkpoint"))
    recipe = _require_mapping(package.get("recipe"), "package.recipe")
    for key in ("name", "recipe_hash", "yaml_path"):
        _require_str(recipe.get(key), f"package.recipe.{key}")
    calibration = _require_mapping(package.get("calibration"), "package.calibration")
    _require_str(calibration.get("dataset_source"), "package.calibration.dataset_source")
    _require_sha256(calibration.get("draw_sha256"), "package.calibration.draw_sha256")
    _require_mapping(package.get("evaluation"), "package.evaluation")


def _validate_relative_path(relative: str) -> None:
    path = Path(relative)
    if not relative or path.is_absolute() or ".." in path.parts or relative.startswith("./"):
        raise GoldenRecordError(f"{relative!r} is not a plain relative path")


def _validate_checkpoint_identity(checkpoint: Mapping[str, Any]) -> None:
    shards = _require_mapping(checkpoint.get("weight_shards"), "checkpoint.weight_shards")
    if not shards:
        raise GoldenRecordError("checkpoint.weight_shards must name at least one shard")
    for name, digest in shards.items():
        _require_sha256(digest, f"checkpoint.weight_shards[{name!r}]")
    metadata = _require_mapping(checkpoint.get("metadata"), "checkpoint.metadata")
    if tuple(sorted(metadata)) != tuple(sorted(CHECKPOINT_METADATA_FILES)):
        raise GoldenRecordError(
            f"checkpoint.metadata must digest exactly {list(CHECKPOINT_METADATA_FILES)}"
        )
    for name, digest in metadata.items():
        _require_sha256(digest, f"checkpoint.metadata[{name!r}]")
    fields = _require_mapping(
        checkpoint.get("path_normalized_fields"), "checkpoint.path_normalized_fields"
    )
    if {key: tuple(value) for key, value in fields.items()} != dict(CHECKPOINT_PATH_FIELDS):
        raise GoldenRecordError(
            "checkpoint.path_normalized_fields must be exactly the audit-repointed dependency paths"
        )
    _require_str(checkpoint.get("checkpoint_source"), "checkpoint.checkpoint_source")


def _validate_gate_policy(policy: GatePolicy) -> None:
    ratified = policy.ratified
    table = RATIFIED_GATES_BY_VERSION.get(ratified.get("version"))
    if table is None:
        raise GoldenRecordError(
            f"gate_policy.ratified.version must be one of {sorted(RATIFIED_GATES_BY_VERSION)}, "
            f"got {ratified.get('version')!r}"
        )
    if dict(_require_mapping(ratified.get("values"), "gate_policy.ratified.values")) != dict(table):
        raise GoldenRecordError("gate_policy.ratified.values must equal the ratified gates")
    for key, override in policy.overrides.items():
        if key not in table:
            raise GoldenRecordError(f"gate_policy.overrides names an unknown gate {key!r}")
        body = _require_mapping(override, f"gate_policy.overrides[{key!r}]")
        if "value" not in body:
            raise GoldenRecordError(f"gate_policy.overrides[{key!r}] has no value")
        _require_str(body.get("reason"), f"gate_policy.overrides[{key!r}].reason")


def _validate_certification(certification: Certification, roles: Mapping[str, Any]) -> None:
    if certification.level not in CERTIFICATION_LEVELS:
        raise GoldenRecordError(f"unknown certification level {certification.level!r}")
    if tuple(sorted(certification.runs)) != tuple(sorted(CERTIFIED_RUNS)):
        raise GoldenRecordError(f"certification.runs must be exactly {list(CERTIFIED_RUNS)}")
    cond = COND_CACHED_ROLE in roles
    for name, run in certification.runs.items():
        for key in ("run_id", "variant_id", "recipe_hash"):
            _require_str(run.get(key), f"run {name!r}.{key}")
        _require_sha256(run.get("ledger_row_sha256"), f"run {name!r}.ledger_row_sha256")
        if _require_finite(run.get("floor_paired_worst"), f"run {name!r}.floor_paired_worst") <= 0:
            raise GoldenRecordError(f"run {name!r}.floor_paired_worst must be positive")
        layers = _require_mapping(run.get("layer_outputs"), f"run {name!r}.layer_outputs")
        expected_roles = set(roles) if name != "twin" else set(REQUIRED_ROLES)
        if set(layers) != expected_roles:
            raise GoldenRecordError(
                f"run {name!r}.layer_outputs must cover {sorted(expected_roles)}, got {sorted(layers)}"
            )
        for role, body in layers.items():
            entry = _require_mapping(body, f"run {name!r}.layer_outputs[{role!r}]")
            if entry.get("numeric_format") != roles[role]["numeric_format"]:
                raise GoldenRecordError(
                    f"run {name!r}.layer_outputs[{role!r}].numeric_format contradicts the package role"
                )
            _require_int(entry.get("fp32"), f"run {name!r}.layer_outputs[{role!r}].fp32")
            _require_int(entry.get("fp8"), f"run {name!r}.layer_outputs[{role!r}].fp8")
        blocks = _require_mapping(run.get("blocks"), f"run {name!r}.blocks")
        if not blocks:
            raise GoldenRecordError(f"run {name!r} has no measurement blocks")
        allowed = (
            {"plain"} if name == "twin" else ({"cond_on", "cond_off"} if cond else {"cond_off"})
        )
        if set(blocks) - allowed:
            raise GoldenRecordError(
                f"run {name!r} blocks must be within {sorted(allowed)}, got {sorted(blocks)}"
            )
        for block_name, block in blocks.items():
            body = _require_mapping(block, f"run {name!r}.blocks[{block_name!r}]")
            served = _require_bool(
                body.get("served_cond_kv_freeze"), f"block {block_name!r}.served_cond_kv_freeze"
            )
            if served != (block_name == "cond_on"):
                raise GoldenRecordError(
                    f"block {block_name!r} served_cond_kv_freeze contradicts its name"
                )
            _require_mapping(body.get("amplitude"), f"block {block_name!r}.amplitude")
            _require_mapping(body.get("speed"), f"block {block_name!r}.speed")
            temporal = _require_mapping(body.get("temporal"), f"block {block_name!r}.temporal")
            _require_mapping(
                temporal.get("families", {}), f"block {block_name!r}.temporal.families"
            )
            _require_bool(body.get("verdict_pass"), f"block {block_name!r}.verdict_pass")
    identity = certification.byte_identity
    _require_sha256(identity.get("receipt_sha256"), "byte_identity.receipt_sha256")
    files = _require_list(identity.get("files"), "byte_identity.files")
    if not files:
        raise GoldenRecordError("byte_identity.files must list the compared files")
    for entry in files:
        body = _require_mapping(entry, "byte_identity.files[]")
        _require_str(body.get("file"), "byte_identity.files[].file")
        _require_sha256(body.get("cond_sha256"), "byte_identity.files[].cond_sha256")
        _require_sha256(body.get("twin_sha256"), "byte_identity.files[].twin_sha256")
        _require_int(body.get("bytes"), "byte_identity.files[].bytes")
        _require_bool(body.get("equal"), "byte_identity.files[].equal")
    _require_int(identity.get("external_tensors"), "byte_identity.external_tensors")
    _require_list(identity.get("external_differing"), "byte_identity.external_differing")
    _require_list(identity.get("external_missing"), "byte_identity.external_missing")
    _require_str(identity.get("result"), "byte_identity.result")
    reuse = _require_mapping(identity.get("action_plan_reuse"), "byte_identity.action_plan_reuse")
    _require_sha256(reuse.get("action_engine_sha256"), "action_plan_reuse.action_engine_sha256")
    _require_sha256(
        reuse.get("floor_action_engine_sha256"), "action_plan_reuse.floor_action_engine_sha256"
    )
    _require_bool(reuse.get("equal"), "action_plan_reuse.equal")
    legs = certification.graph_replay
    if not legs or set(legs) - set(GRAPH_LEGS):
        raise GoldenRecordError(
            f"graph_replay legs must be within {list(GRAPH_LEGS)} and non-empty"
        )
    if "cond_on" in legs and not cond:
        raise GoldenRecordError(
            "a record without the conditioning-cached role has no cond_on graph leg"
        )
    for name, leg in legs.items():
        window = _require_list(leg.get("window"), f"graph leg {name!r}.window")
        if len(window) != 2 or any(
            _require_int(item, "window entry", minimum=1) < 1 for item in window
        ):
            raise GoldenRecordError(f"graph leg {name!r}.window must be [warmups, measured]")
        _require_str(leg.get("engine_source"), f"graph leg {name!r}.engine_source")
        _require_bool(leg.get("served"), f"graph leg {name!r}.served")
        _require_bool(leg.get("bundle_flag"), f"graph leg {name!r}.bundle_flag")
        _require_mapping(leg.get("backend_counts"), f"graph leg {name!r}.backend_counts")
        _require_sha256(
            leg.get("graph_off_actions_sha256"), f"graph leg {name!r}.graph_off_actions_sha256"
        )
        _require_sha256(
            leg.get("graph_on_actions_sha256"), f"graph leg {name!r}.graph_on_actions_sha256"
        )
        _require_bool(leg.get("bit_identical"), f"graph leg {name!r}.bit_identical")
    outcome = certification.outcome
    _require_sha256(outcome.get("file_sha256"), "outcome.file_sha256")
    if outcome.get("label") not in OUTCOME_REQUIRED_PASSES:
        raise GoldenRecordError(
            f"outcome.label {outcome.get('label')!r} is not an accepted outcome"
        )
    if (outcome["label"] == "integrated_cond_disabled") == cond:
        raise GoldenRecordError(
            "outcome.label contradicts the presence of the conditioning-cached role"
        )
    _require_mapping(outcome.get("rows"), "outcome.rows")


def validate_admission_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """A platform's immutable admission row: envelope, two-build bounds, pinned by its digest."""
    body = dict(_require_mapping(row, "admission row"))
    envelope = _require_mapping(body.get("envelope"), "admission row envelope")
    _require_str(envelope.get("gpu_name_contains"), "envelope.gpu_name_contains")
    _require_str(envelope.get("sm_arch"), "envelope.sm_arch")
    for key in ("tensorrt", "cuda"):
        value = _require_str(envelope.get(key), f"envelope.{key}")
        if not _MAJOR_MINOR.match(value):
            raise GoldenRecordError(f"envelope.{key} must be major.minor, got {value!r}")
    _require_int(envelope.get("driver_major"), "envelope.driver_major", minimum=1)
    _require_str(body.get("date"), "admission row date")
    if body.get("level") not in CERTIFICATION_LEVELS:
        raise GoldenRecordError(f"admission row level {body.get('level')!r} is unknown")
    if body.get("equivalence_policy") != EQUIVALENCE_POLICY:
        raise GoldenRecordError(f"admission row must be judged by {EQUIVALENCE_POLICY!r}")
    if body.get("single_build"):
        raise GoldenRecordError("a single-build row is provisional and never admits a bundle")
    entries = _require_mapping(body.get("entries"), "admission row entries")
    if not entries:
        raise GoldenRecordError("admission row has no equivalence entries")
    for name, entry in entries.items():
        bounds = _require_mapping(entry, f"entry {name!r}")
        for key in ("envelope", "maxabs_bound", "cosine_floor"):
            _require_finite(bounds.get(key), f"entry {name!r}.{key}")
        builds = _require_list(bounds.get("builds"), f"entry {name!r}.builds")
        if len(builds) != 2:
            raise GoldenRecordError(f"entry {name!r} must record exactly two builds")
        for build in builds:
            distances = _require_mapping(build, f"entry {name!r} build")
            for key in ("rel_rmse", "rel_maxabs", "cosine"):
                _require_finite(distances.get(key), f"entry {name!r} build {key}")
    certified = _require_mapping(body.get("certified"), "admission row certified measurements")
    if _require_finite(body.get("floor_paired_worst"), "admission row floor_paired_worst") <= 0:
        raise GoldenRecordError("admission row floor_paired_worst must be positive")
    explicit = body.get(ADMITTED_MODES_FIELD)
    if explicit is not None:
        modes = _require_list(explicit, f"admission row {ADMITTED_MODES_FIELD}")
        if not modes or len(set(modes)) != len(modes) or any(m not in SERVED_MODES for m in modes):
            raise GoldenRecordError(
                f"{ADMITTED_MODES_FIELD} must be a non-empty subset of {list(SERVED_MODES)} "
                "without repeats"
            )
        for mode in modes:
            block = certified.get(mode)
            if not isinstance(block, Mapping) or block.get("passed") is not True:
                raise GoldenRecordError(
                    f"admitted mode {mode!r} must carry a passed certified block"
                )
    _require_sha256(
        body.get("certification_evidence_sha256"), "admission row certification_evidence_sha256"
    )
    _require_list(body.get("source_runs"), "admission row source_runs")
    digest = _require_sha256(body.get("row_sha256"), "admission row row_sha256")
    if digest != row_sha256(body):
        raise GoldenRecordError("admission row digest does not match its content")
    return body


def validate_record(record: GoldenRecord) -> GoldenRecord:
    """Refuse anything malformed and any stored verdict the record's own evidence contradicts."""
    if record.schema != GOLDEN_RECORD_SCHEMA:
        raise GoldenRecordError(f"unsupported golden record schema {record.schema!r}")
    _validate_package(record.package)
    _validate_gate_policy(record.gate_policy)
    _validate_certification(record.certification, record.package["roles"])
    for row in record.admission_rows:
        validate_admission_row(row)
        if row["level"] != record.certification.level:
            raise GoldenRecordError(
                "an admission row's level must equal the record's certification level"
            )
        judged = row_verdict(row, record.gate_policy)
        if not judged["passed"]:
            raise GoldenRecordError(
                "an admission row's evidence does not support its verdicts: "
                + "; ".join(judged["failures"])
            )
    for row in record.serving_measurements:
        _require_mapping(row, "serving measurement")
    result = verdict(record.certification, record.gate_policy)
    if not result["passed"]:
        raise GoldenRecordError(
            "the record's evidence does not support its certificate: "
            + "; ".join(result["failures"])
        )
    return record


# ── the recomputed verdict ───────────────────────────────────────────────────────────────


def verdict(certification: Certification, gate_policy: GatePolicy) -> dict[str, Any]:
    """Recompute every certification verdict from the stored metrics alone.

    Each run's blocks are judged against the effective gates (amplitude for every level;
    temporal families for a temporal record, required on the main and confirm runs and judged
    when present on the cond-less twin, whose evidence is amplitude and byte identity); the
    byte-identity pairs, the graph-replay digests and the outcome flags are re-derived from
    their stored values. A stored verdict its evidence contradicts is a failure, as is any
    block, pair, leg or flag that fails.
    """
    gates = gate_policy.effective()
    failures: list[str] = []
    runs: dict[str, Any] = {}
    for run_name, run in certification.runs.items():
        run_result: dict[str, Any] = {"layer_outputs": {}, "blocks": {}}
        floor_worst = _number(run, "floor_paired_worst", failures)
        layer_local: list[str] = []
        for role, entry in (run.get("layer_outputs") or {}).items():
            fp32 = _number(entry, "fp32", layer_local)
            fp8 = _number(entry, "fp8", layer_local)
            cap = gates.get("max_fp32_layer_outputs")
            expects_fp8 = entry.get("numeric_format") == "fp8"
            fp32_ok = fp32 is not None and cap is not None and fp32 <= float(cap)
            fp8_ok = fp8 is not None and ((fp8 > 0) if expects_fp8 else (fp8 == 0))
            if not fp32_ok:
                layer_local.append(f"{role}: fp32 layer outputs {fp32} > cap {cap}")
            if not fp8_ok:
                layer_local.append(
                    f"{role}: {fp8} fp8 layer outputs contradict format {entry.get('numeric_format')}"
                )
            run_result["layer_outputs"][role] = {
                "fp32": fp32,
                "fp8": fp8,
                "passed": fp32_ok and fp8_ok,
            }
        failures.extend(f"[{run_name}] {reason}" for reason in layer_local)
        for block_name, block in (run.get("blocks") or {}).items():
            local: list[str] = list(layer_local)
            checks = _amplitude_checks(block.get("amplitude") or {}, gates, floor_worst, local)
            temporal = _temporal_checks(
                (block.get("temporal") or {}).get("families") or {},
                gates,
                certification.level,
                local,
                families_required=run_name != "twin",
            )
            passed = not local
            stored = block.get("verdict_pass")
            if stored is not passed:
                local.append(
                    f"stored verdict {stored!r} contradicts the recomputed verdict {passed!r}"
                )
            failures.extend(f"[{run_name}:{block_name}] {reason}" for reason in local)
            run_result["blocks"][block_name] = {
                "passed": passed and stored is passed,
                "failures": local,
                "amplitude": checks,
                "temporal": temporal,
            }
        runs[run_name] = run_result
    byte_identity = _byte_identity_checks(certification.byte_identity, failures)
    graph_replay = {
        name: _graph_leg_checks(name, leg, failures)
        for name, leg in certification.graph_replay.items()
    }
    outcome = _outcome_checks(certification.outcome, failures)
    return {
        "passed": not failures,
        "failures": failures,
        "gates": gates,
        "runs": runs,
        "byte_identity": byte_identity,
        "graph_replay": graph_replay,
        "outcome": outcome,
    }


def row_verdict(row: Mapping[str, Any], gate_policy: GatePolicy) -> dict[str, Any]:
    """Recompute a platform row's certified verdicts from its stored metrics alone.

    Each certified block is judged like a certification block: amplitude against the
    effective gates with the worst-episode bar from the row's ``floor_paired_worst`` (the
    platform floor's worst paired error), temporal families at the row's level. A block
    without a stored verdict, a stored verdict its metrics contradict and an admitted served
    mode whose block does not pass are failures; a row without ``admitted_modes`` admits every
    certified block, so every block must pass.
    """
    gates = gate_policy.effective()
    failures: list[str] = []
    prefix = str(row.get("row_sha256") or "")[:8]
    floor_local: list[str] = []
    floor_worst = _number(row, "floor_paired_worst", floor_local)
    failures.extend(f"[row {prefix}] {reason}" for reason in floor_local)
    blocks: dict[str, Any] = {}
    for mode, block in (row.get("certified") or {}).items():
        body = block if isinstance(block, Mapping) else {}
        local: list[str] = []
        amplitude = _amplitude_checks(body.get("amplitude") or {}, gates, floor_worst, local)
        temporal = _temporal_checks(
            (body.get("temporal") or {}).get("families") or {},
            gates,
            str(row.get("level")),
            local,
        )
        passed = not local
        stored = body.get("passed")
        # An honest failing block (stored false) is not a record failure unless admitted; a
        # missing or contradicted verdict is, and the failing checks are named beside it.
        problems: list[str] = []
        if not isinstance(stored, bool):
            problems = [*local, f"stored verdict {stored!r} is not a boolean"]
        elif stored is not passed:
            problems = [
                *local,
                f"stored verdict {stored!r} contradicts the recomputed verdict {passed!r}",
            ]
        failures.extend(f"[row {prefix}:{mode}] {reason}" for reason in problems)
        blocks[mode] = {
            "passed": passed and stored is True,
            "failures": local,
            "amplitude": amplitude,
            "temporal": temporal,
        }
    for mode in admitted_served_modes(row):
        block = blocks.get(mode) or {"failures": ["no certified block"]}
        if not block.get("passed"):
            failures.append(
                f"[row {prefix}:{mode}] admitted served mode does not pass its gates: "
                + "; ".join(block["failures"])
            )
    return {"passed": not failures, "failures": failures, "gates": gates, "blocks": blocks}


def _amplitude_checks(
    amplitude: Mapping[str, Any],
    gates: Mapping[str, Any],
    floor_worst: float | None,
    local: list[str],
) -> dict[str, Any]:
    if "rs_increment_mean_floor" in gates:
        return _range_scaled_amplitude_checks(amplitude, gates, local)
    checks: dict[str, Any] = {}
    checks["paired_mean"] = _at_most(
        "paired_mean", _number(amplitude, "paired_mean", local), gates.get("paired_floor"), local
    )
    if gates.get("paired_floor") is None:
        local.append("gates.paired_floor is required")
    checks["paired_p90"] = _at_most(
        "paired_p90", _number(amplitude, "paired_p90", local), gates.get("p90_floor"), local
    )
    checks["increment_pp_vs_floor"] = _at_most(
        "increment_pp_vs_floor",
        _number(amplitude, "increment_pp_vs_floor", local),
        gates.get("increment_floor_pp"),
        local,
    )
    if gates.get("increment_floor_pp") is None:
        local.append("gates.increment_floor_pp is required")
    ratio = gates.get("worst_ratio_floor")
    absolute = gates.get("worst_abs_floor")
    if floor_worst is None or ratio is None or absolute is None:
        local.append(
            "worst-episode bar needs floor_paired_worst, worst_ratio_floor and worst_abs_floor"
        )
        worst_bar = None
    else:
        worst_bar = max(float(ratio) * floor_worst, float(absolute))
    checks["paired_worst"] = _at_most(
        "paired_worst", _number(amplitude, "paired_worst", local), worst_bar, local
    )
    if worst_bar is None:
        checks["paired_worst"]["passed"] = False
    nan_inf_pass = amplitude.get("nan_inf_pass")
    checks["nan_inf_pass"] = {"value": nan_inf_pass, "passed": nan_inf_pass is True}
    if nan_inf_pass is not True:
        local.append("nan_inf_pass is not true")
    return checks


def _range_scaled_amplitude_checks(
    amplitude: Mapping[str, Any], gates: Mapping[str, Any], local: list[str]
) -> dict[str, Any]:
    """The gates-v2 arms: paired amplitude in % of the action half-range against the ratified bars."""
    checks: dict[str, Any] = {}
    if amplitude.get("rs_unit") != RS_UNIT:
        local.append(f"rs_unit must be {RS_UNIT!r}, got {amplitude.get('rs_unit')!r}")
    for key, gate in RS_AMPLITUDE_GATES:
        checks[key] = _at_most(key, _number(amplitude, key, local), gates.get(gate), local)
        if gates.get(gate) is None:
            local.append(f"gates.{gate} is required")
    nan_inf_pass = amplitude.get("nan_inf_pass")
    checks["nan_inf_pass"] = {"value": nan_inf_pass, "passed": nan_inf_pass is True}
    if nan_inf_pass is not True:
        local.append("nan_inf_pass is not true")
    return checks


def _temporal_checks(
    families: Mapping[str, Any],
    gates: Mapping[str, Any],
    level: str,
    local: list[str],
    *,
    families_required: bool = True,
) -> dict[str, Any]:
    if level == "amplitude":
        if families:
            local.append("an amplitude-level record carries temporal families")
        return {"applied": False, "families": {}}
    if not families and not families_required:
        return {"applied": False, "families": {}}
    required = {
        name
        for name, body in families.items()
        if isinstance(body, Mapping) and body.get("role") == "required"
    }
    if not required:
        local.append("a temporal-level block needs at least one required replay family")
    results: dict[str, Any] = {}
    for name, family in families.items():
        if not isinstance(family, Mapping):
            local.append(f"temporal family {name!r} is not an object")
            continue
        family_local: list[str] = []
        reversal_bar = gates.get("replay_reversal_floor")
        minimum_events = gates.get("reversal_min_reference_events", 0)
        checks = {
            "post_lpf_jerk_ratio": _at_most(
                "post_lpf_jerk_ratio",
                _number(family, "post_lpf_jerk_ratio", family_local),
                gates.get("replay_jerk_postlpf_floor"),
                family_local,
            ),
            "reversal_ratio_l": _at_most(
                "reversal_ratio_l",
                _number(family, "reversal_ratio_l", family_local),
                reversal_bar,
                family_local,
            ),
            "reversal_ratio_r": _at_most(
                "reversal_ratio_r",
                _number(family, "reversal_ratio_r", family_local),
                reversal_bar,
                family_local,
            ),
            "reference_reversal_events_l": _at_least(
                "reference_reversal_events_l",
                _number(family, "reference_reversal_events_l", family_local),
                minimum_events,
                family_local,
            ),
            "reference_reversal_events_r": _at_least(
                "reference_reversal_events_r",
                _number(family, "reference_reversal_events_r", family_local),
                minimum_events,
                family_local,
            ),
        }
        if gates.get("replay_mismatch_vs_seed_base", True):
            checks["mode_mismatch_rate"] = _at_most(
                "mode_mismatch_rate",
                _number(family, "mode_mismatch_rate", family_local),
                _number(family, "seed_mismatch_base", family_local),
                family_local,
            )
        else:
            checks["mode_mismatch_rate"] = {"applied": False}
        proof = family.get("backend_proof")
        if not isinstance(proof, Mapping):
            family_local.append("backend_proof is required")
            checks["backend_proof"] = {"passed": False}
        else:
            proof_ok = (
                int(proof.get("eager_calls") or 0) == 0
                and int(proof.get("skipped_calls") or 0) == 0
                and int(proof.get("trt_calls") or 0) > 0
                and proof.get("uniform_per_request") is True
            )
            if not proof_ok:
                family_local.append(f"backend proof is not all-TensorRT and uniform: {dict(proof)}")
            checks["backend_proof"] = {**dict(proof), "passed": proof_ok}
        results[name] = {"role": family.get("role"), "passed": not family_local, "checks": checks}
        if name in required:
            local.extend(f"[temporal:{name}] {reason}" for reason in family_local)
    return {"applied": True, "families": results}


def _byte_identity_checks(identity: Mapping[str, Any], failures: list[str]) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for entry in identity.get("files") or []:
        equal = entry.get("cond_sha256") == entry.get("twin_sha256")
        stored = entry.get("equal")
        ok = equal and stored is True
        if stored is not equal:
            failures.append(
                f"byte identity of {entry.get('file')!r}: stored equal={stored!r} contradicts the digests"
            )
        elif not equal:
            failures.append(
                f"byte identity of {entry.get('file')!r}: the cond and twin files differ"
            )
        files[str(entry.get("file"))] = {"equal": equal, "passed": ok}
    for key in ("external_differing", "external_missing"):
        if identity.get(key):
            failures.append(f"byte identity: {key} is not empty ({len(identity[key])} entries)")
    if identity.get("result") != "PASS":
        failures.append(f"byte identity result is {identity.get('result')!r}")
    reuse = identity.get("action_plan_reuse") or {}
    reuse_equal = reuse.get("action_engine_sha256") == reuse.get("floor_action_engine_sha256")
    if reuse.get("equal") is not reuse_equal:
        failures.append("action-plan reuse: stored equal flag contradicts the engine digests")
    elif not reuse_equal:
        failures.append("action-plan reuse: the action engine is not the floor's plan")
    return {"files": files, "action_plan_reuse": {"equal": reuse_equal}}


def _graph_leg_checks(name: str, leg: Mapping[str, Any], failures: list[str]) -> dict[str, Any]:
    identical = leg.get("graph_off_actions_sha256") == leg.get("graph_on_actions_sha256")
    stored = leg.get("bit_identical")
    if stored is not identical:
        failures.append(
            f"graph leg {name!r}: stored bit_identical={stored!r} contradicts the action digests"
        )
    elif not identical:
        failures.append(f"graph leg {name!r}: graph replay is not bit-identical to graph-off")
    served_expected = name == "cond_on"
    if leg.get("served") is not served_expected:
        failures.append(f"graph leg {name!r}: served={leg.get('served')!r} contradicts the leg")
    if served_expected and leg.get("bundle_flag") is not True:
        failures.append(
            "graph leg 'cond_on' needs a bundle that carries the conditioning-cached engine"
        )
    counts = (leg.get("backend_counts") or {}).get("trt")
    if isinstance(counts, Mapping):
        cond_calls = int(counts.get("joint_cond") or 0)
        if served_expected and cond_calls <= 0:
            failures.append("graph leg 'cond_on' served no conditioning-cached calls")
        if not served_expected and cond_calls > 0:
            failures.append("graph leg 'cond_off' served conditioning-cached calls")
    return {"bit_identical": identical, "passed": identical and stored is True}


def _outcome_checks(outcome: Mapping[str, Any], failures: list[str]) -> dict[str, Any]:
    label = outcome.get("label")
    required = OUTCOME_REQUIRED_PASSES.get(str(label))
    if required is None:
        failures.append(f"outcome label {label!r} is not an accepted outcome")
        return {"label": label, "passed": False}
    rows = outcome.get("rows") or {}
    flags: dict[str, bool] = {}
    for row, flag in required:
        value = (rows.get(row) or {}).get(flag) if isinstance(rows.get(row), Mapping) else None
        flags[f"{row}.{flag}"] = value is True
        if value is not True:
            failures.append(f"outcome row {row}.{flag} is {value!r}, not true")
    return {"label": label, "flags": flags, "passed": all(flags.values())}


# ── checkpoint identity ──────────────────────────────────────────────────────────────────


def _blank_field(payload: dict[str, Any], dotted: str) -> None:
    node: Any = payload
    parts = dotted.split(".")
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return
        node = node[part]
    if isinstance(node, dict) and parts[-1] in node:
        node[parts[-1]] = ""


def checkpoint_identity(checkpoint: str | Path, *, checkpoint_source: str = "") -> dict[str, Any]:
    """The path-independent identity of a local checkpoint.

    Weight shards are hashed as bytes (every shard an index file references); the three
    metadata files are hashed as canonical JSON with the audit-repointed dependency paths
    blanked, so a checkpoint relocated on another host keeps its identity and any other
    metadata change is a different checkpoint.
    """
    root = Path(checkpoint).expanduser().resolve(strict=True)
    metadata: dict[str, str] = {}
    for name in CHECKPOINT_METADATA_FILES:
        path = root / name
        if not path.is_file():
            raise GoldenRecordError(f"checkpoint {root} has no {name}")
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            for dotted in CHECKPOINT_PATH_FIELDS[name]:
                _blank_field(payload, dotted)
        metadata[name] = canonical_sha256(payload)
    shard_names: set[str] = set()
    for index in sorted(root.glob("*.index.json")):
        weight_map = json.loads(index.read_text()).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise GoldenRecordError(f"{index} has no weight_map")
        shard_names.update(str(value) for value in weight_map.values())
    if not shard_names:
        raise GoldenRecordError(f"checkpoint {root} has no *.index.json naming its weight shards")
    shards: dict[str, str] = {}
    for name in sorted(shard_names):
        path = root / name
        if not path.is_file():
            raise GoldenRecordError(f"checkpoint {root} is missing the weight shard {name}")
        shards[name] = sha256_file(path)
    return {
        "weight_shards": shards,
        "metadata": metadata,
        "path_normalized_fields": {
            name: list(fields) for name, fields in CHECKPOINT_PATH_FIELDS.items()
        },
        "checkpoint_source": checkpoint_source,
    }


def checkpoint_identity_matches(
    recorded: Mapping[str, Any], actual: Mapping[str, Any]
) -> list[str]:
    """The differences between a record's checkpoint identity and a local checkpoint's (empty = same)."""
    differences: list[str] = []
    if dict(recorded.get("weight_shards") or {}) != dict(actual.get("weight_shards") or {}):
        differences.append("weight shards differ from the record")
    for name in CHECKPOINT_METADATA_FILES:
        if (recorded.get("metadata") or {}).get(name) != (actual.get("metadata") or {}).get(name):
            differences.append(
                f"{name} differs from the record (beyond the repointed dependency paths)"
            )
    return differences


# ── package content ──────────────────────────────────────────────────────────────────────


def verify_package_files(
    root: str | Path, package: Mapping[str, Any], *, allow_extra: bool = False
) -> dict[str, Any]:
    """Every recorded file present with its digest and size; extras refused unless allowed.

    A pulled package directory may carry only the record's files plus the two control files;
    a build root (engines and receipts beside the export half) is checked with
    ``allow_extra=True``.
    """
    base = Path(root)
    files = package["files"]
    failures: list[str] = []
    for relative, entry in files.items():
        path = base / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
            continue
        size = path.stat().st_size
        if size != int(entry["bytes"]):
            failures.append(f"{relative}: {size} bytes, record says {entry['bytes']}")
            continue
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            failures.append(f"{relative}: sha256 {digest} != record {entry['sha256']}")
    extras: list[str] = []
    if not allow_extra:
        recorded = set(files) | set(PACKAGE_CONTROL_FILES)
        recorded_dirs = {
            str(parent)
            for relative in files
            for parent in Path(relative).parents
            if str(parent) != "."
        }
        for path in sorted(base.rglob("*")):
            relative = str(path.relative_to(base))
            if path.is_dir():
                if relative not in recorded_dirs:
                    extras.append(relative + "/")
            elif relative not in recorded:
                extras.append(relative)
        failures.extend(f"unexpected entry {extra}" for extra in extras)
    return {"passed": not failures, "failures": failures, "checked": len(files), "extras": extras}


# ── equivalence policy ───────────────────────────────────────────────────────────────────


def tensor_distances(actual: torch.Tensor, oracle: torch.Tensor) -> dict[str, Any]:
    """Relative RMSE, relative max-abs and cosine of ``actual`` against ``oracle``.

    Both are cast to float32 and accumulated in float64. Two zero-norm tensors are equal
    (distances 0, cosine 1); exactly one zero-norm tensor is a mismatch the entry fails on.
    """
    import torch

    if tuple(actual.shape) != tuple(oracle.shape):
        raise GoldenRecordError(
            f"shape {tuple(actual.shape)} != oracle shape {tuple(oracle.shape)}"
        )
    a = actual.detach().to("cpu").to(torch.float32).to(torch.float64).flatten()
    b = oracle.detach().to("cpu").to(torch.float32).to(torch.float64).flatten()
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        raise GoldenRecordError("non-finite values in the compared tensors")
    if torch.equal(a, b):
        return {"rel_rmse": 0.0, "rel_maxabs": 0.0, "cosine": 1.0, "zero_norm_mismatch": False}
    norm_a = float(torch.linalg.vector_norm(a))
    norm_b = float(torch.linalg.vector_norm(b))
    zero_a = norm_a < EQUIVALENCE_EPSILON
    zero_b = norm_b < EQUIVALENCE_EPSILON
    if zero_a and zero_b:
        return {"rel_rmse": 0.0, "rel_maxabs": 0.0, "cosine": 1.0, "zero_norm_mismatch": False}
    if zero_a != zero_b:
        return {
            "rel_rmse": float("inf"),
            "rel_maxabs": float("inf"),
            "cosine": 0.0,
            "zero_norm_mismatch": True,
        }
    diff = a - b
    rel_rmse = float(torch.linalg.vector_norm(diff)) / max(norm_b, EQUIVALENCE_EPSILON)
    rel_maxabs = float(diff.abs().max()) / max(float(b.abs().max()), EQUIVALENCE_EPSILON)
    cosine = float(torch.dot(a, b)) / (norm_a * norm_b)
    return {
        "rel_rmse": rel_rmse,
        "rel_maxabs": rel_maxabs,
        "cosine": cosine,
        "zero_norm_mismatch": False,
    }


def entry_distances(
    actual: Mapping[str, torch.Tensor],
    oracle: Mapping[str, torch.Tensor],
    *,
    oracle_dtypes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """One entry's aggregate distances: max relative RMSE, max relative max-abs, min cosine.

    Structural failures (a missing tensor, a shape or dtype mismatch, non-finite values, a
    zero-norm mismatch) are listed and fail the entry regardless of the bounds.
    """
    failures: list[str] = []
    tensors: dict[str, Any] = {}
    if set(actual) != set(oracle):
        failures.append(f"tensor keys {sorted(actual)} != oracle keys {sorted(oracle)}")
    rel_rmse = 0.0
    rel_maxabs = 0.0
    cosine = 1.0
    for key in sorted(set(actual) & set(oracle)):
        expected_dtype = (oracle_dtypes or {}).get(key)
        if expected_dtype is not None and str(actual[key].dtype) != expected_dtype:
            failures.append(f"{key}: dtype {actual[key].dtype} != oracle dtype {expected_dtype}")
            continue
        try:
            distances = tensor_distances(actual[key], oracle[key])
        except GoldenRecordError as error:
            failures.append(f"{key}: {error}")
            continue
        if distances["zero_norm_mismatch"]:
            failures.append(f"{key}: exactly one of the tensors is zero")
        tensors[key] = distances
        rel_rmse = max(rel_rmse, distances["rel_rmse"])
        rel_maxabs = max(rel_maxabs, distances["rel_maxabs"])
        cosine = min(cosine, distances["cosine"])
    return {
        "rel_rmse": rel_rmse,
        "rel_maxabs": rel_maxabs,
        "cosine": cosine,
        "tensors": tensors,
        "failures": failures,
    }


def judge_entry(distances: Mapping[str, Any], bounds: Mapping[str, Any]) -> dict[str, Any]:
    """Pass iff within the row's envelope and both tripwires, with no structural failure."""
    failures = list(distances.get("failures") or [])
    checks = {
        "rel_rmse": _at_most(
            "rel_rmse", distances.get("rel_rmse"), bounds.get("envelope"), failures
        ),
        "rel_maxabs": _at_most(
            "rel_maxabs", distances.get("rel_maxabs"), bounds.get("maxabs_bound"), failures
        ),
        "cosine": _at_least(
            "cosine", distances.get("cosine"), bounds.get("cosine_floor"), failures
        ),
    }
    return {"passed": not failures, "failures": failures, "checks": checks}


def envelope(
    build1: Mapping[str, Mapping[str, Any]], build2: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """A platform's per-entry bounds from two independent builds' distances to the canonical outputs.

    ``envelope = max(0.005, 3 x max(r1, r2))`` on relative RMSE,
    ``maxabs_bound = max(0.02, 3 x max(m1, m2))`` on relative max-abs and
    ``cosine_floor = 1 - max(1e-4, 3 x max(1 - c1, 1 - c2))``.
    """
    if set(build1) != set(build2):
        raise GoldenRecordError(
            f"the two builds cover different entries: {sorted(build1)} vs {sorted(build2)}"
        )
    bounds: dict[str, Any] = {}
    for name in sorted(build1):
        first, second = build1[name], build2[name]
        for build in (first, second):
            for key in ("rel_rmse", "rel_maxabs", "cosine"):
                _require_finite(build.get(key), f"entry {name!r}.{key}")
        bounds[name] = {
            "envelope": max(
                REL_RMSE_FLOOR, ENVELOPE_ARM * max(first["rel_rmse"], second["rel_rmse"])
            ),
            "maxabs_bound": max(
                REL_MAXABS_FLOOR, ENVELOPE_ARM * max(first["rel_maxabs"], second["rel_maxabs"])
            ),
            "cosine_floor": 1.0
            - max(
                COSINE_SLACK_FLOOR,
                ENVELOPE_ARM * max(1.0 - first["cosine"], 1.0 - second["cosine"]),
            ),
            "builds": [
                {key: float(first[key]) for key in ("rel_rmse", "rel_maxabs", "cosine")},
                {key: float(second[key]) for key in ("rel_rmse", "rel_maxabs", "cosine")},
            ],
        }
    return bounds


__all__ = [
    "ADMITTED_MODES_FIELD",
    "BLOCK_NAMES",
    "BUILD_SETTINGS_FIELDS",
    "CERTIFICATION_LEVELS",
    "CERTIFIED_RUNS",
    "CHECKPOINT_METADATA_FILES",
    "CHECKPOINT_PATH_FIELDS",
    "COND_CACHED_ROLE",
    "EQUIVALENCE_POLICY",
    "GOLDEN_OUTPUTS_DIR",
    "GOLDEN_OUTPUTS_MANIFEST",
    "GOLDEN_OUTPUTS_SCHEMA",
    "GOLDEN_RECORD_SCHEMA",
    "GOLDEN_SIDECAR_FILENAME",
    "GOLDEN_SIDECAR_MODES",
    "GOLDEN_SIDECAR_SCHEMA",
    "GRAPH_LEGS",
    "KNOWN_ROLES",
    "OUTCOME_REQUIRED_PASSES",
    "PACKAGE_CONTROL_FILES",
    "RATIFIED_GATES",
    "RATIFIED_GATES_BY_VERSION",
    "RATIFIED_GATES_V2",
    "RATIFIED_GATES_VERSION",
    "RATIFIED_GATES_VERSION_V2",
    "RECORDS_DIR",
    "REQUIRED_ROLES",
    "RS_AMPLITUDE_GATES",
    "RS_UNIT",
    "SERVED_MODES",
    "Certification",
    "GatePolicy",
    "GoldenRecord",
    "GoldenRecordError",
    "admitted_served_modes",
    "canonical_sha256",
    "checkpoint_identity",
    "checkpoint_identity_matches",
    "entry_distances",
    "envelope",
    "envelope_matches",
    "judge_entry",
    "list_records",
    "load_record",
    "package_sha256",
    "read_golden_sidecar",
    "read_record",
    "row_sha256",
    "runtime_envelope_identity",
    "served_mode_name",
    "sha256_file",
    "tensor_distances",
    "validate_admission_row",
    "validate_record",
    "verdict",
    "verify_package_files",
    "write_golden_sidecar",
    "write_record",
]
