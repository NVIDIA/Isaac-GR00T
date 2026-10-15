# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Versioned gate evidence that admits quantized N2 TensorRT bundles.

A quantized transformer engine is not a BF16 transcription, so the parity policy
cannot judge it. Admission instead requires this receipt: the measured amplitude
and temporal outcomes of the bundle against its BF16 TensorRT floor, the
thresholds they were judged by, the calibration and evaluation draws, the backend
call counts proving the engines served every request, and content digests binding
the evidence to exactly these engines, this checkpoint and this contract. Every
verdict is RECOMPUTED from the stored metrics whenever the evidence is read (stored
booleans are never trusted), and the rule set is versioned by digest so a policy
change can never silently re-admit older evidence. A bundle that carries the
conditioning-cached joint engine (``cond_kv_freeze``) is judged by the second schema,
which covers that engine too; the bundle contract selects the schema, never the
evidence itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

GATE_EVIDENCE_SCHEMA = "groot_n2_quantized_gate_evidence_v1"
# A cond_kv_freeze bundle's evidence also covers its conditioning-cached joint engine.
COND_CACHED_GATE_EVIDENCE_SCHEMA = "groot_n2_quantized_gate_evidence_v2"
GATE_EVIDENCE_FILENAME = "n2_trt_gate_evidence.json"
EVIDENCE_ENGINE_ROLES = ("transformer_joint", "transformer_action_cache", "vae_encoder")
COND_CACHED_EVIDENCE_ENGINE_ROLES = (*EVIDENCE_ENGINE_ROLES, "transformer_joint_condcached")
GATE_EVIDENCE_POLICY = {
    "schema": GATE_EVIDENCE_SCHEMA,
    "amplitude": {
        "paired_mean": "<= gates.paired_floor",
        "paired_p90": "<= gates.p90_floor (not applied when null)",
        "increment_pp_vs_floor": "<= gates.increment_floor_pp",
        "paired_worst": "<= max(gates.worst_ratio_floor * floor.paired_worst, gates.worst_abs_floor)",
        "nan_inf_pass": "== true",
        "fp32_layer_outputs": "every role <= gates.max_fp32_layer_outputs",
        "fp8_layer_outputs": "> 0 for every fp8 role, == 0 for every bf16 role",
        "evaluation_episode_ids": "non-empty and disjoint from calibration.episode_ids",
        "engine_calls": (
            "vae >= 1; joint == vae * num_video_steps; action == vae * (num_action_steps - num_video_steps)"
        ),
    },
    "temporal": {
        "required_families": ">= 1 family with role 'required'; every required family passes",
        "post_lpf_jerk_ratio": "<= gates.replay_jerk_postlpf_floor (not applied when null)",
        "reversal_ratio": "L and R <= gates.replay_reversal_floor (not applied when null)",
        "reference_reversal_events": "L and R >= gates.reversal_min_reference_events",
        "mode_mismatch_rate": "<= seed_mismatch_base when gates.replay_mismatch_vs_seed_base",
        "backend_proof": "eager_calls == 0, skipped_calls == 0, trt_calls > 0, uniform_per_request",
    },
    "linkage": {
        "engines": "sha256 per role == the bundle engine sha256",
        "checkpoint_sha256": "== canonical sha256 of the checkpoint file hashes",
        "contract_sha256": "== canonical sha256 of the resolved contract payload",
    },
}
# The conditioning-cached joint engine serves every joint denoising step after the
# first, so the call schedule splits the joint calls between the two joint engines;
# every other rule is applied over the four roles.
COND_CACHED_GATE_EVIDENCE_POLICY = {
    **GATE_EVIDENCE_POLICY,
    "schema": COND_CACHED_GATE_EVIDENCE_SCHEMA,
    "engine_roles": list(COND_CACHED_EVIDENCE_ENGINE_ROLES),
    "amplitude": {
        **GATE_EVIDENCE_POLICY["amplitude"],
        "engine_calls": (
            "vae >= 1; joint == vae; joint_cond == vae * (num_video_steps - 1); "
            "action == vae * (num_action_steps - num_video_steps)"
        ),
    },
}


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


GATE_EVIDENCE_POLICY_SHA256 = _canonical_sha256(GATE_EVIDENCE_POLICY)
COND_CACHED_GATE_EVIDENCE_POLICY_SHA256 = _canonical_sha256(COND_CACHED_GATE_EVIDENCE_POLICY)
# Each schema fixes the engine roles its evidence covers and the rule set it is judged by.
_EVIDENCE_SCHEMAS: dict[str, tuple[tuple[str, ...], str]] = {
    GATE_EVIDENCE_SCHEMA: (EVIDENCE_ENGINE_ROLES, GATE_EVIDENCE_POLICY_SHA256),
    COND_CACHED_GATE_EVIDENCE_SCHEMA: (
        COND_CACHED_EVIDENCE_ENGINE_ROLES,
        COND_CACHED_GATE_EVIDENCE_POLICY_SHA256,
    ),
}


def gate_evidence_policy_for(cond_kv_freeze: bool) -> tuple[str, str]:
    """The evidence schema and policy digest a bundle's contract selects."""
    schema = COND_CACHED_GATE_EVIDENCE_SCHEMA if cond_kv_freeze else GATE_EVIDENCE_SCHEMA
    return schema, _EVIDENCE_SCHEMAS[schema][1]


_REQUIRED_SECTIONS = (
    "recipe",
    "engines",
    "floor",
    "gates",
    "amplitude",
    "temporal",
    "calibration",
    "sources",
    "gpu",
    "produced_by",
)


class GateEvidenceError(RuntimeError):
    """Malformed evidence or recomputed verdicts that do not admit the bundle."""


def _number(section: Mapping[str, Any], key: str, failures: list[str]) -> float | None:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        failures.append(f"{key}: missing or non-finite ({value!r})")
        return None
    return float(value)


def _at_most(name: str, value: float | None, bar: Any, failures: list[str]) -> dict[str, Any]:
    if bar is None:
        return {"value": value, "bar": None, "applied": False}
    if value is None:
        return {"value": None, "bar": bar, "applied": True, "passed": False}
    passed = value <= float(bar)
    if not passed:
        failures.append(f"{name}: {value:g} > {float(bar):g}")
    return {"value": value, "bar": float(bar), "applied": True, "passed": passed}


def _at_least(name: str, value: float | None, bar: Any, failures: list[str]) -> dict[str, Any]:
    if value is None:
        return {"value": None, "bar": bar, "passed": False}
    passed = value >= float(bar)
    if not passed:
        failures.append(f"{name}: {value:g} < {float(bar):g}")
    return {"value": value, "bar": float(bar), "passed": passed}


@dataclass(frozen=True)
class GateEvidence:
    """The gate-evidence receipt of one quantized bundle (see the module docstring)."""

    schema: str
    policy_sha256: str
    checkpoint_sha256: str
    contract_sha256: str
    recipe: dict[str, Any]
    engines: dict[str, dict[str, Any]]
    floor: dict[str, Any]
    gates: dict[str, Any]
    amplitude: dict[str, Any]
    temporal: dict[str, Any]
    calibration: dict[str, Any]
    sources: dict[str, Any]
    gpu: dict[str, Any]
    produced_by: dict[str, Any]

    @classmethod
    def init_from_dict(cls, payload: Mapping[str, Any]) -> GateEvidence:
        if not isinstance(payload, Mapping):
            raise GateEvidenceError("gate evidence must be a JSON object")
        schema = payload.get("schema")
        if not isinstance(schema, str) or schema not in _EVIDENCE_SCHEMAS:
            raise GateEvidenceError(f"unsupported gate evidence schema: {schema!r}")
        roles, policy_sha256 = _EVIDENCE_SCHEMAS[schema]
        if payload.get("policy_sha256") != policy_sha256:
            raise GateEvidenceError(
                "gate evidence was written under another policy digest "
                f"({payload.get('policy_sha256')!r} != {policy_sha256!r})"
            )
        for name in ("checkpoint_sha256", "contract_sha256"):
            if not isinstance(payload.get(name), str) or not payload[name]:
                raise GateEvidenceError(f"gate evidence has no {name}")
        for name in _REQUIRED_SECTIONS:
            if not isinstance(payload.get(name), Mapping):
                raise GateEvidenceError(f"gate evidence has no {name} section")
        engines = {role: dict(record) for role, record in payload["engines"].items()}
        if set(engines) != set(roles):
            raise GateEvidenceError(
                f"gate evidence engines must cover exactly {list(roles)}, got {sorted(engines)}"
            )
        for role, record in engines.items():
            for field in ("relative_path", "sha256", "numeric_format"):
                if not isinstance(record.get(field), str) or not record[field]:
                    raise GateEvidenceError(f"gate evidence engine {role!r} has no {field}")
        return cls(
            schema=str(payload["schema"]),
            policy_sha256=str(payload["policy_sha256"]),
            checkpoint_sha256=str(payload["checkpoint_sha256"]),
            contract_sha256=str(payload["contract_sha256"]),
            recipe=dict(payload["recipe"]),
            engines=engines,
            floor=dict(payload["floor"]),
            gates=dict(payload["gates"]),
            amplitude=dict(payload["amplitude"]),
            temporal=dict(payload["temporal"]),
            calibration=dict(payload["calibration"]),
            sources=dict(payload["sources"]),
            gpu=dict(payload["gpu"]),
            produced_by=dict(payload["produced_by"]),
        )

    @classmethod
    def read(cls, path: str | Path) -> GateEvidence:
        return cls.init_from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        os.replace(temporary, target)

    @property
    def engine_roles(self) -> tuple[str, ...]:
        """The engine roles this evidence's schema covers."""
        return _EVIDENCE_SCHEMAS[self.schema][0]

    @property
    def engine_precisions(self) -> dict[str, str]:
        return {role: record["numeric_format"] for role, record in self.engines.items()}

    def recompute(
        self,
        *,
        engine_sha256: Mapping[str, str],
        checkpoint_sha256: str,
        contract_sha256: str,
    ) -> dict[str, Any]:
        """Re-derive every verdict from the stored metrics and the caller's identities."""
        failures: list[str] = []
        linkage = self._recompute_linkage(
            engine_sha256, checkpoint_sha256, contract_sha256, failures
        )
        amplitude = self._recompute_amplitude(failures)
        temporal = self._recompute_temporal(failures)
        return {
            "policy_sha256": _EVIDENCE_SCHEMAS[self.schema][1],
            "passed": not failures,
            "failures": failures,
            "linkage": linkage,
            "amplitude": amplitude,
            "temporal": temporal,
        }

    def require(
        self,
        *,
        engine_sha256: Mapping[str, str],
        checkpoint_sha256: str,
        contract_sha256: str,
    ) -> dict[str, Any]:
        """``recompute`` that fails closed."""
        verdict = self.recompute(
            engine_sha256=engine_sha256,
            checkpoint_sha256=checkpoint_sha256,
            contract_sha256=contract_sha256,
        )
        if not verdict["passed"]:
            raise GateEvidenceError(
                "quantized gate evidence does not admit the bundle: "
                + "; ".join(verdict["failures"])
            )
        return verdict

    def _recompute_linkage(
        self,
        engine_sha256: Mapping[str, str],
        checkpoint_sha256: str,
        contract_sha256: str,
        failures: list[str],
    ) -> dict[str, Any]:
        engines = {}
        for role in self.engine_roles:
            recorded = self.engines[role]["sha256"]
            actual = engine_sha256.get(role)
            engines[role] = {"evidence": recorded, "bundle": actual, "passed": recorded == actual}
            if recorded != actual:
                failures.append(f"linkage: {role} sha256 {recorded} != bundle {actual}")
        checkpoint_ok = self.checkpoint_sha256 == checkpoint_sha256
        if not checkpoint_ok:
            failures.append("linkage: checkpoint_sha256 does not match the measured checkpoint")
        contract_ok = self.contract_sha256 == contract_sha256
        if not contract_ok:
            failures.append("linkage: contract_sha256 does not match the resolved contract")
        return {"engines": engines, "checkpoint": checkpoint_ok, "contract": contract_ok}

    def _recompute_amplitude(self, failures: list[str]) -> dict[str, Any]:
        amplitude = self.amplitude
        gates = self.gates
        checks: dict[str, Any] = {}
        checks["paired_mean"] = _at_most(
            "paired_mean",
            _number(amplitude, "paired_mean", failures),
            gates.get("paired_floor"),
            failures,
        )
        if gates.get("paired_floor") is None:
            failures.append("gates.paired_floor is required")
        checks["paired_p90"] = _at_most(
            "paired_p90",
            _number(amplitude, "paired_p90", failures),
            gates.get("p90_floor"),
            failures,
        )
        checks["increment_pp_vs_floor"] = _at_most(
            "increment_pp_vs_floor",
            _number(amplitude, "increment_pp_vs_floor", failures),
            gates.get("increment_floor_pp"),
            failures,
        )
        if gates.get("increment_floor_pp") is None:
            failures.append("gates.increment_floor_pp is required")
        floor_worst = _number(self.floor, "paired_worst", failures)
        ratio = gates.get("worst_ratio_floor")
        absolute = gates.get("worst_abs_floor")
        if floor_worst is None or ratio is None or absolute is None:
            failures.append(
                "worst-episode bar needs floor.paired_worst, gates.worst_ratio_floor, gates.worst_abs_floor"
            )
            worst_bar = None
        else:
            worst_bar = max(float(ratio) * floor_worst, float(absolute))
        checks["paired_worst"] = _at_most(
            "paired_worst", _number(amplitude, "paired_worst", failures), worst_bar, failures
        )
        if worst_bar is None:
            checks["paired_worst"]["passed"] = False
        nan_inf_pass = amplitude.get("nan_inf_pass")
        checks["nan_inf_pass"] = {"value": nan_inf_pass, "passed": nan_inf_pass is True}
        if nan_inf_pass is not True:
            failures.append("nan_inf_pass is not true")

        fp32_cap = gates.get("max_fp32_layer_outputs")
        fp32 = amplitude.get("fp32_layer_outputs")
        fp8 = amplitude.get("fp8_layer_outputs")
        if fp32_cap is None or not isinstance(fp32, Mapping) or not isinstance(fp8, Mapping):
            failures.append(
                "fp32/fp8 layer-output counts and gates.max_fp32_layer_outputs are required"
            )
            checks["layer_outputs"] = {"passed": False}
        else:
            layer_checks = {}
            for role in self.engine_roles:
                fp32_count = _number(fp32, role, failures)
                fp8_count = _number(fp8, role, failures)
                expects_fp8 = self.engines[role]["numeric_format"] == "fp8"
                fp32_ok = fp32_count is not None and fp32_count <= float(fp32_cap)
                fp8_ok = fp8_count is not None and (
                    (fp8_count > 0) if expects_fp8 else (fp8_count == 0)
                )
                if not fp32_ok:
                    failures.append(f"{role}: fp32 layer outputs {fp32_count} > cap {fp32_cap}")
                if not fp8_ok:
                    numeric_format = self.engines[role]["numeric_format"]
                    failures.append(
                        f"{role}: {fp8_count} fp8 layer outputs contradict format {numeric_format}"
                    )
                layer_checks[role] = {
                    "fp32": fp32_count,
                    "fp8": fp8_count,
                    "passed": fp32_ok and fp8_ok,
                }
            checks["layer_outputs"] = layer_checks

        evaluation = amplitude.get("evaluation_episode_ids")
        calibration = self.calibration.get("episode_ids")
        have_lists = isinstance(evaluation, list) and isinstance(calibration, list)
        if not have_lists or not evaluation or not calibration:
            failures.append(
                "evaluation and calibration episode id lists are required and non-empty"
            )
            checks["disjoint_draw"] = {"passed": False}
        else:
            overlap = sorted(set(evaluation) & set(calibration))
            if overlap:
                failures.append(f"evaluation episodes overlap the calibration draw: {overlap[:8]}")
            checks["disjoint_draw"] = {
                "evaluation": len(evaluation),
                "calibration": len(calibration),
                "overlap": overlap,
                "passed": not overlap,
            }

        calls = amplitude.get("engine_calls")
        video_steps = amplitude.get("num_video_steps")
        action_steps = amplitude.get("num_action_steps")
        if (
            not isinstance(calls, Mapping)
            or not isinstance(video_steps, int)
            or not isinstance(action_steps, int)
            or video_steps <= 0
            or action_steps < video_steps
        ):
            failures.append("engine_calls with num_video_steps/num_action_steps are required")
            checks["engine_calls"] = {"passed": False}
        else:
            predicts = int(calls.get("vae") or 0)
            if self.schema == COND_CACHED_GATE_EVIDENCE_SCHEMA:
                # The conditioning-cached engine serves every joint step after the first.
                expected = {
                    "joint": predicts,
                    "joint_cond": predicts * (video_steps - 1),
                    "action": predicts * (action_steps - video_steps),
                    "vae": predicts,
                }
            else:
                expected = {
                    "joint": predicts * video_steps,
                    "action": predicts * (action_steps - video_steps),
                    "vae": predicts,
                }
            actual = {key: int(calls.get(key) or 0) for key in expected}
            passed = predicts >= 1 and actual == expected
            if not passed:
                failures.append(
                    f"engine calls {actual} != the {video_steps}v{action_steps}a schedule over {predicts} predicts"
                )
            checks["engine_calls"] = {"actual": actual, "expected": expected, "passed": passed}
        return checks

    def _recompute_temporal(self, failures: list[str]) -> dict[str, Any]:
        gates = self.gates
        families = self.temporal.get("families")
        if not isinstance(families, Mapping):
            failures.append("temporal.families is required")
            return {"passed": False, "families": {}}
        required = {
            name: body
            for name, body in families.items()
            if isinstance(body, Mapping) and body.get("role") == "required"
        }
        if not required:
            failures.append("temporal evidence has no required family")
        results: dict[str, Any] = {}
        for name, family in families.items():
            if not isinstance(family, Mapping):
                failures.append(f"temporal family {name!r} is not an object")
                continue
            local: list[str] = []
            reversal_bar = gates.get("replay_reversal_floor")
            checks = {
                "post_lpf_jerk_ratio": _at_most(
                    "post_lpf_jerk_ratio",
                    _number(family, "post_lpf_jerk_ratio", local),
                    gates.get("replay_jerk_postlpf_floor"),
                    local,
                ),
                "reversal_ratio_l": _at_most(
                    "reversal_ratio_l",
                    _number(family, "reversal_ratio_l", local),
                    reversal_bar,
                    local,
                ),
                "reversal_ratio_r": _at_most(
                    "reversal_ratio_r",
                    _number(family, "reversal_ratio_r", local),
                    reversal_bar,
                    local,
                ),
                "reference_reversal_events_l": _at_least(
                    "reference_reversal_events_l",
                    _number(family, "reference_reversal_events_l", local),
                    gates.get("reversal_min_reference_events", 0),
                    local,
                ),
                "reference_reversal_events_r": _at_least(
                    "reference_reversal_events_r",
                    _number(family, "reference_reversal_events_r", local),
                    gates.get("reversal_min_reference_events", 0),
                    local,
                ),
            }
            if gates.get("replay_mismatch_vs_seed_base", True):
                checks["mode_mismatch_rate"] = _at_most(
                    "mode_mismatch_rate",
                    _number(family, "mode_mismatch_rate", local),
                    _number(family, "seed_mismatch_base", local),
                    local,
                )
            else:
                checks["mode_mismatch_rate"] = {"applied": False}
            proof = family.get("backend_proof")
            if not isinstance(proof, Mapping):
                local.append("backend_proof is required")
                checks["backend_proof"] = {"passed": False}
            else:
                proof_ok = (
                    int(proof.get("eager_calls") or 0) == 0
                    and int(proof.get("skipped_calls") or 0) == 0
                    and int(proof.get("trt_calls") or 0) > 0
                    and proof.get("uniform_per_request") is True
                )
                if not proof_ok:
                    local.append(f"backend proof is not all-TensorRT and uniform: {dict(proof)}")
                checks["backend_proof"] = {**dict(proof), "passed": proof_ok}
            results[name] = {
                "role": family.get("role"),
                "passed": not local,
                "failures": local,
                "checks": checks,
            }
            if name in required:
                failures.extend(f"[temporal:{name}] {reason}" for reason in local)
        required_pass = bool(required) and all(results[name]["passed"] for name in required)
        return {"passed": required_pass, "families": results}


__all__ = [
    "COND_CACHED_EVIDENCE_ENGINE_ROLES",
    "COND_CACHED_GATE_EVIDENCE_POLICY",
    "COND_CACHED_GATE_EVIDENCE_POLICY_SHA256",
    "COND_CACHED_GATE_EVIDENCE_SCHEMA",
    "EVIDENCE_ENGINE_ROLES",
    "GATE_EVIDENCE_FILENAME",
    "GATE_EVIDENCE_POLICY",
    "GATE_EVIDENCE_POLICY_SHA256",
    "GATE_EVIDENCE_SCHEMA",
    "GateEvidence",
    "GateEvidenceError",
    "gate_evidence_policy_for",
]
