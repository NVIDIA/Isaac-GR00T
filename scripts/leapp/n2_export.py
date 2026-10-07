#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# ruff: noqa: T201
"""Annotate and export the original fixed-language N2 flat-policy inference path.

Runtime inputs are the same unbatched numeric camera/state arrays accepted by
Gr00tFlatPolicyWrapper. The instruction remains a compile-time constant. Codec
roundtrips, network serialization, validation, and robot safety remain outside
the exported tensor graph. Relative EEF outputs stop before pose composition;
the deployment application must supply that postprocessing.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import leapp
from leapp import InputKindEnum, OutputKindEnum, TensorSemantics, annotate
import numpy as np
import onnx
import onnxruntime as ort
import torch

from groot.core.data.schema import EmbodimentTag
from groot.core.policy.gr00t_policy import Gr00tFlatPolicyWrapper
from groot.core.policy.n2_policy import Gr00tN2Policy
import groot.core.training.models.n2  # noqa: F401
from groot.core.training.models.n2.utils.cosmos_model_deps import ensure_cosmos_model_deps

try:
    from .n2_export_patch import (  # noqa: TID252
        annotate_action_noise_for_leapp,
        n2_function_patches,
        partial_eef_actions_for_leapp,
        patch_n2_for_leapp,
    )
except ImportError:
    from n2_export_patch import (
        annotate_action_noise_for_leapp,
        n2_function_patches,
        partial_eef_actions_for_leapp,
        patch_n2_for_leapp,
    )

DEFAULT_OUTPUT_DIR = Path("n2_leapp_export")
DEFAULT_EMBODIMENT = "unitree_g1_whole_body_teleop_latent"
DEFAULT_INSTRUCTION = "Pick up the red object from the table."
GRAPH_NAME = "n2_get_actions"
NODE_NAME = "n2_get_actions"


def load_joint_names(path: Path | None) -> dict[str, list[str]]:
    """Load optional tensor-to-joint-name metadata from a JSON sidecar."""
    if path is None:
        return {}

    with path.open(encoding="utf-8") as stream:
        joint_names = json.load(stream)
    if not isinstance(joint_names, dict) or not all(
        isinstance(tensor_name, str)
        and isinstance(names, list)
        and names
        and all(isinstance(name, str) and name for name in names)
        for tensor_name, names in joint_names.items()
    ):
        raise ValueError(
            "--joint-names must contain a JSON object mapping tensor names "
            "to non-empty lists of joint names"
        )
    return joint_names


def _element_names(
    tensor_name: str,
    value: Any,
    joint_names: dict[str, list[str]],
) -> list[str] | None:
    """Return and validate optional names for a tensor's feature axis."""
    names = joint_names.get(tensor_name)
    if names is not None and len(names) != value.shape[-1]:
        raise ValueError(
            f"Joint-name count for {tensor_name!r} is {len(names)}, "
            f"but its feature dimension is {value.shape[-1]}"
        )
    return names


def _input_kind(
    input_name: str,
    joint_position_kind: Any,
    joint_names: dict[str, list[str]],
) -> Any:
    """Map only unambiguous observation tensors to deployment semantics."""
    if input_name.startswith("video."):
        return "state/camera/image"
    if input_name == "state.projected_gravity":
        return "state/vector3d"
    if input_name in joint_names or input_name.endswith("_joints"):
        return joint_position_kind
    return None


def _output_kind(
    action_name: str,
    joint_position_kind: Any,
    joint_names: dict[str, list[str]],
) -> Any:
    """Map N2 action groups to deployment semantics understood by Isaac ROS."""
    if action_name in joint_names or action_name.endswith("_joints"):
        return joint_position_kind
    if action_name == "action.navigate_command":
        return "velocity_command"
    if action_name == "action.base_height_command":
        return "target/base/height"
    return None


def _output_element_names(
    action_name: str,
    value: Any,
    joint_names: dict[str, list[str]],
) -> list[str] | None:
    """Return names for the feature axis of each exported action tensor."""
    names = _element_names(action_name, value, joint_names)
    if names is not None:
        return names
    if action_name == "action.navigate_command":
        return ["linear_x", "linear_y", "angular_z"]
    if action_name == "action.base_height_command":
        return ["height"]
    return None


def _restore_fixed_batch_dimension(value: Any) -> Any:
    """Restore the batch-of-one axis removed by Gr00tFlatPolicyWrapper."""
    return value[None] if value.ndim == 2 else value


def load_flat_policy(
    checkpoint: Path,
    device: torch.device,
    *,
    model_deps: Path | None,
    embodiment: str,
    num_action_steps: int,
    num_video_steps: int,
) -> Any:
    """Load the production N2 policy and its production flat-observation adapter."""
    # Keep the GPU-only dependency check out of CPU-only module collection.
    import flash_attn  # noqa: F401

    checkpoint = Path(ensure_cosmos_model_deps(checkpoint, cache_dir=model_deps))
    try:
        embodiment_tag = EmbodimentTag(embodiment)
    except ValueError as exc:
        raise ValueError(f"Unknown embodiment tag: {embodiment!r}") from exc

    policy = Gr00tN2Policy(
        embodiment_tag,
        str(checkpoint),
        num_inference_timesteps=num_action_steps,
        shift=5.0,
        seed=42,
        device=str(device),
        strict=False,
        codec_roundtrip="none",
    )
    policy.num_video_steps = num_video_steps
    policy.joint_denoise = True
    policy.kv_cache_cond = True

    pattern_config = getattr(policy.model, "attention_mask_pattern", None)
    if pattern_config is not None and len(pattern_config) != 1:
        raise ValueError("N2 edge export requires one deterministic attention-mask pattern")
    return Gr00tFlatPolicyWrapper(policy, strict=False)


def make_dummy_observation(
    flat_policy: Any,
    *,
    instruction: str,
    input_height: int,
    input_width: int,
) -> dict[str, Any]:
    """Create one plausible observation using the checkpoint's production contract."""
    policy = flat_policy.policy
    modality_configs = policy.get_modality_config()
    observation: dict[str, Any] = {}

    # Video inputs: repeat a synthetic tabletop frame for each required timestep.
    frame = np.zeros((input_height, input_width, 3), dtype=np.uint8)
    frame[: input_height // 2] = (185, 205, 225)
    frame[input_height // 2 :] = (105, 95, 85)
    tabletop_top = int(input_height * 0.62)
    frame[tabletop_top : tabletop_top + 24, input_width // 6 : 5 * input_width // 6] = (
        125,
        82,
        45,
    )
    frame[
        tabletop_top - max(24, input_height // 9) : tabletop_top,
        input_width // 2 - 16 : input_width // 2 + 16,
    ] = (220, 35, 35)
    video_config = modality_configs["video"]
    for key in video_config.modality_keys:
        observation[f"video.{key}"] = np.repeat(
            frame[None],
            len(video_config.delta_indices),
            axis=0,
        )

    # State inputs: use each checkpoint modality's normalization mean.
    state_processor = policy.processor.state_action_processor
    state_config = modality_configs["state"]
    state_steps = len(state_config.delta_indices)
    for key in state_config.modality_keys:
        mean = np.asarray(
            state_processor.norm_params[policy.embodiment_tag.value]["state"][key]["mean"],
            dtype=np.float32,
        )
        if mean.ndim == 1:
            mean = np.repeat(mean[None], state_steps, axis=0)
        if mean.ndim != 2 or mean.shape[0] < state_steps:
            raise ValueError(
                f"Normalization mean for state.{key} has shape {mean.shape}; "
                f"expected enough rows for {state_steps} timesteps"
            )
        observation[f"state.{key}"] = mean[:state_steps].copy()

    # Language input: keep the instruction constant for this exported graph.
    observation[policy.language_key] = instruction
    return observation


def run_annotated_get_action(
    flat_policy: Any,
    observation: dict[str, Any],
    annotate: Any,
    *,
    opset_version: int = 22,
    joint_names: dict[str, list[str]] | None = None,
    relative_eef_outputs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Annotate get_action, distinguishing any partially decoded EEF outputs."""
    joint_names = joint_names or {}
    relative_eef_outputs = relative_eef_outputs or {}
    eef_states = {info["reference_state"] for info in relative_eef_outputs.values()}
    mislabeled = joint_names.keys() & (relative_eef_outputs.keys() | eef_states)
    if mislabeled:
        raise ValueError(
            f"--joint-names cannot label EEF pose tensors as joints: {sorted(mislabeled)}"
        )
    numeric_inputs = {
        name: value
        for name, value in observation.items()
        if isinstance(value, np.ndarray | torch.Tensor)
    }
    annotation_inputs = [
        TensorSemantics(
            name=name.replace(".", "_"),
            ref=value,
            kind=(
                None
                if name in eef_states
                else _input_kind(name, InputKindEnum.JOINT_POSITION, joint_names)
            ),
            element_names=_element_names(name, value, joint_names),
        )
        for name, value in numeric_inputs.items()
    ]
    traced = annotate.input_tensors(NODE_NAME, annotation_inputs)
    traced_values = (traced,) if len(numeric_inputs) == 1 else tuple(traced)

    traced_observation = dict(observation)
    traced_observation.update(dict(zip(numeric_inputs, traced_values, strict=True)))
    for name in numeric_inputs:
        value = traced_observation[name]
        if name.startswith("video.") and value.dtype == np.uint8:
            # TensorRT accepts uint8 network inputs only when they are cast before
            # any other operation. Keep the public camera contract while making
            # all image preprocessing inside the graph float32.
            traced_observation[name] = value.astype(np.float32)
    actions, _ = flat_policy.get_action(traced_observation)
    # Some absolute action groups can be compile-time constants. LEAPP requires
    # every declared output to retain trace provenance, so attach only a
    # zero-valued trace dependency; the arm position anchoring itself happens in
    # relative_joint_action_to_absolute_for_leapp.
    trace_anchor = traced_observation[next(iter(numeric_inputs))].reshape(-1)[:1] * 0.0
    print("ACTION_OUTPUT_TYPES", {name: type(value).__name__ for name, value in actions.items()})
    # Gr00tN2Policy.decode_action has already denormalized actions and converted
    # relative joint deltas to absolute positions using the current joint state.
    # Partial EEF outputs remain relative and require external pose composition.
    # Gr00tFlatPolicyWrapper removes its fixed batch dimension; restore it so
    # deployment consumers receive the unambiguous [1, horizon, features]
    # trajectory contract.
    actions = {
        name: _restore_fixed_batch_dimension(trace_anchor + value)
        for name, value in actions.items()
    }
    for name in relative_eef_outputs:
        if name not in actions or actions[name].shape[-1] != 9:
            raise ValueError(f"Expected partial EEF output {name!r} with xyz + rot6d (9 features)")
    unknown_joint_tensors = joint_names.keys() - (numeric_inputs.keys() | actions.keys())
    if unknown_joint_tensors:
        unknown = ", ".join(sorted(unknown_joint_tensors))
        raise ValueError(f"--joint-names contains tensors not used by this export: {unknown}")
    print(
        "TRACEABLE_ACTION_OUTPUT_TYPES",
        {name: type(value).__name__ for name, value in actions.items()},
    )
    semantic_actions = [
        TensorSemantics(
            name=name,
            ref=value,
            kind=(
                None
                if name in relative_eef_outputs
                else _output_kind(name, OutputKindEnum.JOINT_POSITION, joint_names)
            ),
            element_names=(
                relative_eef_outputs[name]["element_names"]
                if name in relative_eef_outputs
                else _output_element_names(name, value, joint_names)
            ),
        )
        for name, value in actions.items()
    ]
    annotate.output_tensors(
        NODE_NAME,
        semantic_actions,
        export_with="onnx",
        backend_params={"opset_version": opset_version},
    )
    return actions


def export_n2(
    checkpoint: Path,
    output_dir: Path,
    *,
    model_deps: Path | None,
    embodiment: str,
    instruction: str,
    input_height: int,
    input_width: int,
    num_action_steps: int,
    num_video_steps: int,
    opset_version: int,
    validate_runtime: bool,
    validation_rtol: float,
    validation_atol: float,
    joint_names: dict[str, list[str]] | None = None,
) -> Path:
    """Export flat-policy inference, omitting relative EEF pose composition."""
    device = torch.device("cuda:0")
    flat_policy = load_flat_policy(
        checkpoint,
        device,
        model_deps=model_deps,
        embodiment=embodiment,
        num_action_steps=num_action_steps,
        num_video_steps=num_video_steps,
    )
    observation = make_dummy_observation(
        flat_policy,
        instruction=instruction,
        input_height=input_height,
        input_width=input_width,
    )

    original_randn = torch.randn
    with (
        patch_n2_for_leapp(flat_policy.policy.model),
        partial_eef_actions_for_leapp(
            flat_policy.policy.processor.state_action_processor,
            flat_policy.policy.embodiment_tag.value,
        ) as relative_eef_outputs,
    ):
        if relative_eef_outputs:
            logging.getLogger(__name__).warning(
                "PARTIAL EEF EXPORT: %s remain denormalized relative xyz + rot6d predictions, "
                "NOT absolute pose targets. Implement pose conversion in the deployment "
                "application. See relative_eef_outputs.json in the exported bundle.",
                ", ".join(relative_eef_outputs),
            )
        leapp.start(
            name=GRAPH_NAME,
            save_path=str(output_dir),
            max_cached_io=2,
            patching=n2_function_patches(),
        )
        try:
            with annotate_action_noise_for_leapp(
                flat_policy.policy.model,
                annotate,
                NODE_NAME,
                original_randn,
            ):
                run_annotated_get_action(
                    flat_policy,
                    observation,
                    annotate,
                    opset_version=opset_version,
                    joint_names=joint_names,
                    relative_eef_outputs=relative_eef_outputs,
                )
        finally:
            leapp.stop()

    has_cuda_ort = "CUDAExecutionProvider" in ort.get_available_providers()
    run_validation = has_cuda_ort and validate_runtime
    leapp.compile_graph(
        visualize=True,
        validate=run_validation,
        strict=True,
        rtol=validation_rtol,
        atol=validation_atol,
    )
    onnx_path = output_dir / GRAPH_NAME / f"{GRAPH_NAME}.onnx"
    onnx.checker.check_model(str(onnx_path))
    if relative_eef_outputs:
        (onnx_path.parent / "relative_eef_outputs.json").write_text(
            json.dumps(
                {
                    "end_to_end": False,
                    "omitted_postprocessing": "relative EEF to absolute pose conversion",
                    "outputs": relative_eef_outputs,
                },
                indent=2,
            )
            + "\n"
        )
    if not run_validation:
        reason = (
            "ONNX Runtime has no CUDAExecutionProvider"
            if not has_cuda_ort
            else "--skip-runtime-validation was specified"
        )
        print(f"ONNX schema validation passed; numerical replay was skipped because {reason}.")

    return output_dir / GRAPH_NAME / f"{GRAPH_NAME}.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-deps",
        type=Path,
        default=None,
        help="Local root containing cosmos25/ and Wan2.1-I2V-14B-480P/.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--embodiment", default=DEFAULT_EMBODIMENT)
    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help="Compile-time instruction; it is not an ONNX input.",
    )
    parser.add_argument("--input-height", type=int, default=360)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--denoise-steps", type=int, default=1)
    parser.add_argument("--video-denoise-steps", type=int, default=1)
    parser.add_argument("--opset-version", type=int, default=22)
    parser.add_argument(
        "--joint-names",
        type=Path,
        default=None,
        help="Optional JSON mapping flat tensor names to ordered joint names.",
    )
    parser.add_argument("--skip-runtime-validation", action="store_true")
    parser.add_argument("--validation-rtol", type=float, default=2e-2)
    parser.add_argument("--validation-atol", type=float, default=5e-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"--checkpoint must be an existing directory: {checkpoint}")
    if args.denoise_steps < 1 or args.video_denoise_steps < 1:
        raise ValueError("Denoising step counts must be positive")
    if args.denoise_steps % args.video_denoise_steps:
        raise ValueError("--denoise-steps must be divisible by --video-denoise-steps")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else DEFAULT_OUTPUT_DIR.resolve()
    )
    artifact = export_n2(
        checkpoint,
        output_dir,
        model_deps=(
            args.model_deps.expanduser().resolve() if args.model_deps is not None else None
        ),
        embodiment=args.embodiment,
        instruction=args.instruction,
        input_height=args.input_height,
        input_width=args.input_width,
        num_action_steps=args.denoise_steps,
        num_video_steps=args.video_denoise_steps,
        opset_version=args.opset_version,
        validate_runtime=not args.skip_runtime_validation,
        validation_rtol=args.validation_rtol,
        validation_atol=args.validation_atol,
        joint_names=load_joint_names(
            args.joint_names.expanduser().resolve() if args.joint_names is not None else None
        ),
    )
    print(artifact)


if __name__ == "__main__":
    main()
