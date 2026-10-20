# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Typed fixed-shape N2 TensorRT contract derived from checkpoint metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any

from groot.core.utils.storage.checkpoint import checkpoint_source_path


class N2ModelVariant(str, Enum):
    """Promoted checkpoint execution family."""

    CORE = "n2-core"
    TURBO = "n2-turbo"
    TURBO2 = "n2-turbo2"


class TransformerPrecision(str, Enum):
    """Promoted unquantized transformer precision contract.

    Weights, activations, and exported SDPA inputs use BF16. Numerical behavior
    is admitted by the immutable parity policy and decoded-action gates rather
    than by forcing an FP32 attention island into the strongly typed graph.
    Quantized transformer execution is outside this contract.
    """

    BF16 = "bf16"


def resolve_checkpoint_actor_config(model_config: dict[str, Any]) -> dict[str, Any]:
    """Read the executed N2 actor config without converting the checkpoint.

    Gr00tN2Policy loads the original GrootN2RL weights and unwraps
    ``wrapped_model`` for inference. Metadata consumers must use the same
    actor, while manifests continue hashing the original wrapper and weights.
    Plain configs pass through; each consumer retains its existing validation.
    """
    if model_config.get("model_type") != "GrootN2RL":
        return model_config
    actor = model_config.get("base_model_cfg")
    if not isinstance(actor, dict):
        raise TypeError("GrootN2RL requires a base_model_cfg object")
    if actor.get("model_type") != "GrootN2":
        raise ValueError("GrootN2RL base_model_cfg must describe a GrootN2 actor")
    # Includes the legacy BC aliases normalized by Gr00tN2Policy. Full wrapper
    # config/weight validation still runs in its ordinary pretrained loader.
    actor_modes = {
        "BC",
        "BC_NORMAL_ONLY",
        "BC_CONTINUOUS_WEIGHTED",
        "BC_QUALITY_WEIGHTED",
        "BC_HUMAN_NORMAL_ONLY",
    }
    if model_config.get("loss_mode", "BC_QUALITY_WEIGHTED") not in actor_modes:
        raise ValueError("N2 TensorRT supports only actor-only GrootN2RL checkpoints")
    if model_config.get("residual", False):
        raise ValueError("N2 TensorRT supports only actor-only GrootN2RL without residual policies")
    return actor


def resolve_checkpoint_vae_type(model_config: dict[str, Any]) -> str:
    """Resolve the serialized VAE family with N2's historical default.

    Checkpoints written before ``vae_type`` was introduced are Wan2.1. This is
    the same compatibility rule as ``GrootN2Config.vae_type``; an explicitly
    serialized invalid value still fails closed.
    """
    value = model_config.get("vae_type", "wan21")
    if value not in {"wan21", "wan22"}:
        raise ValueError(f"unsupported vae_type={value!r}")
    return value


@dataclass(frozen=True)
class N2TensorRTContract:
    """All engine-shape inputs needed to build and admit one checkpoint.

    ``checkpoint`` is the local source path and is deliberately absent from the
    canonical payload. The manifest binds content hashes, so identical admitted
    artifacts remain portable between machines without pretending two different
    checkpoint payloads are interchangeable.
    """

    schema_version: int
    checkpoint: str
    model_type: str
    model_size: str
    model_variant: N2ModelVariant
    vae_type: str
    model_dtype: str
    embodiment: str
    camera_names: tuple[str, ...]
    video_indices: tuple[int, ...]
    history_indices: tuple[int, ...]
    image_height: int
    image_width: int
    action_horizon: int
    max_state_dim: int
    max_action_dim: int
    video_denoising_steps: int
    action_denoising_steps: int
    joint_denoise: bool
    conditional_kv_cache: bool
    token_merge_kernel: tuple[int, int, int] | None
    token_merge_layers: tuple[int, int] | None
    transformer_precision: TransformerPrecision
    vae_precision: str

    @property
    def camera_count(self) -> int:
        return len(self.camera_names)

    @property
    def conditioning_frames(self) -> int:
        return len(self.history_indices)

    @property
    def total_video_frames(self) -> int:
        return len(self.video_indices)

    @property
    def conditioning_latent_frames(self) -> int:
        """Temporal VAE tokens produced from the observed video window."""
        return (self.conditioning_frames - 1) // 4 + 1

    @property
    def total_latent_frames(self) -> int:
        """Temporal VAE tokens in the complete conditioned/generated sequence."""
        return (self.total_video_frames - 1) // 4 + 1

    @property
    def image_size(self) -> tuple[int, int]:
        return self.image_height, self.image_width

    @property
    def schedule(self) -> str:
        return f"{self.video_denoising_steps}v{self.action_denoising_steps}a"

    @property
    def latent_shape(self) -> tuple[int, int, int, int]:
        """Per-camera VAE latent geometry for the admitted fixed input."""
        channels, spatial_ratio = {
            "wan21": (16, 8),
            "wan22": (48, 16),
        }[self.vae_type]
        return (
            channels,
            self.conditioning_latent_frames,
            self.image_height // spatial_ratio,
            self.image_width // spatial_ratio,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic, path-independent engine contract."""
        return {
            "schema_version": self.schema_version,
            "model_type": self.model_type,
            "model_size": self.model_size,
            "model_variant": self.model_variant.value,
            "vae_type": self.vae_type,
            "model_dtype": self.model_dtype,
            "embodiment": self.embodiment,
            "camera_names": list(self.camera_names),
            "camera_count": self.camera_count,
            "conditioning_frames": self.conditioning_frames,
            "total_video_frames": self.total_video_frames,
            "video_indices": list(self.video_indices),
            "history_indices": list(self.history_indices),
            "image_size": [self.image_height, self.image_width],
            "action_horizon": self.action_horizon,
            "max_state_dim": self.max_state_dim,
            "max_action_dim": self.max_action_dim,
            "schedule": self.schedule,
            "video_denoising_steps": self.video_denoising_steps,
            "action_denoising_steps": self.action_denoising_steps,
            "joint_denoise": self.joint_denoise,
            "conditional_kv_cache": self.conditional_kv_cache,
            "token_merge_kernel": (
                list(self.token_merge_kernel) if self.token_merge_kernel is not None else None
            ),
            "token_merge_layers": (
                list(self.token_merge_layers) if self.token_merge_layers is not None else None
            ),
            "transformer_precision": self.transformer_precision.value,
            "vae_precision": self.vae_precision,
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        embodiment: str | None = None,
        video_denoising_steps: int,
        action_denoising_steps: int,
        joint_denoise: bool,
        conditional_kv_cache: bool,
    ) -> N2TensorRTContract:
        """Resolve the promoted fixed-shape contract without Torch/TensorRT."""
        checkpoint_path = checkpoint_source_path(Path(checkpoint).expanduser()).resolve(strict=True)
        model = resolve_checkpoint_actor_config(_read_json(checkpoint_path / "config.json"))
        processor = _read_json(checkpoint_path / "processor_config.json")
        model_type = _require_string(model, "model_type")
        if model_type != "GrootN2":
            raise ValueError(f"checkpoint is not GrootN2: {model_type!r}")
        model_size = _require_string(model, "model_size")
        model_dtype = _require_string(model, "model_dtype")
        vae_type = resolve_checkpoint_vae_type(model)

        processor_kwargs = processor.get("processor_kwargs")
        if not isinstance(processor_kwargs, dict):
            raise TypeError("processor_config.json has no processor_kwargs object")
        modality_configs = processor_kwargs.get("modality_configs")
        if not isinstance(modality_configs, dict) or not modality_configs:
            raise ValueError("processor_config.json has no modality_configs")
        resolved_embodiment = _select_embodiment(modality_configs, embodiment)
        modality = modality_configs[resolved_embodiment]
        if not isinstance(modality, dict):
            raise TypeError(f"invalid modality config for {resolved_embodiment!r}")

        video = _require_mapping(modality, "video")
        action = _require_mapping(modality, "action")
        camera_names = _require_unique_strings(video, "modality_keys")
        video_deltas = _require_ints(video, "delta_indices")
        history_indices = tuple(delta for delta in video_deltas if delta <= 0)
        image_size = processor_kwargs.get("image_target_size", model.get("image_target_size"))
        if (
            not isinstance(image_size, list | tuple)
            or len(image_size) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in image_size)
        ):
            raise ValueError(f"invalid image_target_size: {image_size!r}")

        action_deltas = _require_ints(action, "delta_indices")
        action_horizon = len(action_deltas)
        if action_deltas != tuple(range(action_horizon)):
            raise ValueError("TensorRT requires contiguous action delta_indices starting at zero")
        action_cfg = _require_mapping(model, "action_cfg")
        configured_horizon = _require_positive_int(action_cfg, "action_horizon")
        if configured_horizon != action_horizon:
            raise ValueError(
                f"model and processor action horizons disagree: {configured_horizon} != {action_horizon}"
            )
        max_n_cameras = _require_positive_int(model, "max_n_cameras")
        if max_n_cameras != len(camera_names):
            raise ValueError(
                "unsupported N2 TensorRT contract: padded camera views are not supported yet; "
                "model max_n_cameras must equal the processor camera count "
                f"({max_n_cameras} != {len(camera_names)})"
            )

        kernel = _optional_int_tuple(model.get("token_merge_kernel"), "token_merge_kernel", 3)
        layers = _optional_int_tuple(model.get("token_merge_layers"), "token_merge_layers", 2)
        if kernel is None:
            # Layer defaults in historical configs do not enable token merge and
            # therefore do not belong to the effective engine contract.
            layers = None
        if vae_type == "wan22" and kernel is not None:
            raise ValueError("unsupported N2 TensorRT contract: token_merge_vae requires Wan2.1")
        try:
            model_variant = {
                ("wan21", False): N2ModelVariant.CORE,
                ("wan21", True): N2ModelVariant.TURBO2,
                ("wan22", False): N2ModelVariant.TURBO,
            }[(vae_type, kernel is not None)]
        except KeyError as error:
            raise ValueError(
                f"unsupported N2 architecture/token-merge combination: {vae_type!r}, {kernel!r}"
            ) from error

        contract = cls(
            schema_version=3,
            checkpoint=str(checkpoint_path),
            model_type=model_type,
            model_size=model_size,
            model_variant=model_variant,
            vae_type=vae_type,
            model_dtype=model_dtype,
            embodiment=resolved_embodiment,
            camera_names=camera_names,
            video_indices=video_deltas,
            history_indices=history_indices,
            image_height=image_size[0],
            image_width=image_size[1],
            action_horizon=action_horizon,
            max_state_dim=_require_positive_int(action_cfg, "max_state_dim"),
            max_action_dim=_require_positive_int(action_cfg, "max_action_dim"),
            video_denoising_steps=video_denoising_steps,
            action_denoising_steps=action_denoising_steps,
            joint_denoise=joint_denoise,
            conditional_kv_cache=conditional_kv_cache,
            token_merge_kernel=kernel,
            token_merge_layers=layers,
            transformer_precision=TransformerPrecision.BF16,
            vae_precision="bf16",
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        """Admit checkpoint-derived shapes while failing closed on other features."""
        failures: dict[str, Any] = {}
        if self.schema_version != 3:
            failures["schema_version"] = {"actual": self.schema_version, "supported": [3]}
        if self.model_type != "GrootN2":
            failures["model_type"] = {"actual": self.model_type, "supported": ["GrootN2"]}
        if self.model_size != "2B":
            failures["model_size"] = {"actual": self.model_size, "supported": ["2B"]}
        if self.camera_count <= 0:
            failures["camera_count"] = {
                "actual": self.camera_count,
                "requirement": "at least one configured camera",
            }
        steps_are_positive_integers = all(
            isinstance(steps, int) and not isinstance(steps, bool) and steps > 0
            for steps in (self.video_denoising_steps, self.action_denoising_steps)
        )
        if (
            not steps_are_positive_integers
            or self.action_denoising_steps % self.video_denoising_steps != 0
        ):
            failures["schedule"] = {
                "actual": self.schedule,
                "requirement": (
                    "positive integer video/action steps with action steps an exact multiple "
                    "of video steps"
                ),
            }
        if self.model_dtype != "bfloat16":
            failures["model_dtype"] = {"actual": self.model_dtype, "supported": ["bfloat16"]}
        if (
            len(set(self.history_indices)) != len(self.history_indices)
            or tuple(sorted(self.history_indices)) != self.history_indices
            or not self.history_indices
            or self.history_indices[-1] != 0
        ):
            failures["history_indices"] = {
                "actual": self.history_indices,
                "requirement": "unique increasing nonpositive indices ending at zero",
            }
        if (
            len(set(self.video_indices)) != len(self.video_indices)
            or tuple(sorted(self.video_indices)) != self.video_indices
            or not self.video_indices
            or 0 not in self.video_indices
            or tuple(delta for delta in self.video_indices if delta <= 0) != self.history_indices
        ):
            failures["video_indices"] = {
                "actual": self.video_indices,
                "requirement": "unique increasing indices containing zero",
            }
        if self.total_latent_frames <= self.conditioning_latent_frames:
            failures["video_generation_window"] = {
                "conditioning_frames": self.conditioning_frames,
                "total_video_frames": self.total_video_frames,
                "requirement": "at least one generated latent frame",
            }
        spatial_ratio = 16 if self.vae_type == "wan22" else 8
        patch_multiple = spatial_ratio * 2
        merge_height = self.token_merge_kernel[1] if self.token_merge_kernel is not None else 1
        merge_width = self.token_merge_kernel[2] if self.token_merge_kernel is not None else 1
        required_height_multiple = patch_multiple * merge_height
        required_width_multiple = patch_multiple * merge_width
        if (
            self.image_height <= 0
            or self.image_width <= 0
            or self.image_height % required_height_multiple
            or self.image_width % required_width_multiple
        ):
            failures["image_size"] = {
                "actual": self.image_size,
                "requirement": (
                    "positive height/width divisible by "
                    f"{required_height_multiple}/{required_width_multiple}"
                ),
            }
        if (self.max_state_dim, self.max_action_dim) != (128, 128):
            failures["state_action_dimensions"] = {
                "actual": (self.max_state_dim, self.max_action_dim),
                "supported": [(128, 128)],
            }
        if not self.joint_denoise or not self.conditional_kv_cache:
            failures["denoising_mode"] = {
                "joint_denoise": self.joint_denoise,
                "conditional_kv_cache": self.conditional_kv_cache,
                "supported": {"joint_denoise": True, "conditional_kv_cache": True},
            }
        if self.token_merge_kernel is not None:
            if self.vae_type != "wan21":
                failures["token_merge_vae"] = "token merge is qualified only for Wan2.1"
            if self.token_merge_kernel != (1, 2, 2) or self.token_merge_layers != (2, 23):
                failures["token_merge"] = {
                    "actual": [self.token_merge_kernel, self.token_merge_layers],
                    "supported": [[(1, 2, 2), (2, 23)]],
                }
        expected_variant = {
            ("wan21", False): N2ModelVariant.CORE,
            ("wan21", True): N2ModelVariant.TURBO2,
            ("wan22", False): N2ModelVariant.TURBO,
        }.get((self.vae_type, self.token_merge_kernel is not None))
        if self.model_variant is not expected_variant:
            failures["model_variant"] = {
                "actual": self.model_variant.value,
                "expected": expected_variant.value if expected_variant is not None else None,
            }
        if failures:
            raise ValueError(
                "unsupported N2 TensorRT contract (not supported yet): "
                + json.dumps(failures, default=str)
            )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required checkpoint metadata not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint metadata must be an object: {path}")
    return payload


def _select_embodiment(configs: dict[str, Any], requested: str | None) -> str:
    if requested is None:
        if len(configs) != 1:
            raise ValueError(
                f"checkpoint contains multiple embodiments; select one explicitly: {sorted(configs)}"
            )
        return next(iter(configs))
    candidates = {key.lower(): key for key in configs}
    try:
        return candidates[requested.lower()]
    except KeyError as error:
        raise ValueError(f"checkpoint does not contain embodiment {requested!r}") from error


def _require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"modality config has no {key!r} object")
    return value


def _require_string(parent: dict[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _require_positive_int(parent: dict[str, Any], key: str) -> int:
    value = parent.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _require_unique_strings(parent: dict[str, Any], key: str) -> tuple[str, ...]:
    value = parent.get(key)
    if (
        not isinstance(value, list | tuple)
        or not value
        or any(not isinstance(item, str) for item in value)
    ):
        raise ValueError(f"{key} must be a non-empty string sequence")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{key} contains duplicates: {result}")
    return result


def _require_ints(parent: dict[str, Any], key: str) -> tuple[int, ...]:
    value = parent.get(key)
    if not isinstance(value, list | tuple) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise ValueError(f"{key} must be an integer sequence")
    return tuple(value)


def _optional_int_tuple(value: Any, name: str, length: int) -> tuple[int, ...] | None:
    if value is None:
        return None
    if (
        not isinstance(value, list | tuple)
        or len(value) != length
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise ValueError(f"{name} must contain {length} integers")
    return tuple(value)
