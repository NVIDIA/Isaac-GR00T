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

from dataclasses import dataclass, replace
from enum import Enum
import json
from pathlib import Path
from typing import Any

from groot.core.utils.storage.checkpoint import checkpoint_source_path

# One camera-aware format covers all supported precision and VAE variants.
# Feature fields describe the workload; only format changes advance this version.
# Schema 7 and earlier use the former camera fields and require rebuilding.
_CONTRACT_SCHEMA_VERSION = 8


class N2ModelVariant(str, Enum):
    """Promoted checkpoint execution family."""

    CORE = "n2-core"
    TURBO = "n2-turbo"
    TURBO2 = "n2-turbo2"
    TURBO3 = "n2-turbo-v3"


class TransformerPrecision(str, Enum):
    """Promoted transformer precision contract, per transformer engine role.

    BF16: weights, activations, and exported SDPA inputs use BF16. Numerical
    behavior is admitted by the immutable parity policy and decoded-action gates
    rather than by forcing an FP32 attention island into the strongly typed graph.
    FP8: a ModelOpt Q/DQ export (per-tensor FP8 linears; the attention core stays
    BF16). An FP8 engine is not a BF16 transcription, so it is admitted by the
    checkpoint-specific gate-evidence receipt, never by transcription parity.
    """

    BF16 = "bf16"
    FP8 = "fp8"


# Numeric formats the VAE encoder engine may carry. ``bf16`` is the transcription of
# the eager encoder; ``fp8`` is a ModelOpt-calibrated Q/DQ export of its convolutions
# (attention core BF16), admitted like a quantized transformer role by gate evidence
# while its parity gate against the unchanged BF16 eager encoder keeps gating.
VAE_PRECISIONS = ("bf16", "fp8")


# How the cached-action engine receives its language conditioning. ``raw`` feeds the
# text embeddings and projects them inside the engine on every call; ``projected``
# feeds the transformer's BF16 ``crossattn_proj`` output, computed once per prompt by
# the runtime conditioner (``conditioner.py``); ``kv`` additionally feeds every
# block's normalized cross-attention keys and values, so the engine keeps only the
# query, softmax, and output projections of cross-attention. Recorded in the bundle
# build settings; the runtime never infers it from binding names alone.
ACTION_CONTEXT_MODES = ("raw", "projected", "kv")

# Which prompt slots the DiT cross-attention reads (the checkpoint's ``dit_text_keys``).
# ``all`` attends every slot of the right-padded prompt, pads included; ``real`` attends
# each sample's real prompt tokens only, so its engines take the real-token count as an
# extra input. Recorded in the contract only for ``real``: ``all`` payloads are unchanged.
TEXT_KEYS = ("all", "real")
# The engine binding of a ``real`` contract: (B,) int32 real-token count of the padded prompt.
TEXT_SEQ_LEN_INPUT = "text_seq_len"


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

    ``input_camera_names`` lists the real camera inputs in processor order.
    ``total_camera_slot_count`` is the engine capacity, including trailing slots
    filled with masked padding. The input and padded counts are derived from
    those fields, so a three-input, four-slot model has one padded slot.
    """

    schema_version: int
    checkpoint: str
    model_type: str
    model_size: str
    model_variant: N2ModelVariant
    vae_type: str
    model_dtype: str
    embodiment: str
    input_camera_names: tuple[str, ...]
    total_camera_slot_count: int
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
    # Precision of the cached-action engine when it differs from the joint
    # engine's (a mixed bundle); None resolves to ``transformer_precision``.
    action_cache_precision: TransformerPrecision | None = None
    # The bundle also carries the conditioning-cached joint engine (cond_kv_freeze serving).
    cond_kv_freeze: bool = False
    vae_bottleneck_kernel: tuple[int, int, int] | None = None
    # The checkpoint's DiT text keys (``TEXT_KEYS``); ``real`` engines take ``text_seq_len``.
    text_keys: str = "all"

    @property
    def patch_size(self) -> tuple[int, int, int]:
        """Projection patches; the bottleneck does not resize VAE latents."""
        kernel = self.vae_bottleneck_kernel or (1, 1, 1)
        return (kernel[0], 2 * kernel[1], 2 * kernel[2])

    def __post_init__(self) -> None:
        if self.action_cache_precision is None:
            object.__setattr__(self, "action_cache_precision", self.transformer_precision)

    @property
    def input_camera_count(self) -> int:
        """Number of real camera inputs required by the selected embodiment."""
        return len(self.input_camera_names)

    @property
    def padded_camera_slot_count(self) -> int:
        """Number of trailing camera slots filled with masked padding."""
        return self.total_camera_slot_count - self.input_camera_count

    @property
    def is_quantized(self) -> bool:
        return (
            self.transformer_precision is not TransformerPrecision.BF16
            or self.action_cache_precision is not TransformerPrecision.BF16
            or self.vae_precision != "bf16"
        )

    @property
    def precision_label(self) -> str:
        """One string naming the engine precisions (``bf16``, ``fp8``, or per role).

        A quantized VAE appends its own role (``.../fp8-vae``); bundles with a BF16 VAE
        keep the transformer-only labels byte-identical.
        """
        joint = self.transformer_precision.value
        action = self.action_cache_precision.value
        label = joint if joint == action else f"{joint}-joint/{action}-action"
        if self.vae_precision != "bf16":
            label = f"{label}/{self.vae_precision}-vae"
        return label

    @property
    def engine_precisions(self) -> dict[str, str]:
        """Numeric format per manifest engine role."""
        precisions = {
            "transformer_joint": self.transformer_precision.value,
            "transformer_action_cache": self.action_cache_precision.value,
            "vae_encoder": self.vae_precision,
        }
        if self.cond_kv_freeze:
            # The conditioning-cached joint graph shares the joint engine's precision.
            precisions["transformer_joint_condcached"] = self.transformer_precision.value
        return precisions

    def with_transformer_precisions(
        self,
        transformer_precision: TransformerPrecision | str,
        action_cache_precision: TransformerPrecision | str | None = None,
    ) -> N2TensorRTContract:
        """The same checkpoint contract resolved for the given transformer precisions."""
        joint = TransformerPrecision(transformer_precision)
        action = (
            TransformerPrecision(action_cache_precision)
            if action_cache_precision is not None
            else joint
        )
        contract = replace(
            self,
            transformer_precision=joint,
            action_cache_precision=action,
        )
        contract.validate()
        return contract

    def with_vae_precision(self, vae_precision: str) -> N2TensorRTContract:
        """The same checkpoint contract resolved for the given VAE engine precision."""
        if vae_precision not in VAE_PRECISIONS:
            raise ValueError(
                f"unsupported VAE precision {vae_precision!r}; expected one of {VAE_PRECISIONS}"
            )
        contract = replace(self, vae_precision=vae_precision)
        contract.validate()
        return contract

    def with_cond_kv_freeze(self, cond_kv_freeze: bool) -> N2TensorRTContract:
        """The same checkpoint contract resolved for a bundle's engine roles.

        ``cond_kv_freeze`` is a bundle property: whether it carries the conditioning-cached
        joint engine. The mode a runtime serves is selected separately at serve time.
        """
        contract = replace(self, cond_kv_freeze=bool(cond_kv_freeze))
        contract.validate()
        return contract

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
        payload = {
            "schema_version": self.schema_version,
            "model_type": self.model_type,
            "model_size": self.model_size,
            "model_variant": self.model_variant.value,
            "vae_type": self.vae_type,
            "model_dtype": self.model_dtype,
            "embodiment": self.embodiment,
            "input_camera_names": list(self.input_camera_names),
            "input_camera_count": self.input_camera_count,
            "total_camera_slot_count": self.total_camera_slot_count,
            "padded_camera_slot_count": self.padded_camera_slot_count,
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
        # Only bottleneck variants carry the VAE projection geometry.
        if self.vae_bottleneck_kernel is not None:
            payload["vae_bottleneck_kernel"] = list(self.vae_bottleneck_kernel)
            payload["patch_size"] = list(self.patch_size)
        if self.is_quantized:
            payload["action_cache_precision"] = self.action_cache_precision.value
        if self.cond_kv_freeze:
            payload["cond_kv_freeze"] = True
        if self.text_keys != "all":
            payload["text_keys"] = self.text_keys
        return payload

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
        transformer_precision: TransformerPrecision | str = TransformerPrecision.BF16,
        action_cache_precision: TransformerPrecision | str | None = None,
        vae_precision: str = "bf16",
        cond_kv_freeze: bool = False,
    ) -> N2TensorRTContract:
        """Resolve the promoted fixed-shape contract without Torch/TensorRT.

        The engine precisions are bundle properties, not checkpoint properties:
        callers admitting a quantized bundle pass what its receipt declares (per
        transformer role and for the VAE encoder); the default is the unquantized
        BF16 contract.
        """
        joint_precision = TransformerPrecision(transformer_precision)
        action_precision = (
            TransformerPrecision(action_cache_precision)
            if action_cache_precision is not None
            else joint_precision
        )
        if vae_precision not in VAE_PRECISIONS:
            raise ValueError(
                f"unsupported VAE precision {vae_precision!r}; expected one of {VAE_PRECISIONS}"
            )
        checkpoint_path = checkpoint_source_path(Path(checkpoint).expanduser()).resolve(strict=True)
        model = resolve_checkpoint_actor_config(_read_json(checkpoint_path / "config.json"))
        processor = _read_json(checkpoint_path / "processor_config.json")
        model_type = _require_string(model, "model_type")
        if model_type != "GrootN2":
            raise ValueError(f"checkpoint is not GrootN2: {model_type!r}")
        # The engines have no inputs for the tactile/deform token groups.
        tactile = [name for name in ("tactile_cfg", "deform_cfg") if model.get(name) is not None]
        if tactile:
            raise ValueError(f"TensorRT does not support tactile/deform checkpoints ({tactile})")
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
        input_camera_names = _require_unique_strings(video, "modality_keys")
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
        total_camera_slot_count = _require_positive_int(model, "max_n_cameras")
        if total_camera_slot_count < len(input_camera_names):
            raise ValueError(
                "processor camera count exceeds model capacity: "
                f"{len(input_camera_names)} input cameras > {total_camera_slot_count} total camera slots"
            )

        bottleneck = _optional_int_tuple(
            model.get("vae_bottleneck_kernel"), "vae_bottleneck_kernel", 3
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

        if bottleneck is not None:
            model_variant = N2ModelVariant.TURBO3
        contract = cls(
            text_keys=checkpoint_text_keys(model),
            schema_version=_CONTRACT_SCHEMA_VERSION,
            vae_bottleneck_kernel=bottleneck,
            checkpoint=str(checkpoint_path),
            model_type=model_type,
            model_size=model_size,
            model_variant=model_variant,
            vae_type=vae_type,
            model_dtype=model_dtype,
            embodiment=resolved_embodiment,
            input_camera_names=input_camera_names,
            total_camera_slot_count=total_camera_slot_count,
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
            transformer_precision=joint_precision,
            vae_precision=vae_precision,
            action_cache_precision=action_precision,
            cond_kv_freeze=cond_kv_freeze,
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        """Admit checkpoint-derived shapes while failing closed on other features."""
        failures: dict[str, Any] = {}
        if self.schema_version != _CONTRACT_SCHEMA_VERSION:
            failures["schema_version"] = {
                "actual": self.schema_version,
                "supported": [_CONTRACT_SCHEMA_VERSION],
            }
        if self.model_type != "GrootN2":
            failures["model_type"] = {"actual": self.model_type, "supported": ["GrootN2"]}
        if self.model_size != "2B":
            failures["model_size"] = {"actual": self.model_size, "supported": ["2B"]}
        if (
            not isinstance(self.total_camera_slot_count, int)
            or isinstance(self.total_camera_slot_count, bool)
            or not 0 < self.input_camera_count <= self.total_camera_slot_count
        ):
            failures["total_camera_slot_count"] = {
                "actual": self.total_camera_slot_count,
                "input_camera_count": self.input_camera_count,
                "requirement": "a positive integer slot count covering at least one input camera",
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
        if self.vae_precision not in VAE_PRECISIONS:
            failures["vae_precision"] = {
                "actual": self.vae_precision,
                "supported": list(VAE_PRECISIONS),
            }
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
        bottleneck = self.vae_bottleneck_kernel
        if bottleneck is not None and (
            bottleneck != (1, 2, 2)
            or self.vae_type != "wan21"
            or self.token_merge_kernel is not None
        ):
            failures["vae_bottleneck_kernel"] = (
                "Turbo-v3 requires Wan2.1, (1, 2, 2), and no token merge"
            )
        required_height_multiple = (
            patch_multiple * merge_height * (2 if bottleneck is not None else 1)
        )
        required_width_multiple = (
            patch_multiple * merge_width * (2 if bottleneck is not None else 1)
        )
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
        if bottleneck is not None:
            supported_dimensions = [(256, 256)]
        elif self.token_merge_kernel is None:
            # Wan2.2 Turbo-v1 and Wan2.1 Core checkpoints may keep native 256-wide state/action
            # heads without a V3 bottleneck. Token-merge (Turbo2) engines are qualified at
            # 128/128 only.
            supported_dimensions = [(128, 128), (256, 256)]
        else:
            supported_dimensions = [(128, 128)]
        if (self.max_state_dim, self.max_action_dim) not in supported_dimensions:
            failures["state_action_dimensions"] = {
                "actual": (self.max_state_dim, self.max_action_dim),
                "supported": supported_dimensions,
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
        if self.cond_kv_freeze and self.token_merge_kernel is not None:
            failures["cond_kv_freeze"] = (
                "the conditioning-cached joint engine is not qualified with token merge"
            )
        if self.text_keys not in TEXT_KEYS:
            failures["text_keys"] = {"actual": self.text_keys, "supported": list(TEXT_KEYS)}
        expected_variant = {
            ("wan21", False): N2ModelVariant.CORE,
            ("wan21", True): N2ModelVariant.TURBO2,
            ("wan22", False): N2ModelVariant.TURBO,
        }.get((self.vae_type, self.token_merge_kernel is not None))
        if bottleneck is not None:
            expected_variant = N2ModelVariant.TURBO3
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


def checkpoint_text_keys(model_config: dict[str, Any]) -> str:
    """The DiT text keys of a checkpoint config (``TEXT_KEYS``).

    Configs written before the string field carry the boolean ``dit_text_pad_mask``
    (True means ``real``); a config with neither attends every slot (``all``).
    """
    value = model_config.get("dit_text_keys")
    legacy = model_config.get("dit_text_pad_mask")
    if value is None:
        value = "real" if legacy else "all"
    elif legacy is not None and value != ("real" if legacy else "all"):
        raise ValueError(
            f"checkpoint config sets dit_text_keys={value!r} and dit_text_pad_mask={legacy!r}"
        )
    if value not in TEXT_KEYS:
        raise ValueError(f"unsupported dit_text_keys={value!r}; expected one of {TEXT_KEYS}")
    return value


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
