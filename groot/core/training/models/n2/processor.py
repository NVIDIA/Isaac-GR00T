# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""GrootN2 processor: images + text + optional action/state.

Extends the video-only GrootN2Processor with StateActionProcessor integration
for joint video+action training. When action_cfg is None, behaves identically
to the video-only processor.
"""

from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch
from transformers import AutoProcessor
from transformers.feature_extraction_utils import BatchFeature

from groot.core.data.schema import EmbodimentTag
from groot.core.data.state_action.state_action_processor import StateActionProcessor
from groot.core.data.types import BaseProcessor, VLAStepDataId
from groot.core.data.utils import parse_modality_configs, to_json_serializable
from groot.core.training.configs.data.embodiment_configs import ModalityConfig
from groot.core.training.configs.model.n2 import (
    ActionConfig,
    GrootN2Config,
    portable_model_dependency_path,
)
from groot.core.training.models.n2.components.image_transforms import (
    FractionalCenterCrop,
    apply_with_replay,
    build_image_transformations_albumentations,
)
from groot.core.training.models.n2.components.text_processor import (
    HuggingfaceTokenizer,
    process_text,
    process_text_negative,
)
from groot.core.utils.storage.msc import resolve_model_path

### Mapping from embodiment tag to projector index.
EMBODIMENT_TAG_TO_PROJECTOR_INDEX = {
    "oxe_droid": 1,
    "oxe_droid_dream": 0,
    "real_r1_pro_sharpa_relative_eef": 2,
    "real_r1_pro_sharpa_relative_eef_human": 2,  # Same projector as robot data
    "real_r1_pro_sharpa_relative_eef_egodex": 2,
    "real_r1_pro_sharpa_relative_eef_maxinsights": 2,
    "real_r1_pro_sharpa_relative_eef_mecka": 2,
    "mecka_relative_eef_active_head": 2,
    "real_g1_relative_eef_absolute_joints": 3,
    "real_g1_relative_eef_absolute_joints_wrist_cam": 3,
    "real_g1_relative_eef_relative_joints": 3,
    # Legacy compatibility only; do not use for new N2 runs. Keep the row-5
    # fallback for reproductions; a saved checkpoint mapping still takes precedence.
    "unitree_g1_whole_body_teleop_latent_gripper": 5,
    "unitree_g1_whole_body_teleop_latent_gripper_new_emb": 15,
    # New pretraining (09/30) uses row 7; original pretraining/resumes retain their saved row.
    "unitree_g1_whole_body_teleop_latent_gripper_wrist": 7,
    "unitree_g1_whole_body_teleop_latent_gripper_new_proj": 7,
    "real_r1_pro_sharpa_relative_joint": 2,
    "oxe_droid_relative_eef_relative_joint": 1,
    "xdof_relative_eef_relative_joint": 0,
    "xdof_relative_eef_relative_joint_subtask": 0,
    "xdof_relative_eef_relative_joint_recap_long": 0,
    "xdof_relative_eef_relative_joint_umi_keys": 0,
    "xdof_relative_eef_relative_joint_umi_keys_4views": 0,
    # RoboDojo ARX X5 shares the XDOF joint-space projector.
    "arx_x5": 0,
    "real_dexmate_sharpa_relative_eef": 2,
    "real_tianji_sharpa_relative_eef": 2,  # Shares the Sharpa projector
    "real_tianji_sharpa_relative_eef_with_tactile": 2,  # Same projector as base tianji tag
    "real_dexmate_sharpa_relative_eef_relative_joint": 2,
    # New controller has a substantially different action distribution from the
    # old controller, so it gets its own projector index instead of sharing 2.
    "real_dexmate_sharpa_relative_eef_new_controller": 4,
    "gripper_umi_4camera": 0,
    "gripper_umi_trigger_camera": 0,
    "single_dexumi": 0,
    "single_dexumi_wo_tac": 0,
    "gripper_umi_trigger_two_head_two_zed_camera_w_rels": 0,
    "single_dexumi_teleop": 0,
    "single_dexumi_teleop_wo_tac": 0,
    "single_dexumi_teleop_wo_tac_abs_hand": 0,
}

_BATCH_UNIFORM_SCALAR_KEYS = frozenset(
    {
        "num_cond_frames",
        "num_total_frames",
        "has_real_action",
    }
)
_ACTION_STATE_KEYS = frozenset(
    {
        "action",
        "action_mask",
        "state",
        "state_mask",
        "state_keep",
        "embodiment_id",
        "has_real_action",
    }
)


def _env_bool_optional(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip().lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _as_python_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError(f"Expected scalar tensor, got shape {tuple(value.shape)}")
        return value.item()

    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"Expected scalar value, got shape {array.shape}")
    return array.item()


def _collate_uniform_scalar(key: str, values: list[Any]) -> Any:
    scalars = [_as_python_scalar(value) for value in values]
    first = scalars[0]
    if any(value != first for value in scalars[1:]):
        raise ValueError(f"GrootN2DataCollator expected batch-uniform `{key}`, got {scalars}")
    return first


def _can_parallelize_eval_images(transform: Any) -> bool:
    # Custom or stochastic pipelines retain serial call and RNG ordering.
    return (
        type(transform) is A.Compose
        and transform.p == 1.0
        and len(transform.transforms) == 2
        and type(transform.transforms[0]) is FractionalCenterCrop
        and type(transform.transforms[1]) is A.Resize
        and all(step.p == 1.0 for step in transform.transforms)
    )


class GrootN2DataCollator:
    """Collator: stack images, tokenize text, stack action/state tensors."""

    def __init__(self, tokenizer: HuggingfaceTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> BatchFeature:
        batch = {}
        keys = list(set().union(*(elem.keys() for elem in features)))
        action_state_keys = _ACTION_STATE_KEYS.intersection(keys)
        for key in action_state_keys:
            if any(key not in elem for elem in features):
                raise ValueError(
                    f"GrootN2DataCollator requires stable action/state key `{key}` "
                    "across the whole batch when action/state metadata is present"
                )
        for key in keys:
            values = [elem[key] for elem in features if key in elem]
            if key == "text":
                batch["text"] = process_text(values, self.tokenizer)
            elif key == "text_negative":
                batch["text_negative"] = process_text_negative(values, self.tokenizer)
            elif key == VLAStepDataId.STEP_DATA_ID_KEY:
                batch[key] = values
            elif key in _BATCH_UNIFORM_SCALAR_KEYS:
                batch[key] = _collate_uniform_scalar(key, values)
            else:
                try:
                    batch[key] = torch.from_numpy(np.stack(values))
                except (ValueError, TypeError):
                    tensors = [torch.as_tensor(v) for v in values]
                    batch[key] = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        return BatchFeature(data={"inputs": batch})


class GrootN2Processor(BaseProcessor):
    """Process robot data for N2 refactor: video + text + optional action/state.

    When action_cfg is None, produces video-only batches (identical to GrootN2Processor).
    When action_cfg is provided, adds action/state/mask/embodiment_id keys.
    """

    data_collator_class = GrootN2DataCollator

    def __init__(
        self,
        modality_configs: dict[str, dict[str, ModalityConfig]],
        # Image
        image_target_size: list[int],
        image_crop_fraction: float = 0.95,
        color_jitter_params: dict[str, float] | None = None,
        codec_aug_params: dict | None = None,
        max_n_cameras: int | None = None,
        # Language
        formalize_language: bool = False,
        max_language_length: int = 512,
        tokenizer_path: str = "",
        language_dropout_prob: float = 0.0,
        null_prompt: str = "low quality, blurry, static",
        # Action/state (None = video-only)
        action_cfg: ActionConfig | None = None,
        statistics: dict | None = None,
        embodiment_id_mapping: dict[str, int] | None = None,
        model_dependency_identities: dict[str, str] | None = None,
        **_kwargs,
    ):
        self.modality_configs = parse_modality_configs(modality_configs)
        self.image_target_size = image_target_size
        # Keep every config-relevant init arg as a self.* attribute so
        # save_pretrained can persist them and from_pretrained restores the
        # full configuration. Previously several of these silently reverted to
        # their defaults on reload (language_dropout_prob, null_prompt,
        # max_language_length, image_crop_fraction, color_jitter_params,
        # embodiment_id_mapping).
        self.image_crop_fraction = image_crop_fraction
        self.color_jitter_params = color_jitter_params
        self.codec_aug_params = codec_aug_params
        self.max_language_length = max_language_length
        self.tokenizer_path = tokenizer_path
        self.model_dependency_identities = model_dependency_identities
        self.formalize_language = formalize_language
        self.max_n_cameras = max_n_cameras
        self.action_cfg = action_cfg
        self.language_dropout_prob = language_dropout_prob
        self.null_prompt = null_prompt

        # Image transforms (always albumentations)
        self.train_image_transform, self.eval_image_transform = (
            build_image_transformations_albumentations(
                target_size=image_target_size,
                crop_fraction=image_crop_fraction,
                interpolation="linear",
                color_jitter_params=color_jitter_params,
                codec_aug_params=codec_aug_params,
            )
        )

        # Tokenizer
        tokenizer_runtime_path = resolve_model_path(tokenizer_path, "processor.tokenizer_path")
        self.tokenizer = HuggingfaceTokenizer(
            tokenizer_runtime_path, seq_len=max_language_length, trust_remote_code=True
        )
        self._collator = self.data_collator_class(tokenizer=self.tokenizer)

        # Action/state processing
        if embodiment_id_mapping is not None:
            self.embodiment_id_mapping = embodiment_id_mapping
        elif action_cfg is not None and action_cfg.get("num_embodiments", 1) == 1:
            # A single-embodiment action head has only projector row 0. Keep
            # globally reserved IDs for multi-embodiment checkpoints only.
            self.embodiment_id_mapping = dict.fromkeys(EMBODIMENT_TAG_TO_PROJECTOR_INDEX, 0)
        else:
            self.embodiment_id_mapping = EMBODIMENT_TAG_TO_PROJECTOR_INDEX
        if action_cfg is not None:
            use_q99_for_rel_action = _env_bool_optional("GROOT_N2_USE_Q99_REL")
            if use_q99_for_rel_action is None:
                use_q99_for_rel_action = action_cfg.get("use_q99_for_rel_action", True)
            self.max_action_dim = action_cfg.get("max_action_dim", 32)
            self.max_state_dim = action_cfg.get("max_state_dim", 64)
            self.max_action_horizon = action_cfg.get("max_action_horizon", None)
            self.state_action_processor = StateActionProcessor(
                modality_configs=modality_configs,
                statistics=statistics,
                apply_sincos_state_encoding=action_cfg.get("apply_sincos_state_encoding", True),
                use_percentiles=action_cfg.get("use_percentiles", False),
                clip_outliers=action_cfg.get("clip_outliers", True),
                use_relative_action=True,
                use_q99_for_rel_action=use_q99_for_rel_action,
                skip_inference_clip=action_cfg.get("skip_inference_clip", False),
            )
            self.state_dropout_prob = action_cfg.get("state_dropout_prob", 0.0)
        else:
            self.state_action_processor = None

        self.train()

    @property
    def collator(self):
        return self._collator

    def train(self):
        super().train()
        if self.state_action_processor is not None:
            self.state_action_processor.train()

    def eval(self):
        super().eval()
        if self.state_action_processor is not None:
            self.state_action_processor.eval()

    def set_statistics(self, statistics, override=False):
        if self.state_action_processor is not None:
            self.state_action_processor.set_statistics(statistics, override=override)

    def decode_action(
        self,
        action: np.ndarray,
        embodiment_tag: EmbodimentTag,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Undo action normalization: trim to horizon, split padded dims, unnormalize."""
        tag = embodiment_tag.value if hasattr(embodiment_tag, "value") else embodiment_tag
        joint_groups = self.modality_configs[tag]["action"].modality_keys
        # Trim temporal dim: model may predict more steps than the data's action horizon
        action_horizon = len(self.modality_configs[tag]["action"].delta_indices)

        # Split concatenated (B, T, max_action_dim) into per-joint-group slices
        # Trim both temporal (to action_horizon) and feature (to joint_dim) padding
        out_dict = {}
        start_idx = 0
        for key in joint_groups:
            joint_dim = int(
                self.state_action_processor.norm_params[tag]["action"][key]["dim"].item()
            )
            out_dict[key] = action[..., :action_horizon, start_idx : start_idx + joint_dim]
            start_idx += joint_dim

        # Unnormalize via StateActionProcessor
        return self.state_action_processor.unapply_action(out_dict, tag, state=state)

    # ─── Image processing (same as GrootN2Processor) ─────────────────────────

    def set_inference_image_workers(self, workers: int) -> None:
        """Set runtime-only camera parallelism; training and custom transforms stay serial."""
        if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 4:
            raise ValueError("N2 inference image workers must be an integer between 1 and 4")
        self._inference_image_workers = workers

    def _process_images(self, content, embodiment_tag):
        """Process images with multicamera temporal stacking. Returns (V*T, H, W, C)."""
        image_transform = self.train_image_transform if self.training else self.eval_image_transform
        image_keys = self.modality_configs[embodiment_tag.value]["video"].modality_keys

        workers = getattr(self, "_inference_image_workers", 1)
        if (
            not self.training
            and workers > 1
            and len(image_keys) > 1
            and _can_parallelize_eval_images(image_transform)
        ):

            def process_view(view):
                # Albumentations mutates per-call parameters. Each camera needs
                # its own transform, even though these operations are deterministic.
                transform = copy.deepcopy(image_transform)
                transformed, _ = apply_with_replay(transform, content.images[view])
                return torch.stack([frame.permute(1, 2, 0) for frame in transformed])

            with ThreadPoolExecutor(max_workers=min(workers, len(image_keys))) as executor:
                temporal_stacked = dict(
                    zip(image_keys, executor.map(process_view, image_keys), strict=True)
                )
        else:
            temporal_stacked = {}
            replay = None
            for view in image_keys:
                transformed, replay = apply_with_replay(
                    image_transform, content.images[view], replay
                )
                # Transforms expose CHW views of HWC image buffers. Stack in the
                # final layout to avoid materializing CHW and transposing it back.
                temporal_stacked[view] = torch.stack(
                    [frame.permute(1, 2, 0) for frame in transformed]
                )

        all_views = [temporal_stacked[k] for k in image_keys]
        actual_n_cameras = len(all_views)
        max_n_cameras = self.max_n_cameras if self.max_n_cameras is not None else actual_n_cameras

        if actual_n_cameras < max_n_cameras:
            pad_view = torch.zeros_like(all_views[0])
            all_views = all_views + [pad_view] * (max_n_cameras - actual_n_cameras)
        images = torch.cat(all_views, dim=0)

        return images, actual_n_cameras, max_n_cameras

    def _build_video_padded_mask(self, content, embodiment_tag, actual_n_cameras, max_n_cameras):
        """Build video padded mask for multiview. Returns ndarray or None."""
        video_delta_indices = self.modality_configs[embodiment_tag.value]["video"].delta_indices
        video_padded_mask = getattr(content, "metadata", {}).get("_padded_mask_video", None)

        if max_n_cameras > 1:
            T_pixel_per_view = len(video_delta_indices)
            if video_padded_mask is not None:
                per_view_mask = video_padded_mask
                real_masks = np.tile(per_view_mask, actual_n_cameras)
                padded_masks = np.ones(
                    len(per_view_mask) * (max_n_cameras - actual_n_cameras),
                    dtype=per_view_mask.dtype,
                )
                video_padded_mask = np.concatenate([real_masks, padded_masks])
            elif actual_n_cameras < max_n_cameras:
                real_masks = np.zeros(T_pixel_per_view * actual_n_cameras, dtype=np.float32)
                padded_masks = np.ones(
                    T_pixel_per_view * (max_n_cameras - actual_n_cameras), dtype=np.float32
                )
                video_padded_mask = np.concatenate([real_masks, padded_masks])

        return video_padded_mask

    @staticmethod
    def _format_reward(reward: Any) -> np.ndarray:
        reward_array = np.asarray(reward, dtype=np.float32)
        if reward_array.size != 1:
            raise ValueError(f"Reward must be one scalar per MDP step, got {reward_array.shape}")
        return reward_array.reshape(1)

    # ─── Action/state processing ─────────────────────────────────────────────

    def _action_horizon_for_tag(self, tag: str) -> int:
        if self.max_action_horizon is not None:
            return self.max_action_horizon

        action_cfg = self.modality_configs.get(tag, {}).get("action")
        if action_cfg is not None:
            return len(action_cfg.delta_indices)

        return self.action_cfg.get("action_horizon", 48) if self.action_cfg is not None else 0

    def _empty_action_state_result(self, tag: str) -> dict[str, np.ndarray]:
        action_horizon = self._action_horizon_for_tag(tag)
        return {
            "action": np.zeros((action_horizon, self.max_action_dim), dtype=np.float32),
            "action_mask": np.zeros((action_horizon, self.max_action_dim), dtype=np.bool_),
            "has_real_action": np.zeros((), dtype=bool),
            "state": np.zeros((1, self.max_state_dim), dtype=np.float32),
            "state_mask": np.zeros((1, self.max_state_dim), dtype=np.bool_),
            "state_keep": np.float32(1.0),
            "embodiment_id": np.int64(self.embodiment_id_mapping.get(tag, 0)),
        }

    def _process_action_state(self, content, embodiment_tag):
        """Normalize, concatenate, and pad action/state. Returns dict of tensors or empty dict."""
        if self.state_action_processor is None:
            return {}

        tag = embodiment_tag.value
        has_states = bool(getattr(content, "states", None))
        has_actions = bool(getattr(content, "actions", None))

        # Keep action/state keys stable for compiled N2 action-head training.
        # The model still skips the action branch when `has_real_action=False`,
        # so video-only/no-action samples keep the legacy training semantics.
        if not has_states and not has_actions:
            return self._empty_action_state_result(tag)

        # Normalize via StateActionProcessor
        if has_actions and has_states:
            normalized_state, normalized_action = self.state_action_processor.apply(
                state=content.states,
                action=content.actions,
                embodiment_tag=tag,
            )
        elif has_states:
            normalized_state = self.state_action_processor.apply_state(
                state=content.states,
                embodiment_tag=tag,
            )
            normalized_action = {}
        else:
            return self._empty_action_state_result(tag)

        # Temporal padding mask from dataset: 1 = clamped past episode end
        action_padded_mask = getattr(content, "metadata", {}).get("_padded_mask_action", None)

        result = {}

        # Concatenate and pad actions
        if normalized_action:
            action_keys = self.modality_configs[tag]["action"].modality_keys
            actions = torch.cat(
                [torch.from_numpy(normalized_action[k]) for k in action_keys], dim=-1
            )
            action_dim = actions.shape[1]
            padded = torch.zeros(actions.shape[0], self.max_action_dim)
            padded[:, :action_dim] = actions
            action_mask = torch.zeros(actions.shape[0], self.max_action_dim, dtype=torch.bool)
            action_mask[:, :action_dim] = True

            # Zero out mask for temporally padded timesteps (clamped past episode end)
            if action_padded_mask is not None:
                action_mask[action_padded_mask.astype(bool), :] = False

            # Pad time dimension to max_action_horizon if set
            if self.max_action_horizon is not None and padded.shape[0] < self.max_action_horizon:
                t_pad = self.max_action_horizon - padded.shape[0]
                padded = torch.cat([padded, torch.zeros(t_pad, self.max_action_dim)], dim=0)
                action_mask = torch.cat(
                    [action_mask, torch.zeros(t_pad, self.max_action_dim, dtype=torch.bool)], dim=0
                )

            result["action"] = padded.to(torch.get_default_dtype()).numpy()
            result["action_mask"] = action_mask.numpy()
            result["has_real_action"] = np.ones((), dtype=bool)
        else:
            action_horizon = self._action_horizon_for_tag(tag)
            result["action"] = np.zeros(
                (action_horizon, self.max_action_dim),
                dtype=np.float32,
            )
            result["action_mask"] = np.zeros(
                (action_horizon, self.max_action_dim),
                dtype=np.bool_,
            )
            result["has_real_action"] = np.zeros((), dtype=bool)

        # Concatenate and pad state
        state_keys = self.modality_configs[tag]["state"].modality_keys
        states = torch.cat([torch.from_numpy(normalized_state[k]) for k in state_keys], dim=-1)
        state_dim = states.shape[1]

        # Sample the state-dropout keep factor here (CPU-side) and pass it as an
        # input tensor. The model forward multiplies state tokens by this scalar
        # unconditionally — no torch.rand / no data-dependent branch in the
        # compiled region, so reduce-overhead CUDA graphs stay valid.
        # At eval time (or when prob == 0) we emit 1.0 → multiply is a no-op.
        # When prob == 1.0, drop state at inference too — the model never saw
        # state during training, so feeding it would be off-distribution.
        # Per-modality exclude_state hard-overrides dropout: deterministically
        # emit 0.0 (train and eval) so this embodiment's state never reaches the
        # transformer. Mirrors N1.7's per-modality exclude_state via state_keep.
        active = self.training or self.state_dropout_prob == 1.0
        if self.modality_configs[tag]["state"].exclude_state or (
            active
            and self.state_dropout_prob > 0
            and np.random.random() < self.state_dropout_prob  # noqa: NPY002
        ):
            state_keep = np.float32(0.0)
        else:
            state_keep = np.float32(1.0)

        padded_state = torch.zeros(states.shape[0], self.max_state_dim)
        padded_state[:, :state_dim] = states
        state_mask = torch.zeros(states.shape[0], self.max_state_dim, dtype=torch.bool)
        state_mask[:, :state_dim] = True

        result["state"] = padded_state.to(torch.get_default_dtype()).numpy()
        result["state_mask"] = state_mask.numpy()
        result["state_keep"] = state_keep

        # Embodiment ID
        result["embodiment_id"] = np.int64(self.embodiment_id_mapping.get(tag, 0))

        return result

    # ─── Main __call__ ───────────────────────────────────────────────────────

    def __call__(self, messages: list[dict[str, Any]]):
        assert len(messages) == 1
        content = messages[0]["content"]
        embodiment_tag = content.embodiment

        # 1. Images
        images, actual_n_cameras, max_n_cameras = self._process_images(content, embodiment_tag)

        # 2. Text
        lang_keys = self.modality_configs[embodiment_tag.value]["language"].modality_keys
        lang_key = np.random.choice(lang_keys)  # noqa: NPY002
        text = content.languages[lang_key]
        if isinstance(text, np.ndarray | list):
            text = text[0]

        # Classifier-free-guidance training: with probability `language_dropout_prob`,
        # replace the positive text with `null_prompt` so the model learns a proper
        # unconditional branch that matches what inference uses for `text_negative`.
        if (
            self.training
            and self.language_dropout_prob > 0
            and np.random.random() < self.language_dropout_prob  # noqa: NPY002
        ):
            text = self.null_prompt

        # 3. Conditioning frames
        video_delta_indices = self.modality_configs[embodiment_tag.value]["video"].delta_indices
        num_cond_frames = sum(1 for d in video_delta_indices if d <= 0)

        result = {
            "images": images.numpy() if isinstance(images, torch.Tensor) else images,
            "text": text,
            # Same string as training dropout; single source of truth via null_prompt.
            "text_negative": self.null_prompt,
            "num_cond_frames": np.int64(num_cond_frames),
            "num_total_frames": np.int64(len(video_delta_indices)),
        }

        # Only propagate ``step_data_id`` when the dataset actually attached it.
        step_data_id = content.metadata.get(VLAStepDataId.STEP_DATA_ID_KEY)
        if step_data_id is not None:
            result[VLAStepDataId.STEP_DATA_ID_KEY] = step_data_id

        # 4. Video padded mask — always include (all-zeros when no padding) so
        # torch.compile sees a consistent input structure every step.
        video_padded_mask = self._build_video_padded_mask(
            content, embodiment_tag, actual_n_cameras, max_n_cameras
        )
        if video_padded_mask is None:
            T_pixel = len(video_delta_indices) * max_n_cameras
            video_padded_mask = np.zeros(T_pixel, dtype=np.float32)
        result["video_padded_mask"] = video_padded_mask

        # 5. Action/state (only when action_cfg is set)
        result.update(self._process_action_state(content, embodiment_tag))

        if "reward" in getattr(content, "metadata", {}):
            result["reward"] = self._format_reward(content.metadata["reward"])

        # 6. Precompute per-latent-frame validity mask for attention masking.
        # This avoids computing it inside the compiled forward (which breaks CUDA graphs).
        # Layout: [video_frames(n_cameras * T_lat_per_view), state(1), action(T_action)]
        # The model expands video frames to tokens with a single repeat_interleave.
        T_pixel_per_view = len(video_delta_indices)
        T_lat_per_view = (T_pixel_per_view - 1) // 4 + 1 if T_pixel_per_view > 0 else 0

        # Video: convert pixel-level padded mask to latent-frame-level
        vpm = video_padded_mask.reshape(max_n_cameras, T_pixel_per_view)
        padded_per_view = vpm.sum(axis=-1)  # (n_cameras,)
        padded_lat_per_view = np.where(
            padded_per_view > 0,
            (padded_per_view.astype(int) - 1) // 4 + 1,
            0,
        )
        video_frame_valid = np.ones(max_n_cameras * T_lat_per_view, dtype=np.bool_)
        for v in range(max_n_cameras):
            if padded_lat_per_view[v] > 0:
                end = (v + 1) * T_lat_per_view
                video_frame_valid[end - padded_lat_per_view[v] : end] = False

        # State: always valid (1 frame)
        state_valid = np.ones(1, dtype=np.bool_)

        # Action: valid where any dim is real
        if "action_mask" in result:
            action_valid = result["action_mask"].any(axis=-1)  # (T_action,)
        else:
            action_valid = np.ones(0, dtype=np.bool_)

        result["frame_valid_mask"] = np.concatenate([video_frame_valid, state_valid, action_valid])

        return result

    # ─── Serialization ───────────────────────────────────────────────────────

    def save_pretrained(self, save_directory: str | Path) -> list[Path]:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        config_file = save_directory / "processor_config.json"
        stats_file = save_directory / "statistics.json"

        config = {
            "processor_class": self.__class__.__name__,
            "processor_kwargs": {
                "modality_configs": to_json_serializable(self.modality_configs),
                "image_target_size": self.image_target_size,
                "image_crop_fraction": self.image_crop_fraction,
                "color_jitter_params": self.color_jitter_params,
                "codec_aug_params": self.codec_aug_params,
                "max_n_cameras": self.max_n_cameras,
                "formalize_language": self.formalize_language,
                "max_language_length": self.max_language_length,
                "tokenizer_path": portable_model_dependency_path(
                    self.tokenizer_path, self.model_dependency_identities
                ),
                "language_dropout_prob": self.language_dropout_prob,
                "null_prompt": self.null_prompt,
                "action_cfg": dict(self.action_cfg) if self.action_cfg else None,
                "embodiment_id_mapping": dict(self.embodiment_id_mapping),
            },
        }
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        stats = {}
        if self.state_action_processor is not None:
            stats = self.state_action_processor.statistics
        with open(stats_file, "w") as f:
            json.dump(to_json_serializable(stats), f, indent=2)
        return [config_file, stats_file]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("_from_auto", None)
        path = Path(pretrained_model_name_or_path)
        with open(path / "processor_config.json") as f:
            config = json.load(f)
        processor_kwargs = config["processor_kwargs"]
        # Load statistics if available
        stats_file = path / "statistics.json"
        if stats_file.exists():
            with open(stats_file) as f:
                processor_kwargs["statistics"] = json.load(f)
        processor_kwargs.update(kwargs)
        return cls(**processor_kwargs)


AutoProcessor.register(GrootN2Config, GrootN2Processor)
