# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Refusals and input checks for tactile/deform GrootN2 models.

Torch-free. Every guard keys off the stored config fields rather than ``model.tactile_spec``,
so it also works before the model loads, on json dicts, on SimpleNamespace test stubs and on
GrootN2RL configs (through ``base_model_cfg``). The exception is ``check_training_inputs``,
which takes fields resolved once at model construction because the training forward may be
compiled. The TRT contract and the RL entry points refuse tactile configs inline, next to
their token-merge refusals, without importing this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

TACTILE_CONFIG_FIELDS = ("tactile_cfg", "deform_cfg")

# Batch key each config field makes the model read, at training and at inference.
_INPUT_KEYS = {"tactile_cfg": "tactile_raw", "deform_cfg": "deform_raw"}
# Future-force target and its clamp count are training-only. Inference initializes
# future latents from noise and never consumes these ground-truth targets.
_FUTURE_KEYS = ("tactile_raw_future", "tactile_future_n_padded")
# Observation group (policy modality config) that produces each config field's inputs.
_POLICY_GROUPS = {"tactile_cfg": "tactile", "deform_cfg": "deform"}


class TactileUnsupportedError(NotImplementedError):
    """A path that does not carry the tactile/deform token groups was asked to run them."""


class MissingTactileInputError(ValueError):
    """A tactile/deform model was given a batch, observation or policy without its inputs."""


def _get(obj: Any, name: str) -> Any:
    # Configs arrive as PretrainedConfig objects, SimpleNamespace stubs or json dicts.
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def tactile_config_fields(config: Any) -> dict[str, Any]:
    """Return the tactile/deform fields a config sets, e.g. ``{"tactile_cfg": {...}}``.

    Only ``None`` means unset: ``{}`` counts as set, matching the ``is not None`` test the
    stored checkpoints were trained under. Recurses into ``base_model_cfg`` so a GrootN2RL
    config reports the fields of the GrootN2 it wraps.
    """
    if config is None:
        return {}
    if not isinstance(config, Mapping) and not hasattr(config, "tactile_cfg"):
        # A model (or wrapper) was passed: read the config it was built from.
        config = getattr(config, "config", config)
    fields = tactile_config_fields(_get(config, "base_model_cfg"))
    for name in TACTILE_CONFIG_FIELDS:
        value = _get(config, name)
        if value is not None:
            fields[name] = value
    return fields


def _predicts_future(fields: dict[str, Any]) -> bool:
    # Lazy: configs.model imports every model config.
    from groot.core.training.configs.model.n2_tactile import DEFAULT_PREDICT_FUTURE

    tactile_cfg = fields.get("tactile_cfg")
    if tactile_cfg is None:
        return False
    predict = _get(tactile_cfg, "predict_future")
    return bool(DEFAULT_PREDICT_FUTURE if predict is None else predict)


def require_exact_grootn2(model: Any, config: Any) -> None:
    """Refuse tactile/deform configs on anything but an exact ``GrootN2``.

    Subclasses either reuse modality ids 4-6 (TTT register tokens, subtask completion tokens)
    or override the denoise/pack paths without the tactile groups, so they would silently drop
    or corrupt the conditioning. Non-tactile configs pass for every class.
    """
    fields = tactile_config_fields(config)
    if not fields:
        return
    # Lazy: model.py imports this package at module level.
    from groot.core.training.models.n2.model import GrootN2

    if type(model) is not GrootN2:
        raise TactileUnsupportedError(
            f"{sorted(fields)} are only supported on an exact GrootN2, got {type(model).__name__}"
        )


def refuse_inference_acceleration(model_config: Any, acceleration: Any) -> None:
    """Refuse TensorRT, CUDA-graph and torch.compile serving for tactile/deform checkpoints.

    ``acceleration`` is an ``N2AccelerationOptions``. Runs before the checkpoint loads. The TRT
    packer takes no token groups, the predict-graph slice would drop their cached K/V and its
    frozen mask cannot be rebuilt under capture, and compiled predict is unvalidated with them.
    """
    fields = tactile_config_fields(model_config)
    if not fields:
        return
    requested = [
        name
        for name, enabled in (
            ("trt_engine_dir", acceleration.trt_engine_dir is not None),
            ("use_cuda_graph", acceleration.use_cuda_graph),
            ("use_torch_compile", acceleration.use_torch_compile),
        )
        if enabled
    ]
    if requested:
        raise TactileUnsupportedError(
            f"checkpoints with {sorted(fields)} serve eager only; unset {requested}"
        )


def check_policy_modalities(model_config: Any, modality_configs: Mapping[str, Any]) -> None:
    """Check that the served modality configs produce every input the model consumes.

    ``modality_configs`` maps modality name to ``ModalityConfig`` (or its dict form), after
    ``Gr00tPolicy._load_modality_configs`` dropped positive deltas, which turns a predict_future
    checkpoint's 40-frame training window into its 20 present frames. Groups the model does not
    consume are ignored, so checkpoints that carry an unused deform group keep serving.
    """
    fields = tactile_config_fields(model_config)
    for field, group in _POLICY_GROUPS.items():
        if field not in fields:
            continue
        modality = modality_configs.get(group)
        if modality is None or not _get(modality, "modality_keys"):
            raise MissingTactileInputError(
                f"the model sets {field} but the policy has no non-empty {group!r} modality group"
            )
    if "tactile_cfg" in fields:
        # Lazy: configs.model imports every model config.
        from groot.core.training.configs.model.n2_tactile import TACTILE_INPUT_WINDOW

        window = len(_get(modality_configs["tactile"], "delta_indices"))
        if window != TACTILE_INPUT_WINDOW:
            raise ValueError(
                f"the served tactile window must be the {TACTILE_INPUT_WINDOW} present frames, "
                f"got {window} delta indices"
            )


def required_input_keys(config: Any, *, training: bool) -> tuple[str, ...]:
    """Batch keys a model built from ``config`` reads, in a fixed order (empty if non-tactile)."""
    return _input_keys(tactile_config_fields(config), training=training)


def _input_keys(fields: dict[str, Any], *, training: bool) -> tuple[str, ...]:
    keys = tuple(key for field, key in _INPUT_KEYS.items() if field in fields)
    if training and _predicts_future(fields):
        keys += _FUTURE_KEYS
    return keys


def _missing_inputs(
    fields: dict[str, Any], data: Mapping[str, Any], *, training: bool
) -> list[str]:
    return [key for key in _input_keys(fields, training=training) if key not in data]


def check_training_batch(
    config: Any, data: Mapping[str, Any], *, has_action: bool, action_loss_details: bool
) -> None:
    """Refuse training batches a tactile/deform model cannot consume, before any encode.

    Per modality: a group the model does not consume (e.g. ``deform_raw`` without deform_cfg)
    is ignored, and a non-tactile model never raises.
    """
    fields = tactile_config_fields(config)
    if fields:
        check_training_inputs(
            tuple(sorted(fields)),
            _input_keys(fields, training=True),
            data,
            has_action=has_action,
            action_loss_details=action_loss_details,
        )


def check_training_inputs(
    fields: tuple[str, ...],
    required: tuple[str, ...],
    data: Mapping[str, Any],
    *,
    has_action: bool,
    action_loss_details: bool,
) -> None:
    """``check_training_batch`` with the config fields and required keys already resolved.

    Traceable (plain dict lookups), so the possibly compiled training forward can run it.
    """
    if action_loss_details:
        # forward's action-loss-details early return skips the tactile loss term.
        raise TactileUnsupportedError(f"action loss details are not supported with {list(fields)}")
    if not has_action:
        # The training token_valid_mask spans [video|state|action] even when only video is
        # packed, so the appended groups would fail with a misleading length error.
        raise TactileUnsupportedError(
            f"{list(fields)} training needs real actions in every batch (has_action=False)"
        )
    missing = [key for key in required if key not in data]
    if not missing:
        return
    if set(missing) <= set(_FUTURE_KEYS):
        # Lazy: configs.model imports every model config.
        from groot.core.training.configs.model.n2_tactile import TACTILE_INPUT_WINDOW

        hint = (
            "predict_future needs a tactile window of present + future frames: "
            f"{2 * TACTILE_INPUT_WINDOW} delta indices in the tactile modality config"
        )
    else:
        hint = "check that the embodiment's modality config declares the tactile/deform groups"
    raise MissingTactileInputError(
        f"training batch is missing {missing} required by {list(fields)}; {hint}"
    )


def check_inference_batch(config: Any, data: Mapping[str, Any]) -> None:
    """Refuse inference batches without the present tactile/deform inputs the model consumes.

    Never requires the future-force keys, never raises for an emitted-but-unused group, and
    never raises for a non-tactile model.
    """
    missing = _missing_inputs(tactile_config_fields(config), data, training=False)
    if missing:
        raise MissingTactileInputError(
            f"inference batch is missing {missing}; the observation must carry the matching "
            "tactile.<key> / deform.<key> entries"
        )
