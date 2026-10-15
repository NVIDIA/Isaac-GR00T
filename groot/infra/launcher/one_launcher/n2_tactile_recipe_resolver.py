# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Lower training-only tactile switches without changing stored model semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
import itertools
from typing import Any

from groot.infra.launcher.one_launcher.resolved_training_experiment import (
    ResolvedTrainingExperiment,
)
from groot.infra.launcher.one_launcher.training_experiment_override import (
    TrainingExperimentOverride,
)

# Encoder locations belong to the experiment or its overrides, not this public resolver.
_CONDITIONING_PATTERN = "bidirectional_video_action_full"
_PREDICTION_PATTERN = "bidirectional_video_tactile_action_full"
_PREDICTION_DEFAULTS = {
    "tactile_loss_scale": 0.2,
    "tactile_loss_warmup_steps": 0,
    # Match the reference recipe: retain edge-padded future targets at episode ends.
    "future_pad_tolerance": 20,
}
_SWITCHES = frozenset({"enabled", "force_conditioning", "predict_future", "deform_conditioning"})
_FAMILIES = {
    "real_tianji_sharpa_relative_eef": "tianji",
    "real_tianji_sharpa_relative_eef_with_tactile": "tianji",
    "real_dexmate_sharpa_relative_eef": "dexmate",
    "real_dexmate_sharpa_relative_eef_with_tactile": "dexmate",
}
_STRIDES = {"tianji": 3, "dexmate": 2}  # 30/20 Hz datasets, sampled at 10 Hz.
_FINGERS = tuple(
    (hand, finger)
    for hand in ("left", "right")
    for finger in ("thumb", "index", "middle", "ring", "pinky")
)


class N2TactileRecipeResolver:
    """Consume ``training.model.tactile`` after ordinary experiment overrides.

    Without switches, existing recipes are returned unchanged. The reference
    experiment enables tactile explicitly; other homogeneous Tianji/Dexmate
    Sharpa GrootN2 recipes can opt in with ``enabled=True``. That fills missing
    choices per modality, preserving explicit nulls and individual ablations.
    Feature switches change only their requested choices and dependent data.
    ``enabled=False`` is a master opt-out that wins over all other switches.

    This is not an inference or checkpoint-config migration. Resolution performs
    no dataset/encoder I/O, and the source experiment is never mutated.
    """

    def apply(
        self, resolved: ResolvedTrainingExperiment, *, config_overrides: Sequence[str] = ()
    ) -> ResolvedTrainingExperiment:
        result = deepcopy(resolved)
        model = dict(result.model_section)
        options = self._options(model.pop("tactile", {}))
        self._check_switch_overrides(options, config_overrides)
        result = replace(result, model_section=model)
        force_key = self._key(model, "tactile_cfg")
        deform_key = self._key(model, "deform_cfg")
        if not options:
            return result

        tags = self._tags(result.data_section)
        families = {_FAMILIES.get(tag) for tag in tags}
        eligible = model.get("name") == "GrootN2" and len(families) == 1 and None not in families
        if not eligible:
            if options.get("enabled") is not False and any(options.values()):
                raise ValueError(
                    "Tactile recipe switches require exact GrootN2 and selected datasets "
                    "from one supported Tianji/Dexmate Sharpa family"
                )
            # Negative switches on unrelated recipes must not change their architecture,
            # attention rules or registry defaults. Existing tactile models still need
            # explicit supported tags before their data contract can be rewritten.
            active = model.get(force_key) is not None or model.get(deform_key) is not None
            if options and active:
                raise ValueError(
                    "Cannot rewrite an existing tactile recipe without supported model/tags"
                )
            return result

        family = _FAMILIES[tags[0]]
        tactile = self._optional_mapping(model.get(force_key), force_key)
        deform = self._optional_mapping(model.get(deform_key), deform_key)
        enable_defaults = options.get("enabled") is True
        force_on = options.get(
            "force_conditioning", tactile is not None if force_key in model else enable_defaults
        )
        deform_on = options.get(
            "deform_conditioning", deform is not None if deform_key in model else enable_defaults
        )
        predict = options.get(
            "predict_future",
            (
                tactile.get("predict_future", enable_defaults)
                if tactile is not None
                else enable_defaults and force_on
            ),
        )
        if not isinstance(predict, bool):
            raise TypeError("tactile_cfg.predict_future must be a bool")
        if options.get("enabled") is False:
            force_on = deform_on = predict = False
        elif not force_on:
            if options.get("predict_future") is True:
                raise ValueError("predict_future=True requires force_conditioning=True")
            predict = False
        if (force_on or deform_on) and model.get(
            self._key(model, "token_merge_kernel")
        ) is not None:
            raise ValueError("Tactile recipe switches do not support token_merge_kernel")

        if force_on:
            tactile = {} if tactile is None else tactile
            self._require_encoder_path(tactile, "tactile_cfg", "tactile_vae_ckpt_path")
            tactile["predict_future"] = predict
            if predict:
                for key, value in _PREDICTION_DEFAULTS.items():
                    tactile.setdefault(key, value)
            else:
                for key in _PREDICTION_DEFAULTS:
                    tactile.pop(key, None)
        else:
            tactile = None
        if deform_on:
            deform = {} if deform is None else deform
            self._require_encoder_path(deform, "deform_cfg", "deform_encoder_ckpt_path")
        else:
            deform = None
        model[force_key] = tactile
        model[deform_key] = deform
        self._attention(model, predict=predict, active=force_on or deform_on)

        data = dict(result.data_section)
        modalities = self._mapping(data.get("modality_configs", {}), "data.modality_configs")
        data["modality_configs"] = modalities
        for tag in tags:
            groups = self._mapping(modalities.get(tag, {}), f"data.modality_configs.{tag}")
            modalities[tag] = groups
            self._force_group(
                groups,
                family=family,
                enabled=force_on,
                predict=predict,
                reactivate=options.get("force_conditioning") is True,
            )
            self._deform_group(
                groups,
                family=family,
                enabled=deform_on,
                reactivate=options.get("deform_conditioning") is True,
            )
        return replace(result, data_section=data)

    def _require_encoder_path(self, config: dict[str, Any], group: str, key: str) -> None:
        path = config.get(key)
        if not isinstance(path, str) or not path:
            raise ValueError(
                f"Enabling {group} requires an encoder path; set training.model.{group}.{key} "
                "in the experiment or with --set. For a null config, replace the mapping with "
                f"--set 'training.model.{group}={{{key}: /path/to/encoder}}'. "
                "Paths are resolved on the training worker."
            )

    def _check_switch_overrides(self, options: dict[str, bool], expressions: Sequence[str]) -> None:
        # A master opt-out always wins. Otherwise don't silently reinterpret explicit
        # low-level CLI choices through a contradictory recipe switch.
        if not options or options.get("enabled") is False:
            return
        low_level = {}
        for expression in expressions:
            override = TrainingExperimentOverride.init_from_expression(expression)
            path = tuple(part.replace("-", "_") for part in override.path)
            if path == ("training", "model", "tactile_cfg"):
                if override.value is None or isinstance(override.value, Mapping):
                    low_level["force_conditioning"] = override.value is not None
                    if override.value is None:
                        low_level["predict_future"] = False
                    elif "predict_future" in override.value:
                        low_level["predict_future"] = override.value["predict_future"]
                    else:
                        low_level.pop("predict_future", None)
            elif path == ("training", "model", "tactile_cfg", "predict_future"):
                low_level["predict_future"] = override.value
            elif path == ("training", "model", "deform_cfg") and (
                override.value is None or isinstance(override.value, Mapping)
            ):
                low_level["deform_conditioning"] = override.value is not None
        conflicts = [
            name for name in low_level if name in options and low_level[name] != options[name]
        ]
        if options.get("force_conditioning") is False and low_level.get("predict_future") is True:
            conflicts.append("predict_future")
        if conflicts:
            raise ValueError(
                "Conflicting tactile switches and low-level --set values for "
                f"{sorted(set(conflicts))}; use training.model.tactile.* for ablation switches "
                "and tactile_cfg/deform_cfg for encoder paths and loss settings"
            )

    def _options(self, value: Any) -> dict[str, bool]:
        options = self._mapping(value, "training.model.tactile")
        unknown = set(options) - _SWITCHES
        if unknown:
            raise ValueError(f"Unknown training.model.tactile switches: {sorted(unknown, key=str)}")
        for name, enabled in options.items():
            if not isinstance(enabled, bool):
                raise TypeError(f"training.model.tactile.{name} must be a bool")
        return options

    def _tags(self, data: Mapping[str, Any]) -> tuple[str, ...]:
        datasets = data.get("datasets", ())
        if not isinstance(datasets, list | tuple) or not datasets:
            return ()
        tags = []
        for dataset in datasets:
            if not isinstance(dataset, Mapping) or not isinstance(
                dataset.get("embodiment_tag"), str
            ):
                return ()
            tags.append(dataset["embodiment_tag"])
        return tuple(dict.fromkeys(tags))

    def _mapping(self, value: Any, name: str) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping")
        return dict(value)

    def _optional_mapping(self, value: Any, name: str) -> dict[str, Any] | None:
        return None if value is None else self._mapping(value, name)

    def _key(self, mapping: Mapping[str, Any], name: str) -> str:
        aliases = [key for key in mapping if isinstance(key, str) and key.replace("-", "_") == name]
        if len(aliases) > 1:
            raise ValueError(f"Ambiguous aliases for {name}: {aliases}")
        return aliases[0] if aliases else name

    def _attention(self, model: dict[str, Any], *, predict: bool, active: bool) -> None:
        if not active:
            # Disabling all tactile inputs must preserve ordinary attention policies,
            # including mixtures. Only the prediction alias needs to be removed.
            for key in ("action-cfg", "action_cfg"):
                action = model.get(key)
                if isinstance(action, dict):
                    for name in ("attention-mask-pattern", "attention_mask_pattern"):
                        if name in action:
                            action[name] = self._without_prediction_alias(action[name])
            return
        # Some recipes use a string discriminator plus an underscore mapping; others
        # use a single mapping with _config. Normalize both to one CLI subcommand.
        action: dict[str, Any] = {}
        for key in ("action-cfg", "action_cfg"):
            if key not in model:
                continue
            value = model.pop(key)
            if isinstance(value, str):
                if value != "action-config":
                    raise ValueError(f"Unsupported tactile action config: {value!r}")
            else:
                if action:
                    raise ValueError("Ambiguous action-cfg/action_cfg mappings")
                action = self._mapping(value, key)
        if action.get("_config", "action-config") != "action-config":
            raise ValueError("Tactile recipe switches require action-config")
        pattern_key = self._key(action, "attention_mask_pattern")
        old = action.get(pattern_key)
        if old is not None:
            if isinstance(old, tuple | list) and len(old) == 1:
                old = old[0]
            if not isinstance(old, tuple | list) or tuple(old) not in (
                (_CONDITIONING_PATTERN, 1.0),
                (_PREDICTION_PATTERN, 1.0),
            ):
                raise ValueError(
                    "Tactile recipe switches cannot replace an unsupported attention_mask_pattern"
                )
        action["_config"] = "action-config"
        action[pattern_key] = [_PREDICTION_PATTERN if predict else _CONDITIONING_PATTERN, 1.0]
        model["action-cfg"] = action

    def _without_prediction_alias(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self._without_prediction_alias(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._without_prediction_alias(item) for item in value)
        return _CONDITIONING_PATTERN if value == _PREDICTION_PATTERN else value

    def _force_group(
        self, groups: dict[str, Any], *, family: str, enabled: bool, predict: bool, reactivate: bool
    ) -> None:
        group = self._mapping(groups.get("tactile", {}), "tactile modality")
        groups["tactile"] = group
        if not enabled:
            # Empty keys suppress loader reads. Keep a nonempty, current-only delta
            # list so disabled groups neither break max(delta) nor extend padding.
            group.update(modality_keys=[], delta_indices=[0])
            return
        if reactivate and group.get("modality_keys") == []:
            group.pop("modality_keys")
            if group.get("delta_indices") == [0]:
                group.pop("delta_indices")
        group.setdefault("modality_keys", [f"{hand}_{finger}_f6" for hand, finger in _FINGERS])
        self._validate_keys(group["modality_keys"], "tactile")
        stride = _STRIDES[family]
        deltas = group.get("delta_indices", list(range(-19 * stride, 1, stride)))
        if (
            not isinstance(deltas, list | tuple)
            or len(deltas) not in (20, 40)
            or any(type(delta) is not int for delta in deltas)
            or any(a >= b for a, b in itertools.pairwise(deltas))
            or any(delta > 0 for delta in deltas[:20])
            or any(delta <= 0 for delta in deltas[20:])
        ):
            raise ValueError(
                "tactile delta_indices need 20 ordered observed frames and optionally 20 future frames"
            )
        if predict and len(deltas) == 20:
            strides = {b - a for a, b in itertools.pairwise(deltas)}
            if len(strides) != 1:
                raise ValueError(
                    "Provide 40 explicit tactile delta_indices to extend a nonuniform observed window"
                )
            stride = strides.pop()
            deltas = [*deltas, *range(stride, 21 * stride, stride)]
        group["delta_indices"] = list(deltas if predict else deltas[:20])

    def _deform_group(
        self, groups: dict[str, Any], *, family: str, enabled: bool, reactivate: bool
    ) -> None:
        group = self._mapping(groups.get("deform", {}), "deform modality")
        groups["deform"] = group
        if not enabled:
            group.update(modality_keys=[], delta_indices=[0])
            return
        if reactivate and group.get("modality_keys") == []:
            group.pop("modality_keys")
        group.setdefault(
            "modality_keys",
            [
                (
                    f"{hand}_hand_tactile_deform_{finger}"
                    if family == "tianji"
                    else f"tactile_deform_{hand}_{finger}_freq20"
                )
                for hand, finger in _FINGERS
            ],
        )
        self._validate_keys(group["modality_keys"], "deform")
        group.setdefault("delta_indices", [0])
        deltas = group["delta_indices"]
        if (
            not isinstance(deltas, list | tuple)
            or list(deltas) != [0]
            or type(deltas[0]) is not int
        ):
            raise ValueError("deform delta_indices must be [0]")

    def _validate_keys(self, keys: Any, group: str) -> None:
        if (
            not isinstance(keys, list | tuple)
            or len(keys) != 10
            or any(not isinstance(key, str) or not key for key in keys)
            or len(set(keys)) != 10
        ):
            raise ValueError(f"{group} needs ten distinct nonempty finger keys in hand-major order")
