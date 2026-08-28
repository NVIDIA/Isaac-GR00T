# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Build and run experiment_ray.py commands for One Launcher experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Any

import yaml

from groot.core.training.configs.training.training_config import TrainingConfig
from groot.infra.launcher.one_launcher.training_experiment_resolver import (
    TrainingExperimentResolver,
)

_SECTION_ORDER = ("data", "training", "model")
_DEFAULT_EXPERIMENT_CONFIG_ROOT = Path(".ol/experiments")
_DEFAULT_EXPERIMENT_DIR = Path(__file__).resolve().parents[2] / "experiments/training/configs"
_DEFAULT_OUTPUT_DIR = "/tmp/groot_outputs/{USER_SLUG}/{JOB_NAME}"  # noqa: S108
_MODEL_CKPT_ROOT_FLAGS = {
    "--model.tokenizer_path",
    "--model.diffusion-model-cfg.diffusion_model_pretrained_path",
    "--model.text-encoder-cfg.text_encoder_pretrained_path",
    "--model.vae-cfg.vae_pretrained_path",
}
_CLI_KEY_OVERRIDES = {
    "data": {
        "no_download_cache": "no-download-cache",
    },
    "model": {
        "color_jitter_params": "color-jitter-params",
        "load_bf16": "load_bf16",
        "reproject_vision": "reproject_vision",
        "use_percentiles": "use-percentiles",
        "use_mean_std": "use-mean-std",
        "exclude_state": "exclude_state",
    },
}


@dataclass(frozen=True)
class ExperimentRayTrainingConfig:
    path: Path
    body: Mapping[str, Any]

    @classmethod
    def init_from_path(cls, path: Path) -> ExperimentRayTrainingConfig:
        body = yaml.safe_load(path.read_text()) or {}
        if not isinstance(body, dict):
            raise TypeError(f"{path}: top-level YAML must be a mapping")
        unknown = set(body) - {"data", "training", "model"}
        if unknown:
            raise ValueError(f"{path}: unknown section(s): {', '.join(sorted(unknown))}")
        return cls(path=path, body=body)

    def section(self, name: str) -> Mapping[str, Any]:
        section = self.body.get(name, {})
        if not isinstance(section, dict):
            raise TypeError(f"{self.path}: {name} must be a mapping")
        return section


@dataclass(frozen=True)
class ExperimentRayTrainCommand:
    data_config: Path
    training_config: Path
    env: Mapping[str, str]

    @classmethod
    def init_from_experiment(
        cls,
        *,
        experiment: str,
        compute_type: str,
        config_overrides: Sequence[str] = (),
        experiment_dir: Path = _DEFAULT_EXPERIMENT_DIR,
        output_root: Path = _DEFAULT_EXPERIMENT_CONFIG_ROOT,
        env: Mapping[str, str] | None = None,
    ) -> ExperimentRayTrainCommand:
        resolver = TrainingExperimentResolver.init_default(experiment_dir=experiment_dir)
        resolved = resolver.resolve(
            name=experiment,
            compute_type=compute_type,
            config_overrides=config_overrides,
        )
        if config_overrides:
            baseline = resolver.resolve(name=experiment, compute_type=compute_type)
        else:
            baseline = None
        if baseline is not None:
            resolved_fingerprint = resolved.config_fingerprint()
            if resolved_fingerprint != baseline.config_fingerprint():
                output_root = output_root / "_overrides" / resolved_fingerprint[:16]
        files = resolved.write_configs(output_root)
        return cls(
            data_config=files.data_config,
            training_config=files.training_config,
            env=dict(env or os.environ),
        )

    def run(self) -> None:
        subprocess.run(self.build_command(), check=True)

    def build_command(self) -> list[str]:
        return [
            "python",
            "-m",
            "groot.core.training.experiment.experiment_ray",
            "--load_config_path",
            str(self.data_config),
            *self.training_args(self.placeholder_values()),
        ]

    def training_args(self, values: Mapping[str, str]) -> list[str]:
        config = ExperimentRayTrainingConfig.init_from_path(self.training_config)
        args: list[str] = []
        for section in _SECTION_ORDER:
            section_body = config.section(section)
            if section == "model":
                args.extend(self.model_args(section_body, values))
            elif section == "training":
                args.extend(self.training_section_args(section_body, values))
            else:
                args.extend(self.section_args(section, section_body, values))
        return args

    def training_section_args(
        self,
        body: Mapping[str, Any],
        values: Mapping[str, str],
    ) -> list[str]:
        body = dict(body)
        discriminator = body.pop("_config", None)
        if discriminator is not None and not isinstance(discriminator, str):
            raise TypeError(f"{self.training_config}: training._config must be a string")
        # Capture this launcher-only sizing value before subtype filtering;
        # TrainingConfig does not declare it, but the CLI expansion below needs it.
        global_batch_size = body.pop("global_batch_size", None)
        if discriminator is not None:
            # `_config` and subtype-only fields are YAML inputs, not arguments on
            # the base training CLI. They were already materialized in data.yaml.
            base_fields = TrainingConfig.__dataclass_fields__
            body = {key: value for key, value in body.items() if key in base_fields}
        body.setdefault("output_dir", _DEFAULT_OUTPUT_DIR)
        args = self.section_args("training", body, values)
        if global_batch_size is None:
            return args
        if "per_device_batch_size" in body or "num_gpus" in body:
            raise ValueError(
                f"{self.training_config}: training.global_batch_size cannot be combined "
                "with training.per_device_batch_size or training.num_gpus"
            )
        return [
            *args,
            "--training.per_device_batch_size",
            self.required_placeholder_value(values, "BATCH_SIZE"),
            "--training.num_gpus",
            self.required_placeholder_value(values, "NUM_GPUS"),
        ]

    def model_args(self, body: Mapping[str, Any], values: Mapping[str, str]) -> list[str]:
        model_name = body.get("name")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(f"{self.training_config}: model.name must be a non-empty string")
        args = [f"model:{self.expand_placeholders(model_name.strip(), values)}"]
        args.extend(
            self.model_mapping_args(
                {key: value for key, value in body.items() if key != "name"},
                values,
            )
        )
        return args

    def model_mapping_args(
        self,
        body: Mapping[str, Any],
        values: Mapping[str, str],
        prefix: str = "",
    ) -> list[str]:
        model_overrides: list[tuple[str, Any]] = []
        subcommands: list[tuple[str, str, dict[str, Any]]] = []
        subcommand_indexes: dict[str, int] = {}
        for raw_key, value in body.items():
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise TypeError(f"{self.training_config}: model keys must be strings")
            key = self.cli_key("model", raw_key) if not prefix else raw_key
            key_path = f"{prefix}.{key}" if prefix else key
            if key.endswith("-cfg") and isinstance(value, str):
                subcommand_indexes[key] = len(subcommands)
                subcommands.append((key_path, self.expand_placeholders(value, values), {}))
                continue
            if isinstance(value, dict) and isinstance(value.get("_config"), str):
                subcommand_indexes[key] = len(subcommands)
                subcommands.append(
                    (
                        key_path,
                        self.expand_placeholders(value["_config"], values),
                        {
                            nested_key: nested_value
                            for nested_key, nested_value in value.items()
                            if nested_key != "_config"
                        },
                    )
                )
                continue
            subcommand_key = key.replace("_", "-")
            if isinstance(value, dict) and subcommand_key in subcommand_indexes:
                index = subcommand_indexes[subcommand_key]
                existing_key_path, config_name, nested = subcommands[index]
                nested.update(value)
                subcommands[index] = (existing_key_path, config_name, nested)
                continue
            model_overrides.append((key_path, value))
        args: list[str] = []
        for key_path, value in model_overrides:
            if isinstance(value, dict):
                args.extend(self.model_mapping_args(value, values, key_path))
            else:
                args.extend(self.flag_args(f"--model.{key_path}", value, values))
        for key_path, config_name, nested in subcommands:
            args.append(f"model.{key_path}:{config_name}")
            args.extend(self.model_mapping_args(nested, values, key_path))
        return args

    def section_args(
        self,
        section: str,
        body: Mapping[str, Any],
        values: Mapping[str, str],
    ) -> list[str]:
        args: list[str] = []
        for key_path, value in self.flatten_section(section, body):
            args.extend(self.flag_args(f"--{section}.{key_path}", value, values))
        return args

    def flatten_section(
        self,
        section: str,
        body: Mapping[str, Any],
        prefix: str = "",
    ) -> list[tuple[str, Any]]:
        items: list[tuple[str, Any]] = []
        for raw_key, value in body.items():
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise TypeError(f"{self.training_config}: {section} keys must be strings")
            key = self.cli_key(section, raw_key) if not prefix else raw_key
            key_path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                items.extend(self.flatten_section(section, value, key_path))
            else:
                items.append((key_path, value))
        return items

    def cli_key(self, section: str, key: str) -> str:
        return _CLI_KEY_OVERRIDES.get(section, {}).get(key, key)

    def flag_args(self, flag: str, value: Any, values: Mapping[str, str]) -> list[str]:
        if isinstance(value, bool):
            if value:
                return [flag]
            prefix, key = flag.rsplit(".", 1)
            return [f"{prefix}.no-{key}"]
        if value is None:
            return []
        if isinstance(value, list):
            return [flag, *(self.expand_placeholders(str(item), values) for item in value)]
        if flag in _MODEL_CKPT_ROOT_FLAGS:
            return [flag, self.resolve_model_ckpt_path(str(value), values)]
        return [flag, self.expand_placeholders(str(value), values)]

    def placeholder_values(self) -> Mapping[str, str]:
        return {
            "JOB_NAME": self.env.get("JOB_NAME", ""),
            "USER_SLUG": self.env.get("USER_SLUG", ""),
            "BATCH_SIZE": self.env.get("BATCH_SIZE", ""),
            "NUM_GPUS": self.env.get("NUM_GPUS", ""),
            "PATHS.checkpoints": self.env.get(
                "TRAINING_OUTPUT_ROOT",
                "/tmp/groot_outputs",  # noqa: S108
            ),
            "PATHS.ckpts": self.env.get("GROOT_CHECKPOINT_ROOT", ""),
            "COSMOS_MODEL_PATH": self.env.get("COSMOS_MODEL_PATH", ""),
            "DIFFUSION_TRANSFORMER_PATH": self.env.get("DIFFUSION_TRANSFORMER_PATH", ""),
            "VAE_PATH": self.env.get("VAE_PATH", ""),
            "PRETRAIN_CHECKPOINT": self.env.get("PRETRAIN_CHECKPOINT", ""),
        }

    def expand_placeholders(self, value: str, values: Mapping[str, str]) -> str:
        expanded = value
        for key, replacement in values.items():
            expanded = expanded.replace("{" + key + "}", replacement)
        if "{" in expanded or "}" in expanded:
            raise ValueError(f"Unknown one-launcher placeholder in {value!r}")
        return expanded

    def resolve_model_ckpt_path(self, value: str, values: Mapping[str, str]) -> str:
        expanded = self.expand_placeholders(value, values)
        if not expanded or "://" in expanded or Path(expanded).is_absolute():
            return expanded
        root = self.required_placeholder_value(values, "PATHS.ckpts")
        return str(Path(root) / expanded)

    def required_placeholder_value(self, values: Mapping[str, str], key: str) -> str:
        value = values.get(key, "")
        if not value:
            raise ValueError(f"{self.training_config}: missing launcher value for {key}")
        return value
