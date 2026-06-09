# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any, List, Optional

import yaml

from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType

from .data.data_config import DataConfig, SingleDatasetConfig
from .model import create_model_union_type
from .model.gr00t_n1d7 import Gr00tN1d7Config
from .training.training_config import TrainingConfig


def _build_safe_tree(obj: Any) -> Any:
    """Convert an ``asdict()`` tree into one that ``yaml.safe_dump`` accepts.

    ``yaml.safe_dump`` only knows how to write plain primitives (str / int /
    bool / None / list / dict); it raises ``RepresenterError`` on anything
    else. The only "anything else" we hit today is ``Enum`` values —
    ``MODALITY_CONFIGS`` carries ``ActionConfig`` fields holding
    ``ActionRepresentation`` / ``ActionType`` / ``ActionFormat`` enums — so we
    walk the tree once and replace each ``Enum`` with its ``.value`` string
    (e.g. ``ActionRepresentation.RELATIVE`` -> ``"relative"``). An explicit
    walk is easier to audit than a custom ``SafeDumper`` and leaves no global
    YAML state behind.
    """
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _build_safe_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_build_safe_tree(v) for v in obj]
    return obj


def _load_safe_yaml(path: Path):
    """Load YAML with ``yaml.safe_load`` (never the object-constructing
    ``yaml.Loader``), turning the safe loader's rejection of legacy
    ``!!python/object`` tags into a friendly migration error.

    ``yaml.safe_load`` refuses those tags with a ``ConstructorError``; we
    re-raise it as a ``ValueError`` that explains the old config can't load
    and needs a one-time re-save, instead of leaking a cryptic PyYAML
    traceback.
    """
    text = path.read_text()
    try:
        return yaml.safe_load(text)
    except yaml.constructor.ConstructorError as e:
        raise ValueError(
            f"{path}: rejected unsafe legacy config YAML (contains "
            f"{e.problem!r}). The pre-2026-05 Config.save() emitted "
            "PyYAML !!python/object tags which the loader is no longer "
            "willing to instantiate — that path was an arbitrary-code-"
            "execution gadget. Re-save the config via the current "
            "Config.save() to migrate to the plain-dict YAML format."
        ) from e


ModelUnionType = create_model_union_type()


@dataclass
class Config:
    """Complete configuration."""

    load_config_path: Optional[str] = None
    model: ModelUnionType = field(default_factory=lambda: Gr00tN1d7Config())
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def save(self, path: Path):
        """Save the config as plain key/value YAML (no Python-object tags).

        Writes ``asdict(self)`` with ``yaml.safe_dump``, so the file is human
        readable, diffable, and — crucially — builds no Python objects when
        read back. ``_build_safe_tree`` first lowers any ``Enum`` fields to
        plain strings, which ``yaml.safe_dump`` would otherwise refuse to
        write. Replaces the old ``yaml.dump(self)``, which embedded
        ``!!python/object`` tags that made loading a config able to run code.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.safe_dump(_build_safe_tree(asdict(self)), f, sort_keys=False)

    def load(self, path: Path):
        """Load config from a plain key/value YAML file into ``self``.

        Reads via the safe loader (so a malicious or legacy
        ``!!python/object`` file is refused with a migration error, never
        executed) and rebuilds the nested dataclasses with :meth:`load_dict`.
        """
        data = _load_safe_yaml(path)
        if not isinstance(data, dict):
            raise ValueError(
                f"Invalid config file: {path}. Expected a YAML mapping at "
                f"the top level (saved by Config.save()); got {type(data).__name__}."
            )
        self.load_dict(data)
        return self

    def load_dict(self, data: dict):
        if "model" in data:
            self.model = self.model.__class__(**data["model"])
        if "data" in data:
            self.data = DataConfig(**data["data"])
            # Ensure nested datasets are converted to dataclass instances
            converted: List[SingleDatasetConfig] = []
            for ds in self.data.datasets:
                if isinstance(ds, dict):
                    converted.append(SingleDatasetConfig(**ds))
                else:
                    converted.append(ds)
            self.data.datasets = converted
        if "training" in data:
            self.training = TrainingConfig(**data["training"])
        return self

    @classmethod
    def from_pretrained(cls, path: Path) -> "Config":
        """Build a fresh ``Config`` from a YAML file saved by :meth:`save`.

        Same safe-load contract as :meth:`load` (no object construction from
        disk), but returns a new instance via ``cls().load_dict(data)`` instead
        of mutating an existing one.
        """
        data = _load_safe_yaml(Path(path))
        if not isinstance(data, dict):
            raise ValueError(
                f"Invalid config file: {path}. Expected a YAML mapping at "
                f"the top level (saved by Config.save()); got {type(data).__name__}."
            )
        return cls().load_dict(data)

    def get_deepspeed_config(self) -> dict:
        """Generate DeepSpeed configuration."""
        stage = self.training.deepspeed_stage

        gr00t_dir = Path(__file__).parent.parent
        if stage == 2:
            config = json.load(open(gr00t_dir / "configs/deepspeed/zero2_config.json"))
        elif stage == 3:
            config = json.load(open(gr00t_dir / "configs/deepspeed/zero3_config.json"))
        else:
            raise ValueError(f"Invalid DeepSpeed stage: {stage}")

        return config

    def validate(self):
        """Validate configuration."""
        # Check dataset path(s)
        embodiment_tags = set()
        for d_cfg in self.data.datasets:
            # (Disable missing data check because we now support caching PDX data sources.)
            # if not Path(d_cfg.dataset_path).exists():
            #     raise ValueError(f"Dataset path does not exist: {d_cfg.dataset_path}")
            if d_cfg.dataset_type == "physical_embodiment" and not d_cfg.embodiment_tag:
                raise ValueError(f"Embodiment tag is empty for dataset {d_cfg.dataset_path}")
            if d_cfg.embodiment_tag is not None:
                embodiment_tags.add(d_cfg.embodiment_tag)

        stripped_modality_configs = {}
        for embodiment_tag in embodiment_tags:
            modality_cfg = self.data.modality_configs.get(embodiment_tag)
            if modality_cfg is None:
                raise ValueError(
                    f"No modality config registered for embodiment tag '{embodiment_tag}'. "
                    f"Available tags: {sorted(self.data.modality_configs.keys())}. "
                    f"Provide --modality-config-path to register a custom modality config, "
                    f"or use one of the pre-registered tags."
                )
            stripped_modality_configs[embodiment_tag] = modality_cfg
        self.data.modality_configs = stripped_modality_configs

        # ensure mix ratios are valid
        total_ratio = sum(d.mix_ratio for d in self.data.datasets)
        if total_ratio <= 0:
            raise ValueError("Sum of mix_ratio must be greater than zero")

        # Fill in default values for action configs
        for embodiment_tag in self.data.modality_configs:
            # Fill in default values for action representation, type and format
            if self.data.modality_configs[embodiment_tag]["action"].action_configs is None:
                self.data.modality_configs[embodiment_tag]["action"].action_configs = [
                    ActionConfig(
                        rep=ActionRepresentation.ABSOLUTE,
                        type=ActionType.NON_EEF,
                        format=ActionFormat.DEFAULT,
                    )
                ] * len(self.data.modality_configs[embodiment_tag]["action"].modality_keys)

        # Validate precision settings
        if self.training.fp16 and self.training.bf16:
            raise ValueError("Cannot use both fp16 and bf16")


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()
