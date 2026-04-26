# Multi-dataset N1.7 finetune launcher.
#
# Mirrors `gr00t/experiment/launch_finetune.py` but replaces the single
# `--dataset_path` / `--embodiment_tag` with a `--data_yaml` pointing at a
# YAML in the N1.5 multi-dataset format:
#
#     train:
#       datasets:
#         - path: /path/to/dataset_a
#           embodiment_tag: new_embodiment
#           data_config: allex_thetwo_ck40_egostereo
#           weight: 1.0
#         - path: /path/to/dataset_b
#           embodiment_tag: new_embodiment
#           data_config: allex_thetwo_ck40_egostereo
#           weight: 0.5
#
# Each row becomes one entry in `config.data.datasets` with `mix_ratio = weight`.
# The `data_config` field is interpreted as a Python filename under the
# configured search dir (default: `configs/<data_config>.py`) that registers
# a modality config via `register_modality_config(..., EmbodimentTag.*)`.
# All other CLI flags are identical to `launch_finetune.py`.

import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import tyro
import yaml

_SDROP_SUFFIX_RE = re.compile(r"_sdrop(\d+)$")

from gr00t.configs.base_config import get_default_config
from gr00t.experiment.experiment import run


@dataclass
class MultiFinetuneConfig:
    """Multi-dataset variant of gr00t.configs.finetune_config.FinetuneConfig.

    The `data_yaml` field replaces `dataset_path` + `embodiment_tag`.
    Every other field mirrors the single-dataset launcher 1:1.
    """

    # --- Multi-dataset inputs ---
    data_yaml: str
    """Path to a YAML file declaring `train.datasets[]` with `path`, `weight`, `embodiment_tag`, `data_config`."""

    config_search_dir: str = "configs"
    """Directory (relative to CWD) searched for `<data_config>.py` modality files referenced by the YAML."""

    # --- Data and Model Paths ---
    base_model_path: str = "nvidia/GR00T-N1.7-2B"
    """Path to the pretrained base model checkpoint."""

    modality_config_path: Optional[str] = None
    """Optional extra modality config .py file to load in addition to YAML-referenced ones."""

    # --- Model Tuning Flags ---
    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    state_dropout_prob: float = 0.2

    # --- Action representation ---
    # "relative" (default, matches N1.7 pretrain recipe; actions stored as deltas
    # from current state) or "absolute" (raw joint targets). May also be set via
    # YAML (`train.action_representation`) or `_ABS` in the YAML filename.
    action_representation: Optional[str] = None

    # --- Data Augmentation ---
    random_rotation_angle: Optional[int] = None
    color_jitter_params: Optional[dict[str, float]] = None
    extra_augmentation_config: Optional[str] = None

    # --- Training Configuration ---
    global_batch_size: int = 64
    dataloader_num_workers: int = 2
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 1
    output_dir: str = "./outputs"
    experiment_name: Optional[str] = None
    wandb_project: str = "finetune-gr00t-n1d7"
    save_steps: int = 1000
    save_total_limit: int = 5
    num_gpus: int = 1
    use_wandb: bool = False
    max_steps: int = 10000
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05

    shard_size: int = 2**10
    episode_sampling_rate: float = 0.1
    num_shards_per_epoch: int = int(1e5)

    save_only_model: bool = False
    skip_weight_loading: bool = False

    # --- Memory knobs (helpful for single-GPU / OOM mitigation) ---
    gradient_checkpointing: bool = False
    """Recompute activations during backward to trade compute for memory. Big win on 1 GPU."""


def _load_modality_py(path: Path) -> None:
    """Import a user modality config file so `register_modality_config(...)` runs."""
    if not path.exists() or path.suffix != ".py":
        raise FileNotFoundError(f"Modality config not found (expected .py): {path}")
    spec = importlib.util.spec_from_file_location(f"_usercfg_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    print(f"[multi_finetune] Loaded modality config: {path}")


def _parse_yaml(yaml_path: Path):
    with open(yaml_path) as f:
        doc = yaml.safe_load(f) or {}
    train = doc.get("train") or {}
    rows = train.get("datasets") or doc.get("datasets") or []
    if not rows:
        raise ValueError(
            f"No datasets found in {yaml_path}. Expected `train.datasets` or top-level `datasets`."
        )
    for i, row in enumerate(rows):
        if "path" not in row:
            raise ValueError(f"dataset #{i} in {yaml_path} missing required field `path`")
    return doc, train, rows


def _derive_sd_from_rows(rows, yaml_stem: str) -> Optional[float]:
    """Infer state_dropout_prob from N1.5-style conventions.

    Precedence:
      1) Every `data_config` ends with `_sdrop<NN>` -> use NN/100
      2) The YAML filename itself ends with `_SD<NN>` -> use NN/100
    Returns None if neither applies.
    """
    sd_values: set[int] = set()
    for row in rows:
        name = (row.get("data_config") or "").strip()
        m = _SDROP_SUFFIX_RE.search(name)
        if m:
            sd_values.add(int(m.group(1)))
    if len(sd_values) == 1:
        return next(iter(sd_values)) / 100.0

    m = re.search(r"_SD(\d+)$", yaml_stem)
    if m:
        return int(m.group(1)) / 100.0
    return None


def _derive_action_rep(doc, train_block, yaml_stem: str) -> Optional[str]:
    """Infer action_representation (relative|absolute) from YAML conventions.

    Precedence:
      1) explicit `train.action_representation` / top-level `action_representation`
      2) `_ABS` token in the YAML filename stem (analogous to `_SD<NN>` for state dropout)
    Returns None if neither applies.
    """
    v = train_block.get("action_representation")
    if v is None:
        v = doc.get("action_representation")
    if v:
        return str(v).strip().lower()
    if re.search(r"(^|_)ABS(_|$)", yaml_stem):
        return "absolute"
    return None


def _apply_action_representation(rep: str) -> None:
    """Mutate registered `MODALITY_CONFIGS` so every action_config uses `rep`.

    The Python modality files (e.g. `configs/allex_thetwo_ck40_egostereo.py`) stay
    authored in a single canonical form (relative). This helper flips them to the
    chosen representation at launch time so one .py file serves both cases.
    """
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.types import ActionRepresentation

    target_map = {
        "relative": ActionRepresentation.RELATIVE,
        "absolute": ActionRepresentation.ABSOLUTE,
    }
    if rep not in target_map:
        raise ValueError(
            f"Unsupported action_representation={rep!r}; expected one of {list(target_map)}"
        )
    target = target_map[rep]
    for tag_val, cfg in MODALITY_CONFIGS.items():
        action_cfg = cfg.get("action")
        if action_cfg is None or action_cfg.action_configs is None:
            continue
        for ac in action_cfg.action_configs:
            ac.rep = target


def _resolve_data_config_path(name: str, search_dir: Path) -> Path:
    """Return the .py path for a `data_config` name.

    If `<name>.py` doesn't exist but `<name>` ends in `_sdrop<NN>`, fall back to
    the base name without that suffix (N1.7 handles SD at training time, so
    only one modality config is needed regardless of SD value).
    """
    primary = search_dir / f"{name}.py"
    if primary.exists():
        return primary
    stripped = _SDROP_SUFFIX_RE.sub("", name)
    if stripped != name:
        fallback = search_dir / f"{stripped}.py"
        if fallback.exists():
            print(
                f"[multi_finetune] data_config='{name}' not found; "
                f"falling back to '{stripped}.py' (SD is a training-time flag in N1.7)."
            )
            return fallback
    return primary  # will raise in _load_modality_py


def main() -> None:
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"

    ft = tyro.cli(MultiFinetuneConfig, description=__doc__)

    from gr00t.data.embodiment_tags import EmbodimentTag

    yaml_path = Path(ft.data_yaml).expanduser().resolve()
    doc, train_block, rows = _parse_yaml(yaml_path)

    # Register all unique modality configs referenced by the YAML.
    # `_sdrop<NN>` suffixes are stripped so we only load the base Python config.
    loaded: set[str] = set()
    search_dir = Path(ft.config_search_dir).expanduser()
    for row in rows:
        name = row.get("data_config")
        if not name:
            continue
        base_name = _SDROP_SUFFIX_RE.sub("", name)
        if base_name in loaded:
            continue
        cfg_py = _resolve_data_config_path(name, search_dir)
        _load_modality_py(cfg_py)
        loaded.add(base_name)

    if ft.modality_config_path:
        _load_modality_py(Path(ft.modality_config_path))

    # action_representation resolution (YAML-native):
    #   1) explicit `train.action_representation` / top-level field
    #   2) `_ABS` token in YAML filename
    #   3) CLI `--action_representation`
    #   4) default "relative" (matches N1.7 pretrain)
    rep = _derive_action_rep(doc, train_block, yaml_path.stem) or ft.action_representation
    rep = (rep or "relative").strip().lower()
    _apply_action_representation(rep)
    ft.action_representation = rep
    print(f"[multi_finetune] action_representation = {rep}")

    # state_dropout_prob resolution (YAML-native, no env var needed):
    #   1) explicit `train.state_dropout_prob` in YAML
    #   2) `_sdrop<NN>` suffix on every dataset's `data_config`
    #   3) `_SD<NN>` suffix on the YAML filename
    #   4) CLI default on MultiFinetuneConfig (ft.state_dropout_prob)
    yaml_sd = train_block.get("state_dropout_prob")
    if yaml_sd is None:
        yaml_sd = doc.get("state_dropout_prob")
    if yaml_sd is None:
        yaml_sd = _derive_sd_from_rows(rows, yaml_path.stem)
    if yaml_sd is not None:
        yaml_sd = float(yaml_sd)
        if abs(yaml_sd - ft.state_dropout_prob) > 1e-9:
            print(
                f"[multi_finetune] state_dropout_prob: using {yaml_sd} from YAML "
                f"(overrides CLI value {ft.state_dropout_prob})"
            )
        ft.state_dropout_prob = yaml_sd
    print(f"[multi_finetune] Effective state_dropout_prob = {ft.state_dropout_prob}")

    # Build N1.7 multi-dataset list. Each YAML row -> one SingleDatasetConfig.
    datasets = []
    for row in rows:
        tag = row.get("embodiment_tag", "new_embodiment")
        resolved = EmbodimentTag.resolve(tag).value
        datasets.append(
            {
                "dataset_paths": [row["path"]],
                "mix_ratio": float(row.get("weight", 1.0)),
                "embodiment_tag": resolved,
            }
        )
    print(f"[multi_finetune] Mixing {len(datasets)} dataset(s):")
    for d in datasets:
        print(
            f"  - mix_ratio={d['mix_ratio']:<5} tag={d['embodiment_tag']:<15} "
            f"path={d['dataset_paths'][0]}"
        )

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": datasets,
            }
        }
    )
    config.load_config_path = None

    # Mirror launch_finetune.py: model overrides
    config.model.tune_llm = ft.tune_llm
    config.model.tune_visual = ft.tune_visual
    config.model.tune_projector = ft.tune_projector
    config.model.tune_diffusion_model = ft.tune_diffusion_model
    config.model.state_dropout_prob = ft.state_dropout_prob
    config.model.random_rotation_angle = ft.random_rotation_angle
    config.model.color_jitter_params = ft.color_jitter_params
    if ft.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = (ft.action_representation == "relative")

    # Mirror launch_finetune.py: training overrides
    config.training.experiment_name = ft.experiment_name
    config.training.start_from_checkpoint = ft.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft.global_batch_size
    config.training.dataloader_num_workers = ft.dataloader_num_workers
    config.training.learning_rate = ft.learning_rate
    config.training.gradient_accumulation_steps = ft.gradient_accumulation_steps
    config.training.output_dir = ft.output_dir
    config.training.save_steps = ft.save_steps
    config.training.save_total_limit = ft.save_total_limit
    config.training.num_gpus = ft.num_gpus
    config.training.use_wandb = ft.use_wandb
    config.training.max_steps = ft.max_steps
    config.training.weight_decay = ft.weight_decay
    config.training.warmup_ratio = ft.warmup_ratio
    config.training.wandb_project = ft.wandb_project

    config.data.shard_size = ft.shard_size
    config.data.episode_sampling_rate = ft.episode_sampling_rate
    config.data.num_shards_per_epoch = ft.num_shards_per_epoch

    config.training.save_only_model = ft.save_only_model
    config.training.skip_weight_loading = ft.skip_weight_loading
    config.training.gradient_checkpointing = ft.gradient_checkpointing

    run(config)


if __name__ == "__main__":
    main()
