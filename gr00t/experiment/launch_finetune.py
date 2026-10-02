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

# Launch finetuning for N1.7 on "single node".
# This script tries to provide a similar user experience as current OSS.

from copy import deepcopy
import json
import os
from pathlib import Path

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run


def load_modality_config(modality_config_path: str, *, embodiment_tag=None, use_tactile=False):
    """Load existing Python registrations or a JSON with optional tactile settings.

    JSON modalities are local to this run; the shared registry is not mutated.
    The dataset's meta/modality.json describes array slices and is a separate format.
    """
    import importlib
    import sys

    from gr00t.data.types import ModalityConfig

    path = Path(modality_config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Modality config path does not exist: {path}")
    if path.suffix == ".py":
        if use_tactile:
            raise ValueError("--use_tactile requires a modality JSON with a tactile section")
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
        return None, {}
    if path.suffix != ".json":
        raise ValueError("Modality config must be a .py or .json file")
    payload = json.loads(path.read_text())
    if (
        embodiment_tag is not None
        and payload.get("embodiment_tag", embodiment_tag) != embodiment_tag
    ):
        raise ValueError("JSON embodiment_tag does not match --embodiment-tag")
    try:
        modalities = {k: ModalityConfig(**v) for k, v in payload["modalities"].items()}
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Expected a modalities object with ModalityConfig fields, not dataset meta/modality.json"
        ) from exc
    if set(modalities) != {"video", "state", "action", "language"}:
        raise ValueError("JSON modalities must define video, state, action, and language")
    tactile_options = {}
    if use_tactile:
        tactile = payload.get("tactile")
        if not tactile or not tactile.get("state_keys"):
            raise ValueError("--use_tactile requires tactile.state_keys in the modality JSON")
        keys = tactile["state_keys"]
        if len(set(keys)) != len(keys) or set(keys).intersection(modalities["state"].modality_keys):
            raise ValueError("Tactile state keys must be unique and separate from joint keys")
        modalities["state"].modality_keys.extend(keys)
        tactile_options = {
            "use_tactile": True,
            "tactile_state_keys": keys,
            "tactile_encoder": tactile.get("encoder", "finger_mlp"),
            "tactile_embed_dim": tactile.get("embed_dim", 32),
            "tactile_input_shape": tuple(tactile.get("input_shape", [2, 5, 9])),
        }
    return modalities, tactile_options


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    # Use tyro for clean CLI
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    from gr00t.data.embodiment_tags import EmbodimentTag

    ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
    embodiment_tag = ft_config.embodiment_tag.value

    # Python configs register as before; JSON settings are applied to this run.
    json_modalities, tactile_options = None, {}
    if ft_config.modality_config_path is not None:
        json_modalities, tactile_options = load_modality_config(
            ft_config.modality_config_path,
            embodiment_tag=embodiment_tag,
            use_tactile=ft_config.use_tactile,
        )
    elif ft_config.use_tactile:
        raise ValueError("--use_tactile requires --modality-config-path")
    if not ft_config.use_tactile and (
        ft_config.tactile_encoder is not None or ft_config.tactile_embed_dim is not None
    ):
        raise ValueError("Tactile encoder options require --use_tactile")

    dataset_paths = [path for path in ft_config.dataset_path.split(os.pathsep) if path]

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": dataset_paths,
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None
    if json_modalities is not None:
        config.data.modality_configs = deepcopy(config.data.modality_configs)
        config.data.modality_configs[embodiment_tag] = json_modalities
    for key, value in tactile_options.items():
        setattr(config.model, key, value)
    if ft_config.tactile_encoder is not None:
        config.model.tactile_encoder = ft_config.tactile_encoder
    if ft_config.tactile_embed_dim is not None:
        config.model.tactile_embed_dim = ft_config.tactile_embed_dim

    # overwrite with finetune config supplied by the user
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    config.model.use_percentiles = ft_config.use_percentiles
    if (ft_config.shortest_image_edge is None) != (ft_config.crop_fraction is None):
        raise ValueError("shortest_image_edge and crop_fraction must be set together")
    if ft_config.shortest_image_edge is not None:
        config.model.shortest_image_edge = ft_config.shortest_image_edge
        config.model.crop_fraction = ft_config.crop_fraction
        config.model.image_crop_size = None
        config.model.image_target_size = None
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch
    config.data.ds_weights_alpha = ft_config.ds_weights_alpha

    config.training.save_only_model = ft_config.save_only_model
    config.training.resume_from_checkpoint = ft_config.resume_from_checkpoint
    config.training.skip_weight_loading = ft_config.skip_weight_loading

    run(config)
