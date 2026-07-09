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
"""GrootN2 pipeline: model + dataset + collator setup."""

import json
import logging
from pathlib import Path

from groot.core.data.dataset.build_artifacts import DatasetBuildArtifacts
from groot.core.data.dataset.factory import DatasetFactory
from groot.core.training.configs.base_config import Config
from groot.core.training.configs.model.n2 import GrootN2Config
from groot.core.training.experiment.dist_utils import get_rank
from groot.core.training.models.base.model_pipeline import ModelPipeline
from groot.core.training.models.n2.model import GrootN2
from groot.core.training.models.n2.processor import GrootN2Processor
from groot.core.training.models.registry import register_model


class GrootN2Pipeline(ModelPipeline):
    model_class = GrootN2
    processor_class = GrootN2Processor

    def __init__(
        self,
        config: Config,
        save_cfg_dir: Path,
        *,
        dataset_build_artifacts: DatasetBuildArtifacts | None = None,
    ):
        super().__init__(
            config,
            dataset_build_artifacts=dataset_build_artifacts,
        )
        self.save_cfg_dir = save_cfg_dir

    @property
    def model_config(self):
        return self.config.model

    def setup(self):
        self.model = self._create_model()
        self.train_dataset, self.eval_dataset = self._create_dataset()
        self.data_collator = self.processor.collator

    def _create_model(self):
        import glob

        from safetensors.torch import load_file
        import torch

        mc = self.model_config
        start_ckpt = self.config.training.start_from_checkpoint

        if start_ckpt is not None:
            # Posttraining: construct model with CLI config (correct shapes from
            # the start — avoids DeepSpeed/CUDA graph issues from post-hoc overrides),
            # then load checkpoint weights.
            logging.info(f"[GrootN2] Posttraining from checkpoint: {start_ckpt}")
            model = self.model_class(mc)

            # Load ALL weights from checkpoint safetensors
            shard_files = sorted(glob.glob(f"{start_ckpt}/model*.safetensors"))
            state_dict = {}
            for f in shard_files:
                state_dict.update(load_file(f))
            logging.info(f"[GrootN2] Loaded {len(state_dict)} keys from {len(shard_files)} shards")

            # Handle view_embed view-count mismatch between the pretrain ckpt and the
            # posttrain config (view_embed.weight is [max_n_cameras, view_embed_dim]; it
            # is the only weight whose shape depends on max_n_cameras).
            vkey = "video_tokenizer.view_embed.weight"
            if vkey in state_dict:
                old_embed = state_dict[vkey]
                if old_embed.shape[0] < mc.max_n_cameras:
                    # Grow (e.g. 3 -> 4): keep learned views, init new ones as their mean.
                    new_embed = model.video_tokenizer.view_embed.weight.data.clone()
                    new_embed[:old_embed.shape[0]] = old_embed
                    new_embed[old_embed.shape[0]:] = old_embed.mean(dim=0)
                    state_dict[vkey] = new_embed
                    logging.info(
                        f"[GrootN2] Extended view_embed: {old_embed.shape[0]} -> {mc.max_n_cameras} "
                        f"(new cameras initialized as mean of existing)"
                    )
                elif old_embed.shape[0] > mc.max_n_cameras:
                    # Shrink (e.g. 4 -> 3): keep the first max_n_cameras view slots and drop
                    # the rest. Pure truncation (no zero-pad) so posttraining runs a genuine
                    # N-view model. Assumes the posttrain cameras map to the pretrain's first
                    # max_n_cameras view slots (view_embed re-learns during posttraining).
                    state_dict[vkey] = old_embed[: mc.max_n_cameras].clone()
                    logging.info(
                        f"[GrootN2] Truncated view_embed: {old_embed.shape[0]} -> {mc.max_n_cameras} "
                        f"(dropped view rows {mc.max_n_cameras}:{old_embed.shape[0]})"
                    )

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                logging.warning(f"[GrootN2] Missing keys ({len(missing)}): {missing[:5]}...")
            if unexpected:
                logging.warning(f"[GrootN2] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

            if hasattr(mc, "action_cfg") and mc.action_cfg:
                model.attention_mask_pattern = mc.action_cfg.get("attention_mask_pattern", None)
            logging.info(f"[GrootN2] Posttraining setup done (max_n_cameras={mc.max_n_cameras})")
        else:
            # Pretraining: fresh init + load base pretrained components
            model = self.model_class(mc)

            if mc.diffusion_model_cfg and mc.diffusion_model_cfg.get("diffusion_model_pretrained_path"):
                model.load_cosmos_pretrained(mc.diffusion_model_cfg["diffusion_model_pretrained_path"])

            if mc.vae_cfg and mc.vae_cfg.get("vae_pretrained_path"):
                vae_target = model.vae.model
                if hasattr(vae_target, "_orig_mod"):
                    vae_target = vae_target._orig_mod
                vae_target.load_state_dict(
                    torch.load(mc.vae_cfg["vae_pretrained_path"], map_location="cuda"), assign=True
                )

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(
            f"[GrootN2] Total: {total:,}, Trainable: {trainable:,} ({100 * trainable / total:.1f}%)"
        )
        return model

    def _create_dataset(self):
        mc = self.model_config
        self.processor = self.processor_class(
            modality_configs=self.config.data.modality_configs,
            image_target_size=mc.image_target_size,
            image_crop_fraction=getattr(mc, "image_crop_fraction", 0.95),
            color_jitter_params=getattr(mc, "color_jitter_params", None),
            codec_aug_params=getattr(mc, "codec_aug_params", None),
            formalize_language=getattr(mc, "formalize_language", False),
            max_language_length=getattr(mc, "max_language_length", 512),
            tokenizer_path=mc.tokenizer_path,
            max_n_cameras=getattr(mc, "max_n_cameras", 1),
            action_cfg=getattr(mc, "action_cfg", None),
            # CFG: training dropout to null_prompt + matching inference text_negative.
            language_dropout_prob=getattr(mc, "language_dropout_prob", 0.0),
            null_prompt=getattr(mc, "null_prompt", "low quality, blurry, static"),
        )
        self._dataset_factory = DatasetFactory(
            self.config.data,
            experiment_name=self.config.training.experiment_name,
            dataset_build_artifacts=self.dataset_build_artifacts,
        )
        train_dataset, eval_dataset = self._dataset_factory.build(processor=self.processor)
        if get_rank() == 0:
            stats = train_dataset.get_dataset_statistics()
            with open(self.save_cfg_dir / "dataset_statistics.json", "w") as f:
                json.dump({k: str(v) for k, v in stats.items()} if stats else {}, f, indent=2)
        return train_dataset, eval_dataset


register_model(GrootN2Config, GrootN2Pipeline)
