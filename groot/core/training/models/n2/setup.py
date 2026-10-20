# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
from groot.core.training.models.n2.processor import (
    EMBODIMENT_TAG_TO_PROJECTOR_INDEX,
    GrootN2Processor,
)
from groot.core.training.models.registry import register_model
from groot.core.utils.storage.msc import resolve_model_path


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
            start_ckpt = resolve_model_path(start_ckpt, "training.start_from_checkpoint")
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
            if mc.pretrained_view_indices is not None and self.resume_checkpoint is None:
                self._initialize_camera_embeddings(model, state_dict, start_ckpt)
            elif vkey in state_dict:
                old_embed = state_dict[vkey]
                if old_embed.shape[0] < mc.max_n_cameras:
                    # Grow (e.g. 3 -> 4): keep learned views, init new ones as their mean.
                    new_embed = model.video_tokenizer.view_embed.weight.data.clone()
                    new_embed[: old_embed.shape[0]] = old_embed
                    new_embed[old_embed.shape[0] :] = old_embed.mean(dim=0)
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
                logging.warning(
                    f"[GrootN2] Unexpected keys ({len(unexpected)}): {unexpected[:5]}..."
                )

            if hasattr(mc, "action_cfg") and mc.action_cfg:
                model.attention_mask_pattern = mc.action_cfg.get("attention_mask_pattern", None)
            logging.info(f"[GrootN2] Posttraining setup done (max_n_cameras={mc.max_n_cameras})")
        else:
            # Pretraining: fresh init + load base pretrained components
            model = self.model_class(mc)

            if mc.diffusion_model_cfg and mc.diffusion_model_cfg.get(
                "diffusion_model_pretrained_path"
            ):
                model.load_cosmos_pretrained(
                    resolve_model_path(
                        mc.diffusion_model_cfg["diffusion_model_pretrained_path"],
                        "model.diffusion_model_cfg.diffusion_model_pretrained_path",
                    )
                )

            if mc.vae_cfg and mc.vae_cfg.get("vae_pretrained_path"):
                vae_target = model.vae.model
                vae_target = getattr(vae_target, "_orig_mod", vae_target)
                vae_path = resolve_model_path(
                    mc.vae_cfg["vae_pretrained_path"],
                    "model.vae_cfg.vae_pretrained_path",
                )
                vae_target.load_state_dict(
                    torch.load(
                        vae_path,
                        map_location="cuda",
                    ),
                    assign=True,
                )

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(
            f"[GrootN2] Total: {total:,}, Trainable: {trainable:,} ({100 * trainable / total:.1f}%)"
        )
        return model

    def _initialize_camera_embeddings(self, model, state_dict, start_ckpt):
        """Apply the requested view mapping once, before the initial weight load."""
        vkey = "video_tokenizer.view_embed.weight"
        indices = self.model_config.pretrained_view_indices
        target = model.video_tokenizer.view_embed.weight.detach().clone()
        config_path = Path(start_ckpt) / "config.json"
        saved = {}
        if config_path.exists():
            with config_path.open() as f:
                saved = json.load(f)
        old_embed = state_dict.get(vkey)
        if saved.get("pretrained_view_indices") is not None:
            # This checkpoint already learned the target camera slots. Never gather
            # or randomize them again when it is used as a weights-only warm start.
            if (
                tuple(saved["pretrained_view_indices"]) != indices
                or old_embed is None
                or old_embed.shape != target.shape
            ):
                raise ValueError("Camera initialization differs from the saved checkpoint")
            logging.info("[GrootN2] Keeping checkpoint's already initialized camera embeddings")
            return
        for slot, source in enumerate(indices):
            if source == -1:
                continue
            if old_embed is None:
                raise ValueError(f"pretrained_view_indices row {source} is absent from the source")
            if old_embed.ndim != 2 or old_embed.shape[1] != target.shape[1]:
                raise ValueError("Pretrained camera embedding width does not match the model")
            if source >= old_embed.shape[0]:
                raise ValueError(f"pretrained_view_indices row {source} is absent from the source")
            target[slot] = old_embed[source]
        state_dict[vkey] = target
        logging.info("[GrootN2] Camera initialization source rows (-1=fresh): %s", indices)

    def _create_processor(self):
        """Construct the selected processor with the canonical N2 configuration."""
        mc = self.model_config
        return self.processor_class(
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
            embodiment_id_mapping=self._checkpoint_embodiment_id_mapping(),
        )

    def _checkpoint_embodiment_id_mapping(self) -> dict[str, int] | None:
        """Keep every embodiment on the projector row its loaded weights were trained on.

        Saved mappings take precedence over the code table; resume_checkpoint takes
        precedence over start_from_checkpoint. New tags need an explicit code-table
        row, and reserved-row reuse requires training.allow_pretrained_projector_row_reuse.
        Returns None for fresh pretraining, single-row heads, or legacy checkpoints
        without mappings. Legacy selected rows are validated, but reservations are unknown.
        """
        action_cfg = getattr(self.model_config, "action_cfg", None)
        if action_cfg is None or action_cfg.get("num_embodiments", 1) == 1:
            return None
        if self.resume_checkpoint is not None:
            ckpt, source = self.resume_checkpoint, "resume checkpoint"
        elif self.config.training.start_from_checkpoint is not None:
            ckpt = resolve_model_path(
                self.config.training.start_from_checkpoint, "training.start_from_checkpoint"
            )
            source = "start checkpoint"
        else:
            return None
        # A missing legacy mapping cannot prove whether a row was used in pretraining.
        config_path = Path(ckpt) / "processor_config.json"
        ckpt_mapping = None
        if config_path.exists():
            with open(config_path) as f:
                ckpt_mapping = json.load(f)["processor_kwargs"].get("embodiment_id_mapping")
        if ckpt_mapping is None:
            logging.warning(
                f"[GrootN2] {source} {ckpt} has no saved embodiment_id_mapping; "
                f"using the code table; checkpoint row reservations cannot be checked"
            )
        elif not isinstance(ckpt_mapping, dict) or any(
            not isinstance(tag, str) or type(row) is not int or row < 0
            for tag, row in ckpt_mapping.items()
        ):
            raise ValueError(f"[GrootN2] {source} {ckpt} has an invalid embodiment_id_mapping")
        saved_mapping = ckpt_mapping or {}
        mapping = {**EMBODIMENT_TAG_TO_PROJECTOR_INDEX, **saved_mapping}
        allow_reuse = (
            getattr(self.config.training, "allow_pretrained_projector_row_reuse", False) is True
        )
        for tag in self.config.data.modality_configs:
            if tag not in mapping:
                raise ValueError(
                    f"[GrootN2] {tag}: no explicit projector row in {source} {ckpt} or "
                    "EMBODIMENT_TAG_TO_PROJECTOR_INDEX; implicit row-0 fallback is not allowed "
                    "for multi-row posttraining. Register an explicit row or use a saved tag."
                )
            row = mapping[tag]
            if type(row) is not int or not 0 <= row < action_cfg["num_embodiments"]:
                raise ValueError(
                    f"[GrootN2] {tag}: invalid projector row {row!r} for "
                    f"num_embodiments={action_cfg['num_embodiments']} ({source} {ckpt})"
                )
            code_row = EMBODIMENT_TAG_TO_PROJECTOR_INDEX.get(tag)
            if tag in saved_mapping:
                if code_row is not None and code_row != row:
                    logging.warning(
                        f"[GrootN2] {tag}: projector row {row} from {source} "
                        f"(code table says {code_row})"
                    )
                continue
            reserved_by = sorted(t for t, r in saved_mapping.items() if r == row)
            if not reserved_by:
                logging.info(f"[GrootN2] {tag}: not in {source}; code-table projector row {row}")
                continue
            conflict = (
                f"[GrootN2] {tag}: absent from {source} {ckpt}, but code-table "
                f"projector row {row} is reserved by checkpoint tags {reserved_by}"
            )
            if not allow_reuse:
                raise ValueError(
                    f"{conflict}. Use the matching saved tag or an unreserved row. "
                    "For deliberate reuse only, explicitly set "
                    "--training.allow_pretrained_projector_row_reuse."
                )
            logging.warning(
                "%s; explicitly allowed by training.allow_pretrained_projector_row_reuse=True",
                conflict,
            )
        return mapping if ckpt_mapping is not None else None

    def _create_dataset(self):
        self.processor = self._create_processor()
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
