#!/usr/bin/env python3
"""Smoke test: apply extra_augmentation_config to raw frames and save comparison images."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model.gr00t_n1d7.image_augmentations import (
    apply_with_replay,
    build_image_transformations_albumentations,
)
import numpy as np
from PIL import Image


def save_comparison(original, augmented, mask, output_path):
    orig_arr = np.array(original)
    aug_arr = augmented.transpose(1, 2, 0) if augmented.shape[0] == 3 else augmented

    panels = [orig_arr, aug_arr]
    if mask is not None:
        mask_vis = np.where(mask[..., None] > 0, 255, 0).astype(np.uint8)
        mask_vis = np.broadcast_to(mask_vis, (*mask.shape[:2], 3)).copy()
        if mask_vis.shape[:2] != orig_arr.shape[:2]:
            mask_vis = np.array(
                Image.fromarray(mask_vis).resize(
                    (orig_arr.shape[1], orig_arr.shape[0]), Image.NEAREST
                )
            )
        panels.append(mask_vis)

    h = panels[0].shape[0]
    resized = []
    for p in panels:
        if p.shape[0] != h:
            new_w = int(p.shape[1] * h / p.shape[0])
            p = np.array(Image.fromarray(p).resize((new_w, h), Image.BILINEAR))
        resized.append(p)

    Image.fromarray(np.concatenate(resized, axis=1)).save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--embodiment_tag", required=True)
    parser.add_argument("--modality_config_path", default=None)
    parser.add_argument("--extra_augmentation_config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/tmp/augmentation_vis")
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--video_backend", type=str, default="torchcodec")
    args = parser.parse_args()

    if args.modality_config_path:
        path = Path(args.modality_config_path)
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)

    embodiment_tag = EmbodimentTag[args.embodiment_tag].value
    modality_configs = MODALITY_CONFIGS[embodiment_tag]
    extra_aug_config = json.loads(args.extra_augmentation_config)

    train_transform, _ = build_image_transformations_albumentations(
        image_target_size=[224, 224],
        image_crop_size=[224, 224],
        random_rotation_angle=0,
        color_jitter_params=None,
        shortest_image_edge=512,
        crop_fraction=0.95,
        extra_augmentation_config=extra_aug_config,
    )

    loader = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality_configs,
        video_backend=args.video_backend,
    )
    episode_df = loader[0]

    video_cols = [c for c in episode_df.columns if c.startswith("video.")]
    mask_cols = [c for c in episode_df.columns if c.startswith("mask.")]
    print(f"Video columns: {video_cols}")
    print(f"Mask columns:  {mask_cols}")

    num_frames = min(args.num_frames, len(episode_df))
    frame_indices = np.linspace(0, len(episode_df) - 1, num_frames, dtype=int)

    os.makedirs(args.output_dir, exist_ok=True)

    for vcol in video_cols:
        view_name = vcol.replace("video.", "")
        mcol = f"mask.{view_name}"
        has_mask = mcol in mask_cols

        view_dir = os.path.join(args.output_dir, view_name.replace(".", "_"))
        os.makedirs(view_dir, exist_ok=True)

        for fidx in frame_indices:
            orig_img = episode_df[vcol].iloc[fidx]
            mask_arr = np.array(episode_df[mcol].iloc[fidx]) if has_mask else None
            masks_list = [mask_arr] if mask_arr is not None else None

            transformed, _ = apply_with_replay(train_transform, [orig_img], masks_list)
            aug_arr = transformed[0].numpy()

            out_path = os.path.join(view_dir, f"frame_{fidx:04d}.png")
            save_comparison(orig_img, aug_arr, mask_arr, out_path)
            print(f"  Saved: {out_path}")

    print(f"\nDone! {num_frames} frames x {len(video_cols)} views saved to {args.output_dir}")

    print("\n" + "=" * 60)
    print("Testing full training pipeline (processor + dataloader) ...")
    print("=" * 60)

    from gr00t.configs.base_config import get_default_config
    from gr00t.data.dataset.factory import DatasetFactory
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    config = get_default_config()
    config = config.load_dict(
        {
            "data": {
                "download_cache": False,
                "video_backend": args.video_backend,
                "datasets": [
                    {
                        "dataset_paths": [args.dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.model.extra_augmentation_config = extra_aug_config
    config.model.use_albumentations_transforms = True

    processor = Gr00tN1d7Processor(
        modality_configs=config.data.modality_configs,
        statistics=None,
        image_crop_size=config.model.image_crop_size,
        image_target_size=config.model.image_target_size,
        random_rotation_angle=config.model.random_rotation_angle,
        color_jitter_params=config.model.color_jitter_params,
        model_name=config.model.model_name,
        model_type=config.model.backbone_model_type,
        formalize_language=config.model.formalize_language,
        max_state_dim=config.model.max_state_dim,
        max_action_dim=config.model.max_action_dim,
        apply_sincos_state_encoding=config.model.apply_sincos_state_encoding,
        max_action_horizon=config.model.action_horizon,
        use_albumentations=config.model.use_albumentations_transforms,
        extra_augmentation_config=config.model.extra_augmentation_config,
        shortest_image_edge=config.model.shortest_image_edge,
        crop_fraction=config.model.crop_fraction,
        use_relative_action=config.model.use_relative_action,
    )
    processor.train()

    dataset_factory = DatasetFactory(config=config)
    train_dataset, _ = dataset_factory.build(processor=processor)
    sample = next(iter(train_dataset))

    print(f"Sample keys: {list(sample.keys())}")
    print(f"VLM keys:    {list(sample['vlm_content'].keys())}")
    for k, v in sample.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    print("\nPipeline test PASSED!")


if __name__ == "__main__":
    main()
