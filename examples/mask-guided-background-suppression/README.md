# Mask-Guided Background Suppression

Mask-guided augmentations leverage per-frame segmentation masks to apply targeted image transformations during training. This enables **domain randomization** on specific regions (e.g., replacing backgrounds with noise, tinting foreground objects) without affecting the rest of the image.

This feature is controlled via the `--extra_augmentation_config` argument, which accepts a JSON string specifying which mask regions to augment and how.

---

## Prerequisites

1. **Segmentation masks** must be pre-generated and stored alongside your dataset. The dataset's `info.json` must include a `mask_path` template, and `modality.json` must define a `"mask"` section mapping camera views.

2. **Albumentations transforms** must be enabled (`--use_albumentations_transforms` is automatically set when `--extra_augmentation_config` is provided via `launch_finetune.py`).

---

## Supported Augmentation Types

### 1. Background Noise Transform

Replaces pixels in specified mask regions with **random RGB noise**. Useful for sim-to-real transfer or preventing the model from overfitting to static backgrounds.

| Parameter | Type | Description |
|-----------|------|-------------|
| `target_mask_values` | `list[int]` | Mask label values to replace with noise (e.g., `[0]` for background) |
| `p` | `float` | Probability of applying the transform per frame (0.0 to 1.0) |

### 2. Masked Region Color Transform

Applies a **random color tint** to pixels in specified mask regions. Useful for augmenting the appearance of specific objects (e.g., tables, tools) to improve color generalization.

| Parameter | Type | Description |
|-----------|------|-------------|
| `target_mask_values` | `list[int]` | Mask label values to apply the tint to (e.g., `[4]`, `[5]`) |
| `p` | `float` | Probability of applying the transform per frame (0.0 to 1.0) |
| `alpha_range` | `[min, max]` | Range for blending intensity between original and tint color (default: `[0.3, 1.0]`) |

---

## Configuration Format

The `--extra_augmentation_config` argument takes a JSON string with two optional keys:

```json
{
    "background_noise_transforms": [
        {"target_mask_values": [0], "p": 0.9}
    ],
    "masked_region_transforms": [
        {"target_mask_values": [4], "p": 1.0, "alpha_range": [0.0, 1.0]}
    ]
}
```

Multiple transforms of each type can be specified (e.g., different mask values with different probabilities).

---

## Quick Start with Demo Data

The included demo dataset `demo_data/cube_to_bowl_5_with_mask` contains a single episode with front and wrist camera views, along with pre-generated segmentation masks. The masks were generated using [SAM 3](https://github.com/facebookresearch/sam3) with the text prompt `"background"`, then converted so that background pixels = `0` and foreground pixels = `1` (see [Generating mask files](#generating-mask-files) below).

### 1. Background noise only

Replace background (mask=0) with random noise:

```bash
uv run python test_extra_augmentation.py \
    --dataset_path ../../demo_data/cube_to_bowl_5_with_mask \
    --embodiment_tag NEW_EMBODIMENT \
    --modality_config_path so101_config.py \
    --extra_augmentation_config '{"background_noise_transforms": [{"target_mask_values": [0], "p": 1.0}]}'
```

### 2. Background noise + foreground color tint

Apply both transforms together:

```bash
uv run python test_extra_augmentation.py \
    --dataset_path ../../demo_data/cube_to_bowl_5_with_mask \
    --embodiment_tag NEW_EMBODIMENT \
    --modality_config_path so101_config.py \
    --extra_augmentation_config '{"background_noise_transforms": [{"target_mask_values": [0], "p": 1.0}], "masked_region_transforms": [{"target_mask_values": [1], "p": 1.0, "alpha_range": [0.3, 1.0]}]}' \
    --output_dir /tmp/augmentation_vis --num_frames 5
```

Both commands save side-by-side comparison images (**Original | Augmented | Mask**) under `output_dir/<view_name>/`, with frames sampled evenly across the episode.

### 3. Fine-tune with mask-guided augmentation

```bash
export NUM_GPUS=8

torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.7-3B \
    --dataset_path <YOUR_DATASET_WITH_MASKS> \
    --embodiment_tag <YOUR_EMBODIMENT_TAG> \
    --num_gpus $NUM_GPUS \
    --output_dir /tmp/mask_augmentation_run \
    --save_steps 1000 \
    --save_total_limit 5 \
    --max_steps 20000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --use_wandb \
    --global_batch_size 640 \
    --dataloader_num_workers 4 \
    --extra_augmentation_config '{"background_noise_transforms": [{"target_mask_values": [0], "p": 0.9}], "masked_region_transforms": [{"target_mask_values": [4], "p": 1.0, "alpha_range": [0, 1]}]}'
```

---

## Dataset Setup

To use mask-guided augmentation with your own dataset, ensure:

1. **Mask files** are stored as `.npz` files under a `masks/` directory, following the same chunk/episode structure as videos. Each `.npz` contains a single `uint8` array of shape `(num_frames, H, W)` where each pixel holds an integer semantic label (e.g., `0` = background, `1` = object A, `2` = object B).

   ```
   masks/
   └── chunk-000/
       └── observation.images.front/
           └── episode_000000_masks.npz
   ```

   See [Generating mask files](#generating-mask-files) below for how to produce these files.

2. **`info.json`** includes a `mask_path` template:

   ```json
   {
       "mask_path": "masks/chunk-{episode_chunk:03d}/{mask_key}/episode_{episode_index:06d}_masks.npz"
   }
   ```

3. **`modality.json`** includes a `"mask"` section mapping view names to their original keys. The keys should match the actual camera view names in your dataset:

   ```json
   {
       "mask": {
           "<view_name>": {
               "original_key": "<observation.images.xxx>"
           }
       }
   }
   ```

   For example, if your dataset has `front` and `wrist` cameras:

   ```json
   {
       "mask": {
           "front": {
               "original_key": "observation.images.front"
           },
           "wrist": {
               "original_key": "observation.images.wrist"
           }
       }
   }
   ```

---

## Generating Mask Files

You can generate mask files using any video segmentation model that produces per-pixel labels. The demo masks in this example were created with [SAM 3](https://github.com/facebookresearch/sam3) (see the [video predictor example](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_video_predictor_example.ipynb) for SAM 3 usage). The workflow was:

1. Run SAM 3 on each episode video with a text prompt such as `"background"`. SAM 3 returns per-frame binary masks via `propagate_in_video`.
2. Convert the binary masks into the label format expected by this pipeline (`0` = background, non-zero = foreground categories) and save as `.npz`:

```python
import numpy as np

# sam3_binary_masks: (num_frames, H, W) bool array from SAM 3 (True where prompt matched)
# For a "background" prompt, invert so that background=0 and foreground=1:
label_masks = (~sam3_binary_masks).astype(np.uint8)

# For multiple prompts, merge into one label array instead:
# label_masks = np.zeros((num_frames, H, W), dtype=np.uint8)
# label_masks[prompt_0_masks] = 1
# label_masks[prompt_1_masks] = 2

np.savez_compressed("episode_000000_masks.npz", label_masks)
```

The pipeline loads the array from the `.npz` file (it expects the key `arr_0`, which is the default for `np.savez_compressed`). A single `.npy` file containing the `(num_frames, H, W)` array also works.

---

## How It Works

The augmentation pipeline applies mask-based transforms **per-frame** before the standard augmentations (crop, resize, color jitter, etc.):

1. For each frame, the corresponding segmentation mask is loaded.
2. `BackgroundNoiseTransform` replaces all pixels where `mask == target_value` with random RGB noise.
3. `MaskedColorTransform` blends a random color into all pixels where `mask == target_value`, controlled by `alpha_range`.
4. Standard augmentations (shared across views via replay) are then applied on top.

This ordering ensures that mask-guided augmentations are applied independently per frame, while standard augmentations remain consistent across camera views within the same timestep.
