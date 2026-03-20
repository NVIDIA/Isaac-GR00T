# GR00T DROID

The N1.7 base model supports DROID inference out of the box via the `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT` pretrain tag. A finetuned checkpoint is also available at [`nvidia/GR00T-N1.7-DROID`](https://huggingface.co/nvidia/GR00T-N1.7-DROID).

> **Note:** The DROID dataset contains multiple language instruction paraphrases per episode (`language_instruction`, `language_instruction_2`, `language_instruction_3`). These are used for language augmentation during training. At inference time, only the first language key is used.

## Data Format

The DROID embodiment expects the following modality structure:

| Modality | Keys | Dimensions |
|----------|------|------------|
| Video | `exterior_image_1_left`, `wrist_image_left` | 2 cameras |
| State | `eef_9d`, `gripper_position`, `joint_position` | 9D + 1D + 7D = 17D |
| Action | `eef_9d`, `gripper_position`, `joint_position` | 9D + 1D + 7D = 17D |
| Language | `annotation.language.language_instruction` | text |

Action representations:
- `eef_9d`: relative end-effector (XYZ + rotation 6D)
- `gripper_position`: absolute (1D)
- `joint_position`: relative joint positions (7D)

### Preparing DROID Demo Data

The full DROID dataset ([lerobot/droid_1.0.1](https://huggingface.co/datasets/lerobot/droid_1.0.1)) is ~358 GB with 95k+ episodes in LeRobot v3.0 format. To create a small sample for testing:

```bash
uv pip install jsonlines   # one-time dependency
python scripts/download_droid_sample.py
```

This downloads the first data/video chunk (~170 MB) and extracts 3 episodes into `demo_data/droid_sample/` in GR00T LeRobot v2.0 format.

**Key conversion notes:**
- Source is LeRobot v3.0 (consolidated parquet + concatenated videos) — the script converts to v2.0 (per-episode parquet + per-episode mp4).
- Video keys in the raw dataset (`exterior_1_left`, `wrist_left`) differ from the model config keys (`exterior_image_1_left`, `wrist_image_left`). The data loader auto-maps by position — no manual renaming needed.
- Language instructions are loaded via the `task_index` column mapped through `tasks.jsonl`.

## 1. Standalone Inference (with demo data)

After preparing demo data, run inference directly (no server needed):

```bash
uv run python scripts/deployment/standalone_inference_script.py \
    --model-path nvidia/GR00T-N1.7-3B \
    --dataset-path demo_data/droid_sample \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --traj-ids 0 1 \
    --inference-mode pytorch \
    --action-horizon 8
```

> **Note:** Episode 0 may have an empty language instruction. If inference fails on episode 0, try `--traj-ids 1 2`.

Expected zero-shot performance on the base model (not finetuned):

| Metric | Value |
|--------|-------|
| Average MSE | ~0.0149 |
| Average MAE | ~0.0753 |
| Inference per step (base) | ~262 ms (H100) |
| Inference per step (finetuned) | ~253 ms (H100) |

## 2. Inference Server (for real-world deployment)

### Using the base model (zero-shot):

```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.7-3B \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
```

### Using the finetuned model:

```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.7-DROID \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
```

## 3. Robot Control Script

1. Install the DROID package on the robot control laptop/workstation — [instructions](https://droid-dataset.github.io/droid/software-setup/host-installation.html#configuring-the-laptopworkstation)

2. Install dependencies for the GR00T control script in the environment from step 1:
```bash
pip install tyro moviepy==1.0.3 pydantic numpy==1.26.4
```

3. Enter the camera IDs for your ZED cameras in `examples/DROID/main_gr00t.py`.

4. Start the control script:
```bash
python examples/DROID/main_gr00t.py --external-camera="left" # or "right"
```
