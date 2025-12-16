# Fine-tune on Custom Embodiments ("NEW_EMBODIMENT")

This guide demonstrates how to finetune GR00T on your own robot data and configuration. We provide a complete example for the Huggingface [SO-100](https://github.com/TheRobotStudio/SO-ARM100) robot under `examples/SO100`, which uses `demo_data/cube_to_bowl_5` as the demo dataset.

## Step 1: Prepare Your Data

Prepare your data in **GR00T-flavored LeRobot v2 format** by following the [data preparation guide](data_preparation.md). 

## Step 2: Prepare Your Modality Configuration

Define your own modality configuration by following the [modality config guide](data_config.md). Below is an example configuration that corresponds to the demo data:
```python
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import ModalityConfig


so100_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "front",
            "wrist",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "single_arm",
            "gripper",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=[
            "single_arm",
            "gripper",
        ],
        action_configs=[
            # single_arm
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # gripper
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(so100_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
```

**Important:** Register your modality configuration under the `EmbodimentTag.NEW_EMBODIMENT` tag:
```python
register_modality_config(so100_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
```

## Step 3: Run Fine-tuning

We'll use `gr00t/experiment/launch_finetune.py` as the entry point. Ensure that the uv environment is enabled before launching. You can do this by running the command `uv run bash <example_script_name>`.

### View Available Arguments
```bash
# Display all available arguments
python gr00t/experiment/launch_finetune.py --help
```

### Execute Fine-tuning
```bash
# Configure for single GPU
export NUM_GPUS=1
CUDA_VISIBLE_DEVICES=0 python \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path ./demo_data/cube_to_bowl_5 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/SO100/so100_config.py \
    --num-gpus $NUM_GPUS \
    --output-dir /tmp/so100 \
    --save-total-limit 5 \
    --save-steps 2000 \
    --max-steps 2000 \
    --use-wandb \
    --global-batch-size 32 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 4
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--base-model-path` | Path to the pre-trained base model checkpoint |
| `--dataset-path` | Path to your training dataset |
| `--embodiment-tag` | Tag to identify your robot embodiment |
| `--modality-config-path` | Path to user-specified modality config (required only for `NEW_EMBODIMENT` tag) |
| `--output-dir` | Directory where checkpoints will be saved |
| `--save-steps` | Save checkpoint every N steps |
| `--max-steps` | Total number of training steps |
| `--use-wandb` | Enable Weights & Biases logging for experiment tracking |

## Step 4: Open Loop Evaluation

After finetuning, evaluate the model's performance using open loop evaluation:
```bash
python gr00t/eval/open_loop_eval.py \
    --dataset-path ./demo_data/cube_to_bowl_5 \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path /tmp/so100/checkpoint-2000 \
    --traj-ids 0 \
    --action-horizon 16 \
    --steps 400 \
    --modality-keys single_arm gripper
```

### Example Evaluation Result

The evaluation generates visualizations comparing predicted actions against ground truth trajectories:

<img src="../media/open_loop_eval_so100.jpeg" width="800" alt="Open loop evaluation results showing predicted vs ground truth trajectories" />