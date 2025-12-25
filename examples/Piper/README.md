# Finetuning Piper Robot Model

This guide shows how to finetune dataset collected from Piper robot, and evaluate the model.

## Configuration Types

This example supports multiple configuration types:

| Configuration | Description | Use Case |
|---------------|-------------|----------|
| `joint_space` | Joint angle control | Direct joint control |
| `task_space_rot6d` | End-effector pose with 6D rotation | Continuous rotation representation |
| `task_space_rpy` | End-effector pose with RPY angles | Euler angle representation |
| `task_space_quat` | End-effector pose with quaternion | Quaternion representation |

## Dataset

Prepare your dataset in LeRobot format with the following structure:

**Required modalities:**
- Video: `hand_camera`, `third_camera`
- State/Action (joint space): `joint_states` (6 DOF), `gripper_distance` (1 DOF)
- State/Action (task space): `eef_pos` (3 DOF), `eef_rpy_external` (3 DOF), `gripper_distance` (1 DOF)

## Handling the Dataset

1. Convert your dataset to LeRobot v2 format if needed:

```bash
uv run python scripts/lerobot_conversion/convert_v3_to_v2.py --repo-id your/dataset \
  --root examples/Piper/piper_dataset_lerobot
```

2. Copy the appropriate modality.json file to the dataset:

For joint space:
```bash
cp examples/Piper/modality_joint_space.json examples/Piper/piper_dataset_lerobot/meta/modality.json
```

For task space:
```bash
cp examples/Piper/modality_task_space.json examples/Piper/piper_dataset_lerobot/meta/modality.json
```

## Finetuning

Run the finetuning script:
```bash
uv run bash examples/Piper/finetune_piper.sh
```

To use a different configuration type, modify `piper_config.py` and uncomment the desired configuration registration.

## Open-Loop Evaluation

Evaluate the finetuned model with the following command:
```bash
uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path examples/Piper/piper_dataset_lerobot \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path /tmp/piper_finetune/checkpoint-10000 \
  --traj-ids 0 \
  --action-horizon 16 \
  --steps 400
```

## Closed-Loop Evaluation

For real robot deployment, use the Policy API:

1. Start policy server:
```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path /tmp/piper_finetune/checkpoint-10000 \
  --embodiment-tag NEW_EMBODIMENT
```

2. Run your robot-specific evaluation client.

## Configuration Details

### Joint Space Configuration
- State: 6 joint positions + 1 gripper distance
- Action: 6 joint positions + 1 gripper distance
- Normalization: min_max for all

### Task Space Configuration
- State: 3 eef position + 3 rotation (RPY) + 1 gripper distance
- Action: 3 eef position + 3 rotation (converted to 6D/RPY/quaternion) + 1 gripper distance
- Normalization: min_max for position and gripper, rotation conversion for orientation

