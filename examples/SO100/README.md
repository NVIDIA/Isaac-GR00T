# Finetuning Models for the SO100/SO101 Robot

This guide shows how to finetune a dataset collected from the [SO101](https://huggingface.co/docs/lerobot/en/so101) robot, and evaluate the model on the real robot.

## Setup

Set an environment variable pointing at your local clone of this repository, then `cd` into it. Every step below references `$GR00T_REPO`, so export it in each terminal you open (or add it to your shell profile).

```bash
export GR00T_REPO=~/Isaac-GR00T # update to match your path
cd "$GR00T_REPO"
```

## Dataset

To collect the dataset via teleoperation, calibrate the robot, and determine camera indices, please refer to the official documentation in lerobot: https://huggingface.co/docs/lerobot/il_robots

If you do not have a dataset, you can use this one as a basic test of the workflow.

**Dataset Path:** [izuluaga/finish_sandwich](https://huggingface.co/datasets/izuluaga/finish_sandwich)

Visualize it with this [link](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fizuluaga%2Ffinish_sandwich%2Fepisode_0)

## Converting the Dataset

1. From the repository root, run the following command to convert the dataset to the LeRobot v2 format necessary for finetuning.

```bash
cd "$GR00T_REPO"
uv run --project scripts/lerobot_conversion \
  python scripts/lerobot_conversion/convert_v3_to_v2.py \
  --repo-id izuluaga/finish_sandwich \
  --root examples/SO100/finish_sandwich_lerobot
```

2. Copy the `modality.json` file for the SO100 to the root of the dataset.
```bash
cp examples/SO100/modality.json examples/SO100/finish_sandwich_lerobot/izuluaga/finish_sandwich/meta/modality.json
```

## Finetuning the Model

1. Run the shared finetune launcher directly, this will use relative actions by default for all axes except the gripper.
```bash
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path examples/SO100/finish_sandwich_lerobot/izuluaga/finish_sandwich \
  --modality-config-path examples/SO100/so100_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /tmp/so100_finetune
```

## Open-Loop Evaluation

1. Evaluate the finetuned model with the following command:

```bash
uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path examples/SO100/finish_sandwich_lerobot/izuluaga/finish_sandwich/ \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path /tmp/so100_finetune/checkpoint-10000 \
  --traj-ids 0 \
  --execution-horizon 16 \
  --steps 400
```

2. The script saves one plot per trajectory to `/tmp/open_loop_eval/traj_<id>.jpeg` by default (pass `--save-plot-path` to override). Open the output directory to view the results:

```bash
xdg-open /tmp/open_loop_eval # macOS: open /tmp/open_loop_eval
```

### Evaluation Results

The evaluation produces visualizations comparing predicted actions against ground truth trajectories:

<img src="../../media/open_loop_eval_so100.jpg" width="800" alt="Open-loop evaluation results showing predicted vs ground truth trajectories" />

To read these numbers and decide whether your fine-tune is working, see [Interpreting the Result: Is My Fine-tune Working?](../../getting_started/finetune_new_embodiment.md#interpreting-the-result-is-my-fine-tune-working).

## Closed-Loop Evaluation

Please refer to [eval_so100.py](../../gr00t/eval/real_robot/SO100/eval_so100.py) for how to write SO100 deployment code using Policy API.

1. From the `gr00t/eval/real_robot/SO100` directory, set up client side dependencies:

```bash
cd "$GR00T_REPO/gr00t/eval/real_robot/SO100"
uv sync
uv pip install --no-deps -e ../../../../
```

2. Start the policy server:
```bash
cd "$GR00T_REPO"
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path /tmp/so100_finetune/checkpoint-10000 \
  --embodiment-tag NEW_EMBODIMENT 
```

3. In a second terminal, navigate to the `gr00t/eval/real_robot/SO100` directory.
```bash
cd "$GR00T_REPO/gr00t/eval/real_robot/SO100"
```

4. Run the eval script as the client from the `gr00t/eval/real_robot/SO100` environment created above:
```bash
# Update these to match the address and indices assigned by your OS
ROBOT_PORT=/dev/ttyACM2
ROBOT_ID=orange_follower
WRIST_CAM_IDX=2
FRONT_CAM_IDX=6
# Task specific prompt
PROMPT="finish the ham cheese olives sandwich"

uv run --no-sync python eval_so100.py \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id="$ROBOT_ID" \
  --robot.cameras="{ front: {type: opencv, index_or_path: $FRONT_CAM_IDX, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: $WRIST_CAM_IDX, width: 640, height: 480, fps: 30}}" \
  --policy_host=localhost \
  --policy_port=5555 \
  --lang_instruction="$PROMPT"
```
