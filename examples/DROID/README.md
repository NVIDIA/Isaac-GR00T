# Gr00t N1.6-DROID

We provide a checkpoint that is post-trained on the [DROID](https://droid-dataset.github.io/) dataset - [GR00T-N1.6-DROID](https://huggingface.co/nvidia/GR00T-N1.6-DROID). Follow the instructions below to run inference for this model.

## 1. Inference Server:

On a machine with a sufficiently powerful GPU, start the policy server from the root folder of this repo:

```bash
uv run python gr00t/eval/run_gr00t_server.py --embodiment-tag OXE_DROID --use_sim_policy_wrapper --model-path=nvidia/GR00T-N1.6-DROID
```

## 2. Control Script:

1. Install the DROID package on the robot control laptop/workstation - [instructions](https://droid-dataset.github.io/droid/software-setup/host-installation.html#configuring-the-laptopworkstation)

2. Install dependencies for the Gr00t control script in the environment from 1.:
```bash
pip install tyro moviepy==1.0.3 pydantic numpy==1.26.4
```

3. Enter the camera IDs for your ZED cameras in `examples/DROID/main_gr00t.py`.

3. Start the control script:
```bash
python examples/DROID/main_gr00t.py --external_camera="left" # or "right"
```
