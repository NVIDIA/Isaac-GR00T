# GR00T DROID

> **Note:** The DROID dataset contains multiple language instruction paraphrases per episode (`language_instruction`, `language_instruction_2`, `language_instruction_3`). These are used for language augmentation during training. At inference time, only the first language key is used.

The N1.7 base model supports DROID inference out of the box via the `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT` embodiment tag.

## 1. Inference Server:

On a machine with a sufficiently powerful GPU, start the policy server from the root folder of this repo:

```bash
uv run python gr00t/eval/run_gr00t_server.py --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT --model-path nvidia/GR00T-N1.7-3B
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
python examples/DROID/main_gr00t.py --external-camera="left" # or "right"
```
