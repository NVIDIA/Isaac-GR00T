# G1 LocoManipulation Benchmark

This is our in-house loco-manipulation task built on MuJoCo, using the Unitree G1 humanoid robot with whole-body control. The task requires the robot to navigate, pick up objects, and place them at target locations while maintaining balance and coordination across the entire body.

---

# G1 LocoManipulation evaluation benchmark result

| Task              | Success Rate |
|-------------------|--------------|
| PnPAppleToPlate   | 58%          |

**Note:** This task has high evaluation variance. Fluctuations of **±15%** are expected.

You can skip (1) and directly evaluate with a trained checkpoint: `https://huggingface.co/nvidia/GR00T-N1.6-G1-PnPAppleToPlate`

# 1. (Optional) Finetune model on the GR00T WholeBodyControl example dataset

To reproduce our finetune results, use the following commands to setup dataset and launch finetune experiments:

```bash
cd examples/GR00T-WholeBodyControl

# Clone the dataset repo without downloading files
git clone --filter=blob:none --no-checkout https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
cd PhysicalAI-Robotics-GR00T-X-Embodiment-Sim

# Initialize sparse-checkout FIRST
git sparse-checkout init --cone

# Set which folder to download
git sparse-checkout set unitree_g1.LMPnPAppleToPlateDC

# Now checkout the files
git checkout

# Pull the LFS files
git lfs pull

cd ../../../
uv run bash examples/GR00T-WholeBodyControl/finetune_g1.sh
```

# 2. Evaluate checkpoint

First, setup the evaluation simulation environment. This only needs to run once for each simulation benchmark. After it's done, we only need to launch server and client.

```bash
apt-get update
apt-get install libegl1-mesa-dev libglu1-mesa
bash gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh
```

Then, run client server evaluation under the project root directory in separate terminals:

**Terminal 1 - Server:**

You can use either a local finetuned checkpoint path or the remote finetuned checkpoint (provided by us):

**Option 1: Local finetuned checkpoint**
```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path /tmp/g1_finetune/checkpoint-10000/ \
    --embodiment-tag UNITREE_G1 \
    --use-sim-policy-wrapper
```

**Option 2: Remote finetuned checkpoint (directly runnable)**
```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-G1-PnPAppleToPlate \
    --embodiment-tag UNITREE_G1 \
    --use-sim-policy-wrapper
```

**Terminal 2 - Client:**
```bash
gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --max_episode_steps=1440 \
    --env_name gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc \
    --n_action_steps 20 \
    --n_envs 5
```

# Full task list

- `gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc`



# Real-robot Fine-tuning

When working with real-robot data, you have two options depending on your setup:

**Option 1: Using GR00T-WholeBodyControl repo for data collection**

If you collected your data using the [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl), you can leverage the `UNITREE_G1` embodiment tag. This is a pre-trained embodiment that comes with models already trained on in-the-wild Unitree G1 datasets.

**Option 2: Using a different whole-body controller**

If your data was collected using a different whole-body controller, we strongly recommend creating and finetuning with a `NEW_EMBODIMENT` tag. This allows you to define a custom embodiment tag tailored to your specific controller setup. Detailed instructions can be found in the [finetune new embodiment guide](../../getting_started/finetune_new_embodiment.md).
