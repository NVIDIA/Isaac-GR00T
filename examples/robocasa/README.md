# RoboCasa Evaluation Benchmark

> **NOT SUPPORTED YET:** `ROBOCASA_PANDA_OMRON` is **not** included in the N1.7 base checkpoint (`nvidia/GR00T-N1.7-3B`). Running the commands below against the base checkpoint will crash. A finetuned RoboCasa checkpoint is required. This page will be updated once N1.7 RoboCasa support is available.

[RoboCasa](https://robocasa.ai/) is a large-scale simulation framework for training generally capable robots to perform everyday tasks, featuring realistic kitchen environments with over 2,500 3D assets and 100 diverse manipulation tasks. This evaluation benchmark uses RoboCasa with the Panda robot equipped with an Omron gripper to test household manipulation tasks including operating kitchen appliances, pick-and-place operations, and interacting with doors, drawers, and various objects.

---

# Evaluate checkpoint

First, setup the evaluation simulation environment. This only needs to run once for each simulation benchmark. After it's done, we only need to launch server and client.

```bash
sudo apt update
sudo apt install libegl1-mesa-dev libglu1-mesa
bash gr00t/eval/sim/robocasa/setup_RoboCasa.sh
```

#### Downloading RoboCasa Datasets (Optional)

To download RoboCasa demonstration datasets, you **must** use the robocasa venv created by the setup script above (the main project venv does not have `robosuite` installed, which `robocasa` requires at import time):

```bash
# Download human demonstration datasets
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
    external_dependencies/robocasa/robocasa/scripts/download_datasets.py --ds_types human_im

# Download machine-generated datasets
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
    external_dependencies/robocasa/robocasa/scripts/download_datasets.py --ds_types mg
```

> **Note:** Running `python -m robocasa.scripts.download_datasets` from the main project environment will fail because `robocasa` depends on `robosuite`, which is only installed in the robocasa venv.

Then, run client server evaluation under the project root directory in separate terminals:

**Terminal 1 - Server:**
```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path <path-to-finetuned-robocasa-checkpoint> \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --use-sim-policy-wrapper
```

**Terminal 2 - Client:**
```bash
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
    --n-action-steps 8 \
    --n-envs 5
```
