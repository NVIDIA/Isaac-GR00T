<!-- TODO: RoboCasa Panda (ROBOCASA_PANDA_OMRON) is not verified with N1.7 checkpoint.
     Remove or update this example once N1.7 support is confirmed. -->

# RoboCasa Evaluation Benchmark

[RoboCasa](https://robocasa.ai/) is a large-scale simulation framework for training generally capable robots to perform everyday tasks, featuring realistic kitchen environments with over 2,500 3D assets and 100 diverse manipulation tasks. This evaluation benchmark uses RoboCasa with the Panda robot equipped with an Omron gripper to test household manipulation tasks including operating kitchen appliances, pick-and-place operations, and interacting with doors, drawers, and various objects.

---

# RoboCasa evaluation benchmark result
Checkpoint: [nvidia/GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B)

| Task | Success rate |
| ---- | ------------ |
| `robocasa_panda_omron/CoffeeSetupMug_PandaOmron_Env` | 31.0% |
| `robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env` | 63.5% |
| `robocasa_panda_omron/CoffeePressButton_PandaOmron_Env` | 98.5% |
| `robocasa_panda_omron/OpenSingleDoor_PandaOmron_Env` | 81.5% |
| `robocasa_panda_omron/OpenDoubleDoor_PandaOmron_Env` | 39.0% |
| `robocasa_panda_omron/CloseSingleDoor_PandaOmron_Env` | 96.0% |
| `robocasa_panda_omron/CloseDoubleDoor_PandaOmron_Env` | 88.5% |
| `robocasa_panda_omron/OpenDrawer_PandaOmron_Env` | 81.1% |
| `robocasa_panda_omron/CloseDrawer_PandaOmron_Env` | 100.0% |
| `robocasa_panda_omron/TurnOnMicrowave_PandaOmron_Env` | 91.5% |
| `robocasa_panda_omron/TurnOffMicrowave_PandaOmron_Env` | 96.0% |
| `robocasa_panda_omron/PnPCounterToCab_PandaOmron_Env` | 47.5% |
| `robocasa_panda_omron/PnPCabToCounter_PandaOmron_Env` | 41.0% |
| `robocasa_panda_omron/PnPCounterToSink_PandaOmron_Env` | 46.0% |
| `robocasa_panda_omron/PnPSinkToCounter_PandaOmron_Env` | 50.0% |
| `robocasa_panda_omron/PnPCounterToMicrowave_PandaOmron_Env` | 19.0% |
| `robocasa_panda_omron/PnPMicrowaveToCounter_PandaOmron_Env` | 24.5% |
| `robocasa_panda_omron/PnPCounterToStove_PandaOmron_Env` | 63.2% |
| `robocasa_panda_omron/PnPStoveToCounter_PandaOmron_Env` | 54.5% |
| `robocasa_panda_omron/TurnOnSinkFaucet_PandaOmron_Env` | 89.0% |
| `robocasa_panda_omron/TurnOffSinkFaucet_PandaOmron_Env` | 93.5% |
| `robocasa_panda_omron/TurnSinkSpout_PandaOmron_Env` | 87.0% |
| `robocasa_panda_omron/TurnOnStove_PandaOmron_Env` | 76.5% |
| `robocasa_panda_omron/TurnOffStove_PandaOmron_Env` | 31.0% |
| **Average** | 66.22% |

> **Note:** Benchmark results above are from N1.6. N1.7 results pending.

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
    --model-path nvidia/GR00T-N1.7-3B \
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
