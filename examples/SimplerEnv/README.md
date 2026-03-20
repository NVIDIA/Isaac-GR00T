# SimplerEnv

Framework for evaluating real-world robot manipulation policies (RT-1, RT-1-X, Octo) in simulation. Replicates common setups like Google Robot and WidowX+Bridge, with GPU-accelerated simulations (10-15x speedup). Offers visual matching and variant aggregation evaluation methods for robust policy assessment.

For more information, see the [official repository](https://github.com/simpler-env/SimplerEnv).

---

# Fine-tune Simpler Env bridge dataset (WidowX robot)

To reproduce our finetune results, use the following commands to setup dataset and launch finetune experiments via the shared `examples/finetune.sh` launcher. Please remember to set `WANDB_API_KEY` since `--use-wandb` is turned on by default. If you don't have a WANDB account, please remove this argument:

```bash
huggingface-cli download \
    --repo-type dataset IPEC-COMMUNITY/bridge_orig_lerobot \
    --local-dir examples/SimplerEnv/bridge_orig_lerobot/

# Copy the patches and run the finetune script
cp -r examples/SimplerEnv/bridge_modality.json examples/SimplerEnv/bridge_orig_lerobot/meta/modality.json
NUM_GPUS=8 MAX_STEPS=20000 GLOBAL_BATCH_SIZE=1024 SAVE_STEPS=1000 uv run bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path examples/SimplerEnv/bridge_orig_lerobot/ \
    --embodiment-tag OXE_WIDOWX \
    --output-dir /tmp/bridge_finetune \
    --state-dropout-prob 0.8
```

# Fine-tune Simpler Env fractal dataset (Google robot)

```bash
huggingface-cli download \
    --repo-type dataset IPEC-COMMUNITY/fractal20220817_data_lerobot \
    --local-dir examples/SimplerEnv/fractal20220817_data_lerobot/

# Copy the patches and run the finetune script
cp -r examples/SimplerEnv/fractal_modality.json examples/SimplerEnv/fractal20220817_data_lerobot/meta/modality.json
uv run python examples/SimplerEnv/convert_av1_to_h264.py --root examples/SimplerEnv/fractal20220817_data_lerobot --jobs 16  # (Optional) if AV1 doesn't work on your machine
NUM_GPUS=8 MAX_STEPS=20000 GLOBAL_BATCH_SIZE=1024 SAVE_STEPS=1000 uv run bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path examples/SimplerEnv/fractal20220817_data_lerobot/ \
    --embodiment-tag OXE_GOOGLE \
    --output-dir /tmp/fractal_finetune \
    --state-dropout-prob 0.5
```

# Evaluate checkpoint

First, setup the evaluation simulation environment. This only needs to run once for each simulation benchmark. After it's done, we only need to launch server and client.

```bash
sudo apt update
sudo apt install libegl1-mesa-dev libglu1-mesa
bash gr00t/eval/sim/SimplerEnv/setup_SimplerEnv.sh
```

Then, run client server evaluation under the project root directory in separate terminals:

## Fractal (Google Robot) Evaluation

**Terminal 1 - Server:**

You can use either a local finetuned checkpoint path or the remote finetuned checkpoint (provided by us):

**Option 1: Local finetuned checkpoint**
```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path /tmp/fractal_finetune/checkpoint-30000 \
    --embodiment-tag SIMPLER_ENV_GOOGLE \
    --use-sim-policy-wrapper
```

**Option 2: Remote finetuned checkpoint (directly runnable)**
```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.7-SimplerEnv-Fractal \
    --embodiment-tag SIMPLER_ENV_GOOGLE \
    --use-sim-policy-wrapper
```

**Terminal 2 - Client:**
```bash
gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 300 \
    --env-name simpler_env_google/google_robot_pick_coke_can \
    --n-action-steps 1 \
    --n-envs 5
```

## Bridge (WidowX) Evaluation

**Terminal 1 - Server:**
```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-bridge \
    --embodiment-tag OXE_WIDOWX \
    --use-sim-policy-wrapper
```

**Terminal 2 - Client:**
```bash
gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --max_episode_steps=300 \
    --env_name simpler_env_widowx/widowx_spoon_on_towel \
    --n_action_steps 4 \
    --n_envs 5
```

Other supported tasks are: 
```
simpler_env_google/google_robot_pick_object
simpler_env_google/google_robot_move_near
simpler_env_google/google_robot_open_drawer
...
simpler_env_widowx/widowx_spoon_on_towel
simpler_env_widowx/widowx_carrot_on_plate
simpler_env_widowx/widowx_stack_cube
```

you can replace the env_name with the corresponding tasks listed in https://github.com/youliangtan/SimplerEnv
