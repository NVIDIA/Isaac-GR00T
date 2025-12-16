# SimplerEnv

Framework for evaluating real-world robot manipulation policies (RT-1, RT-1-X, Octo) in simulation. Replicates common setups like Google Robot and WidowX+Bridge, with GPU-accelerated simulations (10-15x speedup). Offers visual matching and variant aggregation evaluation methods for robust policy assessment.

For more information, see the [official repository](https://github.com/simpler-env/SimplerEnv).

---

# SimplerEnv Bridge (WidowX robot) evaluation benchmark result
Provided checkpoint: [nvidia/GR00T-N1.6-bridge](https://huggingface.co/nvidia/GR00T-N1.6-bridge)

| Task                              | Success rate n1.6 (200) |
| --------------------------------- | ------------------ |
| widowx\_spoon\_on\_towel          | 129/200 (64.5%) |
| widowx\_carrot\_on\_plate         | 131/200 (65.5%) |
| widowx\_put\_eggplant\_in\_basket | 186/200 (93%) |
| widowx\_stack\_cube               | 11/200 (5.5%) |
| widowx\_put\_eggplant\_in\_sink** | 80/200 (40%) |
| widowx\_close\_drawer**           | 141/200 (70.5%) |
| widowx\_open\_drawer**            | 191/200 (95.5%) |
| **Average**                       | **62.07%** |

# SimplerEnv Fractal (Google robot) evaluation benchmark result
Provided checkpoint: [nvidia/GR00T-N1.6-fractal](https://huggingface.co/nvidia/GR00T-N1.6-fractal)
| Task                                     | Success Rate (200) |
| ---------------------------------------- | ------------------ |
| google\_robot\_pick\_coke\_can           | 195/200 (97.5%)    |
| google\_robot\_pick\_object              | 174/200 (87%)      |
| google\_robot\_move\_near                | 151/200 (75.5%)    |
| google\_robot\_open\_drawer              | 88/200 (44%)       |
| google\_robot\_close\_drawer             | 175/200 (87.5%)    |
| google\_robot\_place\_in\_closed\_drawer | 29/200 (14.5%)     |
| **Average**                              | **67.66%**            |


# Fine-tune Simpler Env bridge dataset (WidowX robot)

To reproduce our finetune results, use the following commands to setup dataset and launch finetune experiments. Please remember to set `WANDB_API_KEY` since `--use-wandb` is turned on by default. If you don't have a WANDB account, please remove this argument:

```bash
huggingface-cli download \
    --repo-type dataset IPEC-COMMUNITY/bridge_orig_lerobot \
    --local-dir examples/SimplerEnv/bridge_orig_lerobot/

# Copy the patches and run the finetune script
cp -r examples/SimplerEnv/bridge_modality.json examples/SimplerEnv/bridge_orig_lerobot/meta/modality.json
uv run bash examples/SimplerEnv/finetune_bridge.sh
```

# Fine-tune Simpler Env fractal dataset (Google robot)

```bash
cd examples/SimplerEnv
huggingface-cli download \
    --repo-type dataset IPEC-COMMUNITY/fractal20220817_data_lerobot \
    --local-dir examples/SimplerEnv/fractal20220817_data_lerobot/

# Copy the patches and run the finetune script
cp -r examples/SimplerEnv/fractal_modality.json examples/SimplerEnv/fractal20220817_data_lerobot/meta/modality.json
uv run python convert_av1_to_h264.py --root fractal20220817_data_lerobot --jobs 16  # (Optional) if AV1 doesn't work on your machine
uv run bash examples/SimplerEnv/finetune_fractal.sh
```

# Evaluate checkpoint

First, setup the evaluation simulation environment. This only needs to run once for each simulation benchmark. After it's done, we only need to launch server and client.

```bash
sudo apt update
sudo apt install libegl1-mesa-dev libglu1-mesa
bash gr00t/eval/sim/SimplerEnv/setup_SimplerEnv.sh
```

Then, run client server evaluation under the project root directory in separate terminals:

**Terminal 1 - Server:**

You can use either a local finetuned checkpoint path or the remote finetuned checkpoint (provided by us):

**Option 1: Local finetuned checkpoint**
```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path /tmp/fractal_finetune/checkpoint-30000 \
    --embodiment-tag OXE_GOOGLE \
    --use-sim-policy-wrapper
```

**Option 2: Remote finetuned checkpoint (directly runnable)**
```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-fractal \
    --embodiment-tag OXE_GOOGLE \
    --use-sim-policy-wrapper
```

**Terminal 2 - Client:**
```bash
gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --max_episode_steps=300 \
    --env_name simpler_env_google/google_robot_pick_coke_can \
    --n_action_steps 1 \
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
