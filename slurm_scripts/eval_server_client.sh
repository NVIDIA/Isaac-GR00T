#!/bin/bash
#SBATCH --job-name=eval_robocasa_panda_omron_PnPCabToCounter_PandaOmron_Env
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00

# MODEL_PATH=runs/libero_10_2026-02-02_230619/checkpoint-20000
# EMBODIMENT_TAG=LIBERO_PANDA
# PORT=5556
# PYTHON_BIN="gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"
# N_EPISODES=10
# MAX_EPISODE_STEPS=720
# ENV_NAME=libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
# N_ACTION_STEPS=8
# N_ENVS=5
# RECORDINGS_DIR=recordings

MODEL_PATH=nvidia/GR00T-N1.6-3B
EMBODIMENT_TAG=ROBOCASA_PANDA_OMRON
HOST=127.0.0.1
PORT=5556
CLIENT_PYTHON_BIN=gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python
MAX_EPISODE_STEPS=720
N_EPISODES=10
ENV_NAME=robocasa_panda_omron/PnPCabToCounter_PandaOmron_Env
N_ENVS=5
N_ACTION_STEPS=8
RECORDINGS_DIR=recordings

cd ~/Isaac-GR00T
mkdir -p logs

echo "------------------------------------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "------------------------------------------------"

echo "[$(date)] Launching server..."

uv run python gr00t/eval/run_gr00t_server.py \
    --model-path $MODEL_PATH \
    --embodiment-tag $EMBODIMENT_TAG \
    --host $HOST \
    --port $PORT \
    --use-sim-policy-wrapper &

SERVER_PID=$!
echo "[$(date)] Server launched with PID: $SERVER_PID"
echo "[$(date)] Launching client..."

$CLIENT_PYTHON_BIN gr00t/eval/rollout_policy.py \
    --max_episode_steps $MAX_EPISODE_STEPS \
    --n_episodes $N_EPISODES \
    --policy_client_host $HOST \
    --policy_client_port $PORT \
    --env_name $ENV_NAME \
    --n_envs $N_ENVS \
    --n_action_steps $N_ACTION_STEPS \
    --recordings_dir $RECORDINGS_DIR

CLIENT_EXIT_CODE=$?
echo "[$(date)] Client finished with exit code: $CLIENT_EXIT_CODE"

# 4. Cleanup
# Kill the server now that the client is done.
echo "Killing Server (PID $SERVER_PID)..."
kill $SERVER_PID

# Exit with the client's status (so Slurm knows if the eval failed)
exit $CLIENT_EXIT_CODE