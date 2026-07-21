#!/usr/bin/env bash
# Launch the GR00T G1-Dex3 gRPC policy server on the workstation dGPU.
#
# This is the GR00T analogue of vla_foundry's
# examples/deployment/launch_g1_policy.sh (which runs
# vla_foundry/inference/robotics/g1/g1_inference_diffusion_policy.py for the
# diffusion policy). The anzu client (lbm_policy_client.yaml) and procman flow are
# identical -- only the server process differs.
#
# Run from the Isaac-GR00T repo root inside the co-installed venv (see README §1).
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-finetuned_gr00t}"
SERVER_URI="${SERVER_URI:-0.0.0.0:50051}"
OPEN_LOOP_STEPS="${OPEN_LOOP_STEPS:-8}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    python gr00t/eval/real_robot/G1Dex3/gr00t_g1_policy_server.py \
    --model-path "$MODEL_PATH" \
    --device cuda \
    --open-loop-steps "$OPEN_LOOP_STEPS" \
    --server-uri "$SERVER_URI"
