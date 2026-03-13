#!/usr/bin/env bash
# ============================================================================
# RoboCasa Evaluation: Compare PyTorch Eager vs TRT BF16 vs TRT FP8
#
# Runs the full RoboCasa 24-task GR1 tabletop evaluation suite for each
# inference backend and produces a comparison summary.
#
# Prerequisites:
#   - BF16 TRT engines built: bash scripts/deployment/run_trt_pipeline.sh build_bf16
#   - FP8 TRT engines built:  bash scripts/deployment/run_trt_pipeline.sh build_fp8
#   - RoboCasa environment set up (robocasa_uv venv)
#
# Usage:
#   bash scripts/deployment/eval_trt_robocasa.sh all       # ~10 hours
#   bash scripts/deployment/eval_trt_robocasa.sh eager      # ~3-4 hours
#   bash scripts/deployment/eval_trt_robocasa.sh bf16
#   bash scripts/deployment/eval_trt_robocasa.sh fp8
#   bash scripts/deployment/eval_trt_robocasa.sh compare    # compare existing results
#   bash scripts/deployment/eval_trt_robocasa.sh quick      # smoke test (~10 min)
# ============================================================================

set -euo pipefail

# ---- Configuration (override via environment variables) ----
MODEL_PATH="${MODEL_PATH:-nvidia/GR00T-N1.6-3B}"
EMBODIMENT="${EMBODIMENT:-GR1}"
PORT="${PORT:-5556}"
HOST="${HOST:-127.0.0.1}"
BF16_ENGINE_DIR="${BF16_ENGINE_DIR:-./groot_n1d6_engines}"
FP8_ENGINE_DIR="${FP8_ENGINE_DIR:-./groot_n1d6_engines_fp8}"
RESULTS_DIR="${RESULTS_DIR:-./eval_results}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MAIN_REPO="${MAIN_REPO:-/xiaotongc/gr00t-internal}"
SERVER_SCRIPT="${PROJECT_ROOT}/gr00t/eval/run_gr00t_server.py"

# Eval parameters
N_EPISODES="${N_EPISODES:-20}"
N_ENVS="${N_ENVS:-5}"
MAX_STEPS="${MAX_STEPS:-720}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"

# RoboCasa rollout client — uses the robocasa_uv venv from main repo
ROLLOUT_PYTHON="${MAIN_REPO}/gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python"
ROLLOUT_SCRIPT="${MAIN_REPO}/gr00t/eval/rollout_policy.py"

# ---- 24 RoboCasa GR1 Tabletop Tasks ----
TASKS=(
    "gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env"
)

# ---- Helpers ----

mkdir -p "$RESULTS_DIR"

wait_for_server() {
    local max_wait=180
    local waited=0
    echo "Waiting for server on $HOST:$PORT..."
    while ! uv run python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('$HOST', $PORT)); s.close()" 2>/dev/null; do
        sleep 2
        waited=$((waited + 2))
        if [ "$waited" -ge "$max_wait" ]; then
            echo "ERROR: Server did not start within ${max_wait}s"
            return 1
        fi
    done
    echo "Server is ready (waited ${waited}s)"
}

kill_server() {
    echo "Stopping server..."
    if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    # Also kill any lingering server on our port
    lsof -ti :"$PORT" 2>/dev/null | xargs -r kill 2>/dev/null || true
    sleep 2
}

run_robocasa_eval() {
    # Run all 24 RoboCasa tasks against a running server.
    # Results are written to the given file.
    local results_file="$1"
    local total_tasks=${#TASKS[@]}
    local completed=0
    local sum_success=0

    echo "========================================" | tee "$results_file"
    echo "RoboCasa GR1 Tabletop Tasks Evaluation" | tee -a "$results_file"
    echo "Date: $(date)" | tee -a "$results_file"
    echo "N_EPISODES=$N_EPISODES, N_ENVS=$N_ENVS, PORT=$PORT" | tee -a "$results_file"
    echo "========================================" | tee -a "$results_file"

    for TASK in "${TASKS[@]}"; do
        completed=$((completed + 1))
        echo "" | tee -a "$results_file"
        echo "[$completed/$total_tasks] Running: $TASK" | tee -a "$results_file"
        echo "Start time: $(date)" | tee -a "$results_file"

        OUTPUT=$($ROLLOUT_PYTHON "$ROLLOUT_SCRIPT" \
            --n_episodes "$N_EPISODES" \
            --policy_client_host "$HOST" \
            --policy_client_port "$PORT" \
            --max_episode_steps "$MAX_STEPS" \
            --env_name "$TASK" \
            --n_action_steps "$N_ACTION_STEPS" \
            --n_envs "$N_ENVS" 2>&1) || true

        # Extract success rate
        SR=$(echo "$OUTPUT" | grep "success rate:" | tail -1 | awk '{print $NF}')
        RESULTS_LINE=$(echo "$OUTPUT" | grep "results:" | tail -1)

        if [ -n "$SR" ]; then
            echo "  Success rate: $SR" | tee -a "$results_file"
            sum_success=$(awk "BEGIN {print $sum_success + $SR}")
        else
            echo "  ERROR: Could not extract success rate" | tee -a "$results_file"
            echo "  Last 5 lines of output:" | tee -a "$results_file"
            echo "$OUTPUT" | tail -5 | tee -a "$results_file"
        fi

        if [ -n "$RESULTS_LINE" ]; then
            echo "  $RESULTS_LINE" | tee -a "$results_file"
        fi

        echo "End time: $(date)" | tee -a "$results_file"
        echo "----------------------------------------" | tee -a "$results_file"
    done

    local avg
    avg=$(awk "BEGIN {printf \"%.4f\", $sum_success / $total_tasks}")
    echo "" | tee -a "$results_file"
    echo "========================================" | tee -a "$results_file"
    echo "FINAL AVERAGE SUCCESS RATE: $avg" | tee -a "$results_file"
    echo "========================================" | tee -a "$results_file"
}

run_experiment() {
    local name="$1"
    local extra_args="${2:-}"
    local results_file="${RESULTS_DIR}/robocasa_eval_${name}.txt"
    local server_log="${RESULTS_DIR}/server_${name}.log"

    echo ""
    echo "========================================================================"
    echo "  Experiment: $name"
    echo "  Results: $results_file"
    echo "  Server log: $server_log"
    echo "========================================================================"

    # Kill any existing server
    kill_server

    # Start server in background
    echo "Starting server ($name)..."
    # shellcheck disable=SC2086
    uv run python "$SERVER_SCRIPT" \
        --model_path "$MODEL_PATH" \
        --embodiment_tag "$EMBODIMENT" \
        --port "$PORT" \
        --host "$HOST" \
        --use_sim_policy_wrapper \
        $extra_args \
        > "$server_log" 2>&1 &
    SERVER_PID=$!
    echo "  Server PID: $SERVER_PID"

    # Wait for server to be ready
    if ! wait_for_server; then
        echo "ERROR: Server failed to start. Check $server_log"
        kill_server
        return 1
    fi

    # Run evaluation
    echo "Running RoboCasa evaluation (${#TASKS[@]} tasks, $N_EPISODES episodes each)..."
    local start_time
    start_time=$(date +%s)

    run_robocasa_eval "$results_file"

    local end_time
    end_time=$(date +%s)
    local duration=$(( end_time - start_time ))
    echo "  Evaluation completed in $((duration / 60))m $((duration % 60))s"

    # Stop server
    kill_server

    echo "  Results saved to: $results_file"
    echo ""
}

compare_results() {
    echo ""
    echo "========================================================================"
    echo "  COMPARISON SUMMARY"
    echo "========================================================================"
    echo ""

    local summary_file="${RESULTS_DIR}/comparison_summary.txt"
    {
        echo "RoboCasa TRT Evaluation Comparison"
        echo "Date: $(date)"
        echo "Model: $MODEL_PATH"
        echo "N_EPISODES: $N_EPISODES"
        echo ""
        printf "%-15s  %s\n" "Backend" "Avg Success Rate"
        echo "-------------------------------"

        for name in eager bf16 fp8; do
            local results_file="${RESULTS_DIR}/robocasa_eval_${name}.txt"
            if [ -f "$results_file" ]; then
                local avg
                avg=$(grep "FINAL AVERAGE SUCCESS RATE" "$results_file" | awk '{print $NF}' || echo "N/A")
                printf "%-15s  %s\n" "$name" "$avg"
            else
                printf "%-15s  %s\n" "$name" "(not run)"
            fi
        done

        echo ""
        echo "Per-task breakdown:"
        echo ""

        for name in eager bf16 fp8; do
            local results_file="${RESULTS_DIR}/robocasa_eval_${name}.txt"
            if [ -f "$results_file" ]; then
                echo "--- $name ---"
                grep "Success rate:" "$results_file" || true
                echo ""
            fi
        done
    } | tee "$summary_file"

    echo "Summary saved to: $summary_file"
}

# ---- Main ----

COMMAND="${1:-all}"

case "$COMMAND" in
    eager)
        run_experiment "eager" ""
        ;;
    bf16)
        if [ ! -d "$BF16_ENGINE_DIR" ]; then
            echo "ERROR: BF16 engine dir not found: $BF16_ENGINE_DIR"
            echo "Run: bash scripts/deployment/run_trt_pipeline.sh build_bf16"
            exit 1
        fi
        run_experiment "bf16" "--trt_engine_path $BF16_ENGINE_DIR --trt_mode full_pipeline"
        ;;
    fp8)
        if [ ! -d "$FP8_ENGINE_DIR" ]; then
            echo "ERROR: FP8 engine dir not found: $FP8_ENGINE_DIR"
            echo "Run: bash scripts/deployment/run_trt_pipeline.sh export_fp8 && bash scripts/deployment/run_trt_pipeline.sh build_fp8"
            exit 1
        fi
        run_experiment "fp8" "--trt_engine_path $FP8_ENGINE_DIR --trt_mode full_pipeline"
        ;;
    all)
        echo "Running all 3 experiments: eager -> bf16 -> fp8"
        echo "This will take ~10 hours (24 tasks x 20 episodes x 3 backends)."
        echo ""

        run_experiment "eager" ""

        if [ -d "$BF16_ENGINE_DIR" ]; then
            run_experiment "bf16" "--trt_engine_path $BF16_ENGINE_DIR --trt_mode full_pipeline"
        else
            echo "SKIP: BF16 engines not found at $BF16_ENGINE_DIR"
        fi

        if [ -d "$FP8_ENGINE_DIR" ]; then
            run_experiment "fp8" "--trt_engine_path $FP8_ENGINE_DIR --trt_mode full_pipeline"
        else
            echo "SKIP: FP8 engines not found at $FP8_ENGINE_DIR"
        fi

        compare_results
        ;;
    compare)
        compare_results
        ;;
    quick)
        echo "Running quick smoke test (2 episodes, 1 env)"
        N_EPISODES=2
        N_ENVS=1
        run_experiment "eager" ""
        if [ -d "$BF16_ENGINE_DIR" ]; then
            run_experiment "bf16" "--trt_engine_path $BF16_ENGINE_DIR --trt_mode full_pipeline"
        fi
        compare_results
        ;;
    *)
        echo "Usage: $0 {all|eager|bf16|fp8|compare|quick}"
        echo ""
        echo "  all      Run all 3 experiments and compare (~10 hours)"
        echo "  eager    Run PyTorch eager baseline only (~3-4 hours)"
        echo "  bf16     Run TRT BF16 only (~3-4 hours)"
        echo "  fp8      Run TRT FP8 only (~3-4 hours)"
        echo "  compare  Compare existing results (instant)"
        echo "  quick    Quick smoke test with 2 episodes (~10 min)"
        exit 1
        ;;
esac

echo ""
echo "Done!"
