#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../../.." && pwd)"
config_path="${1:-${script_dir}/local_sim.env.example}"

if [[ ! -f "${config_path}" ]]; then
    echo "local simulation configuration does not exist: ${config_path}" >&2
    exit 2
fi
config_path="$(cd "$(dirname "${config_path}")" && pwd)/$(basename "${config_path}")"

set -a
source "${config_path}"
set +a

case "${ROS_MASTER_URI:-}" in
    http://127.0.0.1:* | http://localhost:*) ;;
    *)
        echo "refusing non-local ROS_MASTER_URI=${ROS_MASTER_URI:-<unset>}" >&2
        exit 2
        ;;
esac
if [[ "${ROS_IP:-}" != "127.0.0.1" ]]; then
    echo "local simulation requires ROS_IP=127.0.0.1" >&2
    exit 2
fi
if [[ "${DRY_RUN:-1}" != "0" ]]; then
    echo "local simulation requires DRY_RUN=0 so DualPose is published" >&2
    exit 2
fi

master_port="${ROS_MASTER_URI##*:}"
master_port="${master_port%/}"
if [[ ! "${master_port}" =~ ^[0-9]+$ ]]; then
    echo "could not parse ROS master port from ${ROS_MASTER_URI}" >&2
    exit 2
fi

if [[ -n "${ROS_SETUP:-}" ]]; then
    source "${ROS_SETUP}"
fi
if [[ -n "${ROBOT_ROS_SETUP:-}" ]]; then
    source "${ROBOT_ROS_SETUP}"
fi
cd "${repo_root}"
ros_message_python_path="$("${PYTHON_BIN:-.venv/bin/python}" -c \
    'from pathlib import Path; import upperlimb; print(Path(upperlimb.__file__).parent.parent)')"

roscore_pid=""
simulator_pid=""
cleanup() {
    if [[ -n "${simulator_pid}" ]]; then
        kill "${simulator_pid}" 2>/dev/null || true
        wait "${simulator_pid}" 2>/dev/null || true
    fi
    if [[ -n "${roscore_pid}" ]]; then
        kill "${roscore_pid}" 2>/dev/null || true
        wait "${roscore_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if ! timeout 1 rosparam list >/dev/null 2>&1; then
    roscore -p "${master_port}" >"/tmp/gr00t_local_roscore_${master_port}.log" 2>&1 &
    roscore_pid=$!
    for _ in {1..50}; do
        if timeout 1 rosparam list >/dev/null 2>&1; then
            break
        fi
        sleep 0.1
    done
fi
if ! timeout 1 rosparam list >/dev/null 2>&1; then
    echo "local ROS master did not start; see /tmp/gr00t_local_roscore_${master_port}.log" >&2
    exit 1
fi

"${PYTHON_BIN:-.venv/bin/python}" \
    -m gr00t.eval.real_robot.TOPPRA.rollout.local_sim_hardware \
    --print-every 25 &
simulator_pid=$!

for _ in {1..50}; do
    if rostopic info /zj_humanoid/upperlimb/servol/dual_arm 2>/dev/null \
        | grep -q '/gr00t_local_sim_hardware'; then
        break
    fi
    sleep 0.1
done
if ! kill -0 "${simulator_pid}" 2>/dev/null; then
    echo "local hardware simulator exited during startup" >&2
    exit 1
fi

echo
echo "Local ROS simulation is ready on ${ROS_MASTER_URI}."
echo "Type 'start' at the rollout prompt to begin inference and action publication."
echo "In another terminal, continuously inspect the output with:"
echo "  export ROS_MASTER_URI=${ROS_MASTER_URI} ROS_IP=127.0.0.1"
echo "  source /opt/ros/noetic/setup.bash"
echo "  export PYTHONPATH=${ros_message_python_path}:\${PYTHONPATH:-}"
echo "  rostopic echo /zj_humanoid/upperlimb/servol/dual_arm"
echo

bash "${script_dir}/launch.sh" "${config_path}"
