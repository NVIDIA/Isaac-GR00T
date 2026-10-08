#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: bash gr00t/eval/real_robot/TOPPRA/rollout/launch_gui.sh CONFIG.local.env" >&2
    exit 2
fi

config_path="$1"
if [[ ! -f "${config_path}" ]]; then
    echo "configuration file does not exist: ${config_path}" >&2
    exit 2
fi
config_path="$(realpath "${config_path}")"

set -a
source "${config_path}"
set +a

control_interface="${CONTROL_INTERFACE:-gui}"
if [[ "${control_interface}" != "gui" && "${control_interface}" != "both" ]]; then
    echo "ObservationGUILite requires CONTROL_INTERFACE=gui or both, got: ${control_interface}" >&2
    exit 2
fi
if [[ "${ROS_NAMESPACE:-/gr00t_rollout}" != "/gr00t_rollout" ]]; then
    echo "The first ObservationGUILite overlay requires ROS_NAMESPACE=/gr00t_rollout" >&2
    exit 2
fi
: "${ROS_MASTER_URI:?set ROS_MASTER_URI in the rollout configuration}"
: "${ROS_IP:?set ROS_IP in the rollout configuration}"

readonly ROLLOUT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPOSITORY_ROOT="$(cd -- "${ROLLOUT_DIR}/../../../../.." && pwd)"
export OBSERVATION_GUI_RUNTIME_ROOT="${OBSERVATION_GUI_RUNTIME_ROOT:-${REPOSITORY_ROOT}/logs/observation_gui_rollout}"

gui_root="${OBSERVATION_GUI_ROOT:-$(dirname -- "${REPOSITORY_ROOT}")/ObservationGUILite-v1.0.0}"
echo "Launching the external ObservationGUILite window."
echo "The rollout client must already be running with CONTROL_INTERFACE=${control_interface}."
exec bash "${ROLLOUT_DIR}/observation_gui/deploy.sh" "${gui_root}"
