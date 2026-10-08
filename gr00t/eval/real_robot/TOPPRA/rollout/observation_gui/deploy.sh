#!/usr/bin/env bash
set -Eeuo pipefail

readonly INTEGRATION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPOSITORY_ROOT="$(cd -- "${INTEGRATION_DIR}/../../../../../.." && pwd)"
readonly DEFAULT_GUI_ROOT="$(dirname -- "${REPOSITORY_ROOT}")/ObservationGUILite-v1.0.0"
readonly GUI_ROOT="${1:-${OBSERVATION_GUI_ROOT:-${DEFAULT_GUI_ROOT}}}"
runtime_root="${OBSERVATION_GUI_RUNTIME_ROOT:-logs/observation_gui_rollout}"
gui_runtime="${OBSERVATION_GUI_RUNTIME:-native}"

if [[ ! -x "${GUI_ROOT}/deploy.sh" ]]; then
    echo "ObservationGUILite deploy.sh is unavailable: ${GUI_ROOT}/deploy.sh" >&2
    exit 2
fi

mkdir -p "${runtime_root}/app"
readonly RUNTIME_ROOT="$(cd -- "${runtime_root}" && pwd)"
readonly overlay_dir="${RUNTIME_ROOT}/app"

for entry in \
    .dockerignore \
    FiraCode-SemiBold.ttf \
    agents \
    build \
    controlloop.py \
    deploy.sh \
    docker \
    gui \
    observations \
    pyproject.toml \
    utils \
    uv.lock; do
    cp -a --reflink=auto "${GUI_ROOT}/${entry}" "${overlay_dir}/"
done
cp "${INTEGRATION_DIR}/main.py" "${overlay_dir}/main.py"
cp "${INTEGRATION_DIR}/rollout_passive.py" "${overlay_dir}/agents/rollout_passive.py"
cp "${INTEGRATION_DIR}/rollout_control.py" "${overlay_dir}/gui/rollout_control.py"

echo "Launching ObservationGUILite in read-only external-rollout mode"
echo "ROS namespace: /gr00t_rollout"
echo "Persistent dataset: ${overlay_dir}/datasets/parcel_rollout_view"

if [[ "${gui_runtime}" == "native" ]]; then
    gui_python="${OBSERVATION_GUI_PYTHON:-${REPOSITORY_ROOT}/.venv/bin/python}"
    if [[ ! -x "${gui_python}" ]]; then
        echo "ObservationGUILite Python is unavailable: ${gui_python}" >&2
        exit 2
    fi
    if [[ -z "${DISPLAY:-}" ]]; then
        echo "DISPLAY is unset; launch the GUI from the graphical desktop session." >&2
        exit 2
    fi
    if ! PYTHONPATH="${overlay_dir}/build/lerobot/src:${overlay_dir}" \
        "${gui_python}" -c 'import av; import dearpygui.dearpygui; from agents.lerobuffer import LeroBuffer'; then
        echo "Native GUI dependencies are missing. Install them once with:" >&2
        echo "  uv pip install --python ${gui_python} --no-deps av==15.1.0 dearpygui==2.0.0" >&2
        exit 2
    fi
    echo "GUI runtime: native (${gui_python})"
    cd "${overlay_dir}"
    export PYTHONPATH="${overlay_dir}/build/lerobot/src:${overlay_dir}${PYTHONPATH:+:${PYTHONPATH}}"
    export LEROBOT_VIDEO_BACKEND="${LEROBOT_VIDEO_BACKEND:-pyav}"
    export PYTHONUNBUFFERED=1
    exec "${gui_python}" main.py
fi

if [[ "${gui_runtime}" != "docker" ]]; then
    echo "OBSERVATION_GUI_RUNTIME must be native or docker, got: ${gui_runtime}" >&2
    exit 2
fi

# The upstream curl installer can hang on restricted robot-lab networks. python3-pip is already
# installed by the preceding Dockerfile step and supplies the same pinned uv executable.
sed -i \
    's|RUN curl -LsSf https://astral.sh/uv/install.sh | sh|RUN python3 -m pip install --user uv==0.12.3|' \
    "${overlay_dir}/docker/Dockerfile"
echo "GUI runtime: docker"
export IMAGE_NAME="${IMAGE_NAME:-gr00t-rollout-gui:latest}"
export CONTAINER_NAME="${CONTAINER_NAME:-gr00t-rollout-gui}"
export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-0}"
bash "${overlay_dir}/deploy.sh"
