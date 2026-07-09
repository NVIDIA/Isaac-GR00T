#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Provision an Ubuntu bare-metal GPU host for GR00T training.
#
# Defaults are intentionally conservative for GB300 hosts:
#   - install Ubuntu's open NVIDIA driver package for branch 595
#   - persist NVreg_EnableUserNUMAManagement=0 to avoid the observed GB300
#     nvidia-smi teardown hang around nv_remove_numa_memory/raw_q_flush
#   - install Docker CE from Docker's official apt repository
#   - install and configure NVIDIA Container Toolkit for Docker
#
# Usage:
#   sudo ci/scripts/setup_bare_metal_gpu_host.sh
#
# Common overrides:
#   NVIDIA_DRIVER_BRANCH=580 sudo ci/scripts/setup_bare_metal_gpu_host.sh
#   NVIDIA_DRIVER_FLAVOR=server-open sudo ci/scripts/setup_bare_metal_gpu_host.sh
#   NVIDIA_DRIVER_PACKAGE=nvidia-driver-595-server-open sudo ci/scripts/setup_bare_metal_gpu_host.sh
#   sudo ci/scripts/setup_bare_metal_gpu_host.sh --validate-groot-image --nvcr-token-file /root/.nvcr-token
#   printf '%s' "${NVCR_TOKEN}" | sudo ci/scripts/setup_bare_metal_gpu_host.sh --validate-groot-image --nvcr-token-stdin
#   ALLOW_UNSAFE_NVIDIA_SMI=1 sudo -E ci/scripts/setup_bare_metal_gpu_host.sh --validate-groot-image

set -euo pipefail

NVIDIA_DRIVER_BRANCH="${NVIDIA_DRIVER_BRANCH:-595}"
NVIDIA_DRIVER_FLAVOR="${NVIDIA_DRIVER_FLAVOR:-open}"
NVIDIA_NUMA_WORKAROUND="${NVIDIA_NUMA_WORKAROUND:-1}"
NVIDIA_TEST_IMAGE="${NVIDIA_TEST_IMAGE:-}"
GROOT_CI_TRAIN_IMAGE="${GROOT_CI_TRAIN_IMAGE:-nvcr.io/nvidian/groot-ci-train:c7a32a6d}"
VALIDATE_GROOT_IMAGE="${VALIDATE_GROOT_IMAGE:-0}"
NVCR_TOKEN="${NVCR_TOKEN:-}"
NVCR_TOKEN_SOURCE="${NVCR_TOKEN_SOURCE:-}"
ALLOW_UNSAFE_NVIDIA_SMI="${ALLOW_UNSAFE_NVIDIA_SMI:-0}"

readonly DOCKER_KEYRING="/etc/apt/keyrings/docker.asc"
readonly DOCKER_SOURCE_LIST="/etc/apt/sources.list.d/docker.list"
readonly NVIDIA_CTK_KEYRING="/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
readonly NVIDIA_CTK_SOURCE_LIST="/etc/apt/sources.list.d/nvidia-container-toolkit.list"
readonly NVIDIA_MODPROBE_CONFIG="/etc/modprobe.d/groot-gb300-nvidia.conf"

TMP_DIR=""

log() {
  printf '[setup_bare_metal_gpu_host] %s\n' "$*"
}

die() {
  printf '[setup_bare_metal_gpu_host] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage:
  sudo ci/scripts/setup_bare_metal_gpu_host.sh [options]

Options:
  --validate-groot-image       Run Docker GPU validation with ${GROOT_CI_TRAIN_IMAGE}.
  --nvidia-test-image IMAGE    Run Docker GPU validation with a custom image.
  --nvcr-token TOKEN           Login to nvcr.io with an NGC/NVCR API token.
  --nvcr-token-file PATH       Read the nvcr.io token from a file.
  --nvcr-token-stdin           Read the nvcr.io token from stdin.
  -h, --help                   Show this help.

Environment overrides:
  NVIDIA_DRIVER_BRANCH         Driver branch to install, default: ${NVIDIA_DRIVER_BRANCH}
  NVIDIA_DRIVER_FLAVOR         open, server-open, proprietary, or server; default: ${NVIDIA_DRIVER_FLAVOR}
  NVIDIA_DRIVER_PACKAGE        Exact NVIDIA driver package name, overrides branch/flavor.
  GROOT_CI_TRAIN_IMAGE         GR00T validation image, default: ${GROOT_CI_TRAIN_IMAGE}
  NVIDIA_TEST_IMAGE            Custom Docker GPU validation image.
  NVCR_TOKEN                   Token used for docker login nvcr.io.
EOF
}

require_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "${value}" ]]; then
    die "${option} requires a value"
  fi
}

read_token_file() {
  local token_file="$1"
  if [[ ! -r "${token_file}" ]]; then
    die "cannot read NVCR token file: ${token_file}"
  fi
  tr -d '\r\n' < "${token_file}"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --validate-groot-image)
        VALIDATE_GROOT_IMAGE="1"
        shift
        ;;
      --nvidia-test-image)
        require_value "$1" "${2:-}"
        NVIDIA_TEST_IMAGE="$2"
        shift 2
        ;;
      --nvcr-token)
        require_value "$1" "${2:-}"
        NVCR_TOKEN="$2"
        NVCR_TOKEN_SOURCE="argv"
        shift 2
        ;;
      --nvcr-token-file)
        require_value "$1" "${2:-}"
        NVCR_TOKEN="$(read_token_file "$2")"
        NVCR_TOKEN_SOURCE="file"
        shift 2
        ;;
      --nvcr-token-stdin)
        NVCR_TOKEN="$(tr -d '\r\n')"
        NVCR_TOKEN_SOURCE="stdin"
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "unknown option: $1"
        ;;
    esac
  done

  if [[ -n "${NVCR_TOKEN}" && -z "${NVCR_TOKEN_SOURCE}" ]]; then
    NVCR_TOKEN_SOURCE="env"
  fi

  if [[ "${VALIDATE_GROOT_IMAGE}" == "1" && -z "${NVIDIA_TEST_IMAGE}" ]]; then
    NVIDIA_TEST_IMAGE="${GROOT_CI_TRAIN_IMAGE}"
  fi
}

cleanup() {
  if [[ -n "${TMP_DIR}" && -d "${TMP_DIR}" ]]; then
    rm -rf "${TMP_DIR}"
  fi
}
trap cleanup EXIT

apt_get() {
  DEBIAN_FRONTEND=noninteractive apt-get "$@"
}

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    die "run this script with sudo or as root"
  fi
}

load_os_release() {
  if [[ ! -r /etc/os-release ]]; then
    die "cannot read /etc/os-release"
  fi

  # shellcheck disable=SC1091
  . /etc/os-release

  if [[ "${ID:-}" != "ubuntu" ]]; then
    die "expected Ubuntu, found ID=${ID:-unknown}"
  fi

  if [[ -z "${VERSION_CODENAME:-}" ]]; then
    die "Ubuntu VERSION_CODENAME is missing from /etc/os-release"
  fi
}

check_architecture() {
  local arch
  arch="$(dpkg --print-architecture)"
  case "${arch}" in
    amd64|arm64) ;;
    *) die "unsupported architecture ${arch}; expected amd64 or arm64" ;;
  esac
}

default_driver_package() {
  if [[ -n "${NVIDIA_DRIVER_PACKAGE:-}" ]]; then
    printf '%s\n' "${NVIDIA_DRIVER_PACKAGE}"
    return
  fi

  case "${NVIDIA_DRIVER_FLAVOR}" in
    open)
      printf 'nvidia-driver-%s-open\n' "${NVIDIA_DRIVER_BRANCH}"
      ;;
    server-open)
      printf 'nvidia-driver-%s-server-open\n' "${NVIDIA_DRIVER_BRANCH}"
      ;;
    proprietary)
      printf 'nvidia-driver-%s\n' "${NVIDIA_DRIVER_BRANCH}"
      ;;
    server)
      printf 'nvidia-driver-%s-server\n' "${NVIDIA_DRIVER_BRANCH}"
      ;;
    *)
      die "unknown NVIDIA_DRIVER_FLAVOR=${NVIDIA_DRIVER_FLAVOR}; use open, server-open, proprietary, or server"
      ;;
  esac
}

install_prerequisites() {
  log "Refreshing apt metadata and installing prerequisites"
  apt_get update
  apt_get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    shellcheck \
    sshpass \
    software-properties-common \
    ubuntu-drivers-common

  log "Ensuring Ubuntu multiverse is enabled for NVIDIA driver packages"
  add-apt-repository -y multiverse >/dev/null
}

install_docker_repository() {
  log "Configuring Docker CE apt repository"

  local arch
  arch="$(dpkg --print-architecture)"

  install -d -m 0755 /etc/apt/keyrings

  TMP_DIR="$(mktemp -d)"
  curl -fsSL "https://download.docker.com/linux/ubuntu/gpg" -o "${TMP_DIR}/docker.asc"
  install -m 0644 "${TMP_DIR}/docker.asc" "${DOCKER_KEYRING}"

  printf 'deb [arch=%s signed-by=%s] https://download.docker.com/linux/ubuntu %s stable\n' \
    "${arch}" "${DOCKER_KEYRING}" "${VERSION_CODENAME}" > "${DOCKER_SOURCE_LIST}"
}

install_nvidia_container_toolkit_repository() {
  log "Configuring NVIDIA Container Toolkit apt repository"

  install -d -m 0755 "$(dirname "${NVIDIA_CTK_KEYRING}")"

  TMP_DIR="${TMP_DIR:-$(mktemp -d)}"
  curl -fsSL "https://nvidia.github.io/libnvidia-container/gpgkey" \
    | gpg --dearmor -o "${TMP_DIR}/nvidia-container-toolkit-keyring.gpg"
  install -m 0644 "${TMP_DIR}/nvidia-container-toolkit-keyring.gpg" "${NVIDIA_CTK_KEYRING}"

  curl -fsSL "https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list" \
    | sed "s#deb https://#deb [signed-by=${NVIDIA_CTK_KEYRING}] https://#g" \
    > "${NVIDIA_CTK_SOURCE_LIST}"
}

install_nvidia_driver() {
  local driver_package
  driver_package="$(default_driver_package)"

  log "Installing NVIDIA driver package: ${driver_package}"
  apt_get update

  if ! apt-cache show "${driver_package}" >/dev/null 2>&1; then
    log "Available NVIDIA driver packages:"
    apt-cache search '^nvidia-driver-[0-9]+.*' | sed 's/^/  /' || true
    die "package ${driver_package} is not available from configured apt repositories"
  fi

  apt_get install -y "${driver_package}"
}

remove_conflicting_docker_packages() {
  local conflicting_packages=(
    docker.io
    docker-doc
    docker-compose
    docker-compose-v2
    podman-docker
    containerd
    runc
  )
  local installed_packages=()
  local package

  for package in "${conflicting_packages[@]}"; do
    if dpkg-query -W -f='${Status}' "${package}" 2>/dev/null | grep -q '^install ok installed$'; then
      installed_packages+=("${package}")
    fi
  done

  if [[ "${#installed_packages[@]}" -gt 0 ]]; then
    log "Removing Docker package conflicts: ${installed_packages[*]}"
    apt_get remove -y "${installed_packages[@]}"
  fi
}

persist_nvidia_module_options() {
  if [[ "${NVIDIA_NUMA_WORKAROUND}" != "1" ]]; then
    log "Skipping NVIDIA NUMA workaround because NVIDIA_NUMA_WORKAROUND=${NVIDIA_NUMA_WORKAROUND}"
    return
  fi

  log "Persisting NVIDIA GB300 NUMA workaround in ${NVIDIA_MODPROBE_CONFIG}"
  cat > "${NVIDIA_MODPROBE_CONFIG}" <<EOF
# Managed by ci/scripts/setup_bare_metal_gpu_host.sh.
# GB300 workaround: avoid nvidia-smi hangs in nv_remove_numa_memory/raw_q_flush.
options nvidia NVreg_EnableUserNUMAManagement=0
EOF

  if command -v update-initramfs >/dev/null 2>&1; then
    log "Updating initramfs so NVIDIA module options apply on next boot"
    update-initramfs -u
  fi
}

install_docker_and_toolkit() {
  log "Installing Docker CE and NVIDIA Container Toolkit"
  apt_get update
  remove_conflicting_docker_packages
  apt_get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin \
    nvidia-container-toolkit

  log "Configuring NVIDIA runtime for Docker"
  nvidia-ctk runtime configure --runtime=docker

  systemctl enable --now docker
  systemctl restart docker

  if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
    log "Adding ${SUDO_USER} to the docker group"
    usermod -aG docker "${SUDO_USER}"
  fi
}

live_nvidia_param() {
  local key="$1"
  if [[ -r /proc/driver/nvidia/params ]]; then
    awk -F': ' -v key="${key}" '$1 == key { print $2 }' /proc/driver/nvidia/params
  fi
}

docker_login_nvcr_if_requested() {
  if [[ -z "${NVCR_TOKEN}" ]]; then
    if [[ "${NVIDIA_TEST_IMAGE}" == nvcr.io/* ]]; then
      log "No NVCR token was provided; assuming Docker already has valid nvcr.io credentials"
    fi
    return
  fi

  if [[ "${NVCR_TOKEN_SOURCE}" == "argv" ]]; then
    log "Warning: --nvcr-token can be visible in shell history or process listings; prefer --nvcr-token-stdin or --nvcr-token-file"
  fi

  log "Logging in to nvcr.io for validation image access"
  printf '%s' "${NVCR_TOKEN}" | docker login nvcr.io --username '$oauthtoken' --password-stdin >/dev/null
}

ensure_validation_image_available() {
  if [[ -z "${NVIDIA_TEST_IMAGE}" ]]; then
    return
  fi

  docker_login_nvcr_if_requested

  if docker image inspect "${NVIDIA_TEST_IMAGE}" >/dev/null 2>&1; then
    log "Using local validation image ${NVIDIA_TEST_IMAGE}"
    return
  fi

  log "Pulling validation image ${NVIDIA_TEST_IMAGE}"
  timeout --kill-after=10s 900s docker pull "${NVIDIA_TEST_IMAGE}"
}

validate_installation() {
  log "Validating Docker service"
  docker info >/dev/null

  if [[ ! -r /proc/driver/nvidia/params ]]; then
    log "NVIDIA driver is not loaded yet; reboot before running GPU validation"
    return
  fi

  local live_numa
  live_numa="$(live_nvidia_param EnableUserNUMAManagement || true)"
  if [[ "${live_numa}" != "0" && "${ALLOW_UNSAFE_NVIDIA_SMI}" != "1" ]]; then
    log "Skipping nvidia-smi validation because the loaded driver has EnableUserNUMAManagement=${live_numa:-unknown}"
    log "Reboot, or unload/reload NVIDIA modules, then run: timeout --kill-after=5s 20s nvidia-smi -L"
    return
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    log "Running bounded nvidia-smi gate"
    timeout --kill-after=5s 20s nvidia-smi -L
  fi

  if [[ -n "${NVIDIA_TEST_IMAGE}" ]]; then
    ensure_validation_image_available
    log "Running bounded Docker GPU gate with ${NVIDIA_TEST_IMAGE}"
    timeout --kill-after=10s 60s docker run --rm --gpus all "${NVIDIA_TEST_IMAGE}" nvidia-smi -L
  else
    log "Skipping Docker GPU image test; set NVIDIA_TEST_IMAGE to enable it"
  fi
}

main() {
  parse_args "$@"
  require_root
  load_os_release
  check_architecture

  install_prerequisites
  install_docker_repository
  install_nvidia_container_toolkit_repository
  install_nvidia_driver
  persist_nvidia_module_options
  install_docker_and_toolkit
  validate_installation

  log "Provisioning complete"
  log "Reboot is recommended before using the GPU so the installed driver and module options are active"
  if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
    log "${SUDO_USER} must log out and back in before docker group membership is active"
  fi
}

main "$@"
