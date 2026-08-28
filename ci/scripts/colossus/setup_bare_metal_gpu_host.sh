#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
#   sudo ci/scripts/colossus/setup_bare_metal_gpu_host.sh
#
# Common overrides:
#   NVIDIA_DRIVER_BRANCH=580 sudo ci/scripts/colossus/setup_bare_metal_gpu_host.sh
#   NVIDIA_DRIVER_FLAVOR=server-open sudo ci/scripts/colossus/setup_bare_metal_gpu_host.sh
#   NVIDIA_DRIVER_PACKAGE=nvidia-driver-595-server-open sudo ci/scripts/colossus/setup_bare_metal_gpu_host.sh
#   sudo ci/scripts/colossus/setup_bare_metal_gpu_host.sh --validate-groot-image --nvcr-token-file /root/.nvcr-token
#   printf '%s' "${NVCR_TOKEN}" | sudo ci/scripts/colossus/setup_bare_metal_gpu_host.sh --validate-groot-image --nvcr-token-stdin
#   ALLOW_UNSAFE_NVIDIA_SMI=1 sudo -E ci/scripts/colossus/setup_bare_metal_gpu_host.sh --validate-groot-image

set -euo pipefail

NVIDIA_DRIVER_BRANCH="${NVIDIA_DRIVER_BRANCH:-595}"
NVIDIA_DRIVER_FLAVOR="${NVIDIA_DRIVER_FLAVOR:-open}"
NVIDIA_NUMA_WORKAROUND="${NVIDIA_NUMA_WORKAROUND:-1}"
NVIDIA_TEST_IMAGE="${NVIDIA_TEST_IMAGE:-}"
NVIDIA_TEST_IMAGE_PULL_TIMEOUT="${NVIDIA_TEST_IMAGE_PULL_TIMEOUT:-3600}"
NVIDIA_DOCKER_GPU_GATE_TIMEOUT="${NVIDIA_DOCKER_GPU_GATE_TIMEOUT:-300}"
GROOT_CI_TRAIN_IMAGE="${GROOT_CI_TRAIN_IMAGE:-nvcr.io/nvidian/groot-ci-train:c7a32a6d}"
VALIDATE_GROOT_IMAGE="${VALIDATE_GROOT_IMAGE:-0}"
NVCR_TOKEN="${NVCR_TOKEN:-}"
NVCR_TOKEN_SOURCE="${NVCR_TOKEN_SOURCE:-}"
ALLOW_UNSAFE_NVIDIA_SMI="${ALLOW_UNSAFE_NVIDIA_SMI:-0}"
VALIDATE_RDMA="${VALIDATE_RDMA:-0}"
RDMA_INTERFACE="${RDMA_INTERFACE:-}"
RDMA_HCA="${RDMA_HCA:-}"
RDMA_GID_INDEX="${RDMA_GID_INDEX:-}"

readonly DOCKER_KEYRING="/etc/apt/keyrings/docker.asc"
readonly DOCKER_SOURCE_LIST="/etc/apt/sources.list.d/docker.list"
readonly NVIDIA_CTK_KEYRING="/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
readonly NVIDIA_CTK_SOURCE_LIST="/etc/apt/sources.list.d/nvidia-container-toolkit.list"
readonly NVIDIA_MODPROBE_CONFIG="/etc/modprobe.d/groot-gb300-nvidia.conf"

TMP_DIR=""
declare -a RDMA_UVERBS_DEVICES=()

log() {
  printf '[setup_bare_metal_gpu_host] %s\n' "$*"
}

die() {
  printf '[setup_bare_metal_gpu_host] ERROR: %s\n' "$*" >&2
  exit 1
}

curl_with_retry() {
  curl --retry 5 --retry-all-errors --retry-delay 2 --connect-timeout 30 "$@"
}

usage() {
  cat <<EOF
Usage:
  sudo ci/scripts/colossus/setup_bare_metal_gpu_host.sh [options]

Options:
  --validate-groot-image       Run Docker GPU validation with ${GROOT_CI_TRAIN_IMAGE}.
  --nvidia-test-image IMAGE    Run Docker GPU validation with a custom image.
  --nvcr-token TOKEN           Login to nvcr.io with an NGC/NVCR API token.
  --nvcr-token-file PATH       Read the nvcr.io token from a file.
  --nvcr-token-stdin           Read the nvcr.io token from stdin.
  --validate-rdma              Fail unless the selected RDMA interface/HCA/GID works.
  --rdma-interface IFACE       Data-plane netdev to validate.
  --rdma-hca HCA[:PORT]        Comma-separated HCA/port list, for example mlx5_3:1.
  --rdma-gid-index INDEX       RoCE GID index to validate.
  -h, --help                   Show this help.

Environment overrides:
  NVIDIA_DRIVER_BRANCH         Driver branch to install, default: ${NVIDIA_DRIVER_BRANCH}
  NVIDIA_DRIVER_FLAVOR         open, server-open, proprietary, or server; default: ${NVIDIA_DRIVER_FLAVOR}
  NVIDIA_DRIVER_PACKAGE        Exact NVIDIA driver package name, overrides branch/flavor.
  GROOT_CI_TRAIN_IMAGE         GR00T validation image, default: ${GROOT_CI_TRAIN_IMAGE}
  NVIDIA_TEST_IMAGE            Custom Docker GPU validation image.
  NVIDIA_TEST_IMAGE_PULL_TIMEOUT
                               Docker pull timeout in seconds, default: ${NVIDIA_TEST_IMAGE_PULL_TIMEOUT}
  NVIDIA_DOCKER_GPU_GATE_TIMEOUT
                               Docker GPU gate timeout in seconds, default: ${NVIDIA_DOCKER_GPU_GATE_TIMEOUT}
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
      --validate-rdma)
        VALIDATE_RDMA="1"
        shift
        ;;
      --rdma-interface)
        require_value "$1" "${2:-}"
        RDMA_INTERFACE="$2"
        shift 2
        ;;
      --rdma-hca)
        require_value "$1" "${2:-}"
        RDMA_HCA="$2"
        shift 2
        ;;
      --rdma-gid-index)
        require_value "$1" "${2:-}"
        RDMA_GID_INDEX="$2"
        shift 2
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

validate_rdma_args() {
  case "${VALIDATE_RDMA}" in
    0|1) ;;
    *) die "VALIDATE_RDMA must be 0 or 1" ;;
  esac
  if [[ "${VALIDATE_RDMA}" != "1" ]]; then
    return
  fi
  [[ -n "${RDMA_INTERFACE}" ]] || die "--rdma-interface is required with --validate-rdma"
  [[ "${RDMA_INTERFACE}" =~ ^[[:alnum:]_.:-]+$ ]] || die "invalid --rdma-interface"
  [[ -n "${RDMA_HCA}" ]] || die "--rdma-hca is required with --validate-rdma"
  [[ "${RDMA_HCA}" =~ ^mlx5_[0-9]+(:[0-9]+)?(,mlx5_[0-9]+(:[0-9]+)?)*$ ]] \
    || die "--rdma-hca must be a comma-separated mlx5_N[:PORT] list"
  [[ "${RDMA_GID_INDEX}" =~ ^[0-9]+$ ]] || die "--rdma-gid-index must be a nonnegative integer"
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
  local -a packages=(
    ca-certificates
    curl
    gnupg
    lsb-release
    shellcheck
    sshpass
    software-properties-common
    ubuntu-drivers-common
  )
  if [[ "${VALIDATE_RDMA}" == "1" ]]; then
    packages+=(
      ibverbs-providers
      ibverbs-utils
      iproute2
      rdma-core
    )
  fi
  apt_get install -y "${packages[@]}"

  log "Ensuring Ubuntu multiverse is enabled for NVIDIA driver packages"
  add-apt-repository -y multiverse >/dev/null
}

install_docker_repository() {
  log "Configuring Docker CE apt repository"

  local arch
  arch="$(dpkg --print-architecture)"

  install -d -m 0755 /etc/apt/keyrings

  TMP_DIR="$(mktemp -d)"
  curl_with_retry -fsSL "https://download.docker.com/linux/ubuntu/gpg" -o "${TMP_DIR}/docker.asc"
  install -m 0644 "${TMP_DIR}/docker.asc" "${DOCKER_KEYRING}"

  printf 'deb [arch=%s signed-by=%s] https://download.docker.com/linux/ubuntu %s stable\n' \
    "${arch}" "${DOCKER_KEYRING}" "${VERSION_CODENAME}" > "${DOCKER_SOURCE_LIST}"
}

install_nvidia_container_toolkit_repository() {
  log "Configuring NVIDIA Container Toolkit apt repository"

  install -d -m 0755 "$(dirname "${NVIDIA_CTK_KEYRING}")"

  TMP_DIR="${TMP_DIR:-$(mktemp -d)}"
  curl_with_retry -fsSL "https://nvidia.github.io/libnvidia-container/gpgkey" \
    | gpg --dearmor -o "${TMP_DIR}/nvidia-container-toolkit-keyring.gpg"
  install -m 0644 "${TMP_DIR}/nvidia-container-toolkit-keyring.gpg" "${NVIDIA_CTK_KEYRING}"

  curl_with_retry -fsSL "https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list" \
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

  local managed_config
  managed_config="$(cat <<'EOF'
# Managed by ci/scripts/colossus/setup_bare_metal_gpu_host.sh.
# GB300 workaround: avoid nvidia-smi hangs in nv_remove_numa_memory/raw_q_flush.
options nvidia NVreg_EnableUserNUMAManagement=0
EOF
  )"

  if [[ -r "${NVIDIA_MODPROBE_CONFIG}" && "$(< "${NVIDIA_MODPROBE_CONFIG}")" == "${managed_config}" ]]; then
    log "NVIDIA GB300 NUMA workaround is already current; skipping initramfs update"
    return
  fi

  log "Persisting NVIDIA GB300 NUMA workaround in ${NVIDIA_MODPROBE_CONFIG}"
  printf '%s\n' "${managed_config}" > "${NVIDIA_MODPROBE_CONFIG}"

  if command -v update-initramfs >/dev/null 2>&1; then
    log "Updating initramfs so NVIDIA module options apply on next boot"
    update-initramfs -u
  fi
}

install_docker_and_toolkit() {
  log "Installing Docker CE, Podman, and NVIDIA Container Toolkit"
  apt_get update
  remove_conflicting_docker_packages
  apt_get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin \
    nvidia-container-toolkit \
    podman

  log "Configuring NVIDIA runtime for Docker"
  nvidia-ctk runtime configure --runtime=docker

  systemctl enable --now docker
  systemctl restart docker
  systemctl enable --now podman.socket

  if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
    log "Adding ${SUDO_USER} to the docker group"
    usermod -aG docker "${SUDO_USER}"
  fi
}

repair_podman_apparmor_profile() {
  local profile="/etc/apparmor.d/podman"
  if [[ ! -f "${profile}" ]] || ! command -v apparmor_parser >/dev/null 2>&1; then
    return
  fi

  # Ubuntu ships this named profile as unconfined so Podman can load its
  # libraries and apply container-specific confinement. Some Colossus host
  # hardening images strip the flag while leaving the otherwise empty profile,
  # which makes AppArmor deny Podman's dynamic loader before main() starts.
  if grep -Fxq 'profile podman /usr/bin/podman {' "${profile}"; then
    log "Restoring Ubuntu's unconfined AppArmor flag for Podman"
    sed -i \
      's|^profile podman /usr/bin/podman {$|profile podman /usr/bin/podman flags=(unconfined) {|' \
      "${profile}"
    apparmor_parser -r "${profile}"
  fi
}

live_nvidia_param() {
  local key="$1"
  if [[ -r /proc/driver/nvidia/params ]]; then
    awk -F': ' -v key="${key}" '$1 == key { print $2 }' /proc/driver/nvidia/params
  fi
}

validate_rdma_host() {
  if [[ "${VALIDATE_RDMA}" != "1" ]]; then
    return
  fi

  log "Validating fail-closed RDMA configuration on ${RDMA_INTERFACE}"
  modprobe ib_uverbs 2>/dev/null || true
  modprobe rdma_cm 2>/dev/null || true
  modprobe mlx5_ib 2>/dev/null || true

  ip link show dev "${RDMA_INTERFACE}" >/dev/null \
    || die "RDMA interface ${RDMA_INTERFACE} does not exist"
  [[ "$(< "/sys/class/net/${RDMA_INTERFACE}/operstate")" == "up" ]] \
    || die "RDMA interface ${RDMA_INTERFACE} is not up"
  ip -o -4 addr show dev "${RDMA_INTERFACE}" scope global | grep -q '[[:space:]]inet[[:space:]]' \
    || die "RDMA interface ${RDMA_INTERFACE} has no global IPv4 address"

  RDMA_UVERBS_DEVICES=()
  local hca_spec hca port state link_layer gid gid_type gid_netdev verbs_dir uverbs
  local state_path link_layer_path gid_path gid_type_path gid_netdev_path
  local -a hca_specs matches
  IFS=',' read -r -a hca_specs <<< "${RDMA_HCA}"
  for hca_spec in "${hca_specs[@]}"; do
    hca="${hca_spec%%:*}"
    port="1"
    if [[ "${hca_spec}" == *:* ]]; then
      port="${hca_spec##*:}"
    fi

    [[ -d "/sys/class/infiniband/${hca}/ports/${port}" ]] \
      || die "RDMA HCA/port ${hca}:${port} is not present"
    state_path="/sys/class/infiniband/${hca}/ports/${port}/state"
    link_layer_path="/sys/class/infiniband/${hca}/ports/${port}/link_layer"
    gid_path="/sys/class/infiniband/${hca}/ports/${port}/gids/${RDMA_GID_INDEX}"
    gid_type_path="/sys/class/infiniband/${hca}/ports/${port}/gid_attrs/types/${RDMA_GID_INDEX}"
    gid_netdev_path="/sys/class/infiniband/${hca}/ports/${port}/gid_attrs/ndevs/${RDMA_GID_INDEX}"
    [[ -r "${state_path}" && -r "${link_layer_path}" ]] \
      || die "RDMA HCA/port ${hca}:${port} is missing state metadata"
    [[ -r "${gid_path}" && -r "${gid_type_path}" && -r "${gid_netdev_path}" ]] \
      || die "RDMA HCA/port ${hca}:${port} has no GID metadata at index ${RDMA_GID_INDEX}"

    state="$(< "${state_path}")"
    [[ "${state}" == 4:* ]] || die "RDMA HCA/port ${hca}:${port} is not active: ${state}"
    link_layer="$(< "${link_layer_path}")"
    [[ "${link_layer}" == "Ethernet" ]] \
      || die "RDMA HCA/port ${hca}:${port} is ${link_layer}, expected Ethernet/RoCE"

    gid="$(< "${gid_path}")"
    [[ "${gid}" != "0000:0000:0000:0000:0000:0000:0000:0000" ]] \
      || die "RDMA HCA/port ${hca}:${port} GID index ${RDMA_GID_INDEX} is empty"
    gid_type="$(< "${gid_type_path}")"
    [[ "${gid_type}" == "RoCE v2" ]] \
      || die "RDMA HCA/port ${hca}:${port} GID ${RDMA_GID_INDEX} is ${gid_type}, expected RoCE v2"
    gid_netdev="$(< "${gid_netdev_path}")"
    [[ "${gid_netdev}" == "${RDMA_INTERFACE}" ]] \
      || die "RDMA GID ${hca}:${port}/${RDMA_GID_INDEX} maps to ${gid_netdev}, expected ${RDMA_INTERFACE}"

    verbs_dir="/sys/class/infiniband/${hca}/device/infiniband_verbs"
    shopt -s nullglob
    matches=("${verbs_dir}"/uverbs*)
    shopt -u nullglob
    [[ "${#matches[@]}" -eq 1 ]] \
      || die "expected exactly one uverbs device for ${hca}, found ${#matches[@]}"
    uverbs="/dev/infiniband/${matches[0]##*/}"
    [[ -c "${uverbs}" ]] || die "matching RDMA character device is missing: ${uverbs}"
    if [[ ! " ${RDMA_UVERBS_DEVICES[*]} " =~ [[:space:]]${uverbs}[[:space:]] ]]; then
      RDMA_UVERBS_DEVICES+=("${uverbs}")
    fi

    ibv_devinfo -d "${hca}" -i "${port}" >/dev/null \
      || die "ibv_devinfo failed for ${hca}:${port}"
    log "RDMA ready: ${hca}:${port} GID ${RDMA_GID_INDEX} (${gid_type}) -> ${RDMA_INTERFACE}, ${uverbs}"
  done
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

podman_login_nvcr_if_requested() {
  if [[ -z "${NVCR_TOKEN}" ]]; then
    return
  fi

  log "Logging rootful Podman in to nvcr.io for Ray runtime images"
  install -d -m 0700 /root/.config/containers
  printf '%s' "${NVCR_TOKEN}" | podman login nvcr.io \
    --authfile /root/.config/containers/auth.json \
    --username '$oauthtoken' \
    --password-stdin >/dev/null
  chmod 0600 /root/.config/containers/auth.json
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
  timeout --kill-after=10s "${NVIDIA_TEST_IMAGE_PULL_TIMEOUT}s" docker pull "${NVIDIA_TEST_IMAGE}"
}

validate_installation() {
  log "Validating Docker service"
  docker info >/dev/null
  log "Validating rootful Podman service"
  podman info >/dev/null
  systemctl is-active --quiet podman.socket
  validate_rdma_host

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
    local -a docker_args=(--rm --gpus all)
    if [[ "${VALIDATE_RDMA}" == "1" ]]; then
      local uverbs expected_uverbs=""
      docker_args+=(--network host --ipc=host --ulimit memlock=-1)
      for uverbs in "${RDMA_UVERBS_DEVICES[@]}"; do
        docker_args+=(--device "${uverbs}:${uverbs}")
        expected_uverbs+="${expected_uverbs:+:}${uverbs}"
      done
      docker_args+=(--env "GROOT_RDMA_UVERBS=${expected_uverbs}")
    fi
    if [[ "${VALIDATE_RDMA}" == "1" ]]; then
      timeout --kill-after=10s "${NVIDIA_DOCKER_GPU_GATE_TIMEOUT}s" \
        docker run "${docker_args[@]}" "${NVIDIA_TEST_IMAGE}" bash -lc '
          set -euo pipefail
          nvidia-smi -L
          IFS=: read -r -a devices <<< "${GROOT_RDMA_UVERBS:?}"
          for device in "${devices[@]}"; do test -c "${device}"; done
        '
    else
      timeout --kill-after=10s "${NVIDIA_DOCKER_GPU_GATE_TIMEOUT}s" \
        docker run "${docker_args[@]}" "${NVIDIA_TEST_IMAGE}" nvidia-smi -L
    fi
  else
    log "Skipping Docker GPU image test; set NVIDIA_TEST_IMAGE to enable it"
  fi
}

main() {
  parse_args "$@"
  validate_rdma_args
  if [[ ! "${NVIDIA_TEST_IMAGE_PULL_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
    die "NVIDIA_TEST_IMAGE_PULL_TIMEOUT must be a positive integer"
  fi
  if [[ ! "${NVIDIA_DOCKER_GPU_GATE_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
    die "NVIDIA_DOCKER_GPU_GATE_TIMEOUT must be a positive integer"
  fi
  require_root
  load_os_release
  check_architecture

  install_prerequisites
  install_docker_repository
  install_nvidia_container_toolkit_repository
  install_nvidia_driver
  persist_nvidia_module_options
  install_docker_and_toolkit
  repair_podman_apparmor_profile
  podman_login_nvcr_if_requested
  validate_installation

  log "Provisioning complete"
  log "Reboot is recommended before using the GPU so the installed driver and module options are active"
  if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
    log "${SUDO_USER} must log out and back in before docker group membership is active"
  fi
}

main "$@"
