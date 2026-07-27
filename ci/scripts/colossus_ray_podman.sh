#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ray 2.48 invokes a binary named `podman` for runtime_env.image_uri. The Ray
# control process lives in Docker, so this wrapper forwards to the host's
# rootful Podman service and adds the node-local mounts and devices required by
# One Launcher jobs.

set -euo pipefail

remote="${COLOSSUS_PODMAN_REMOTE:-/usr/bin/podman}"
data_root="${COLOSSUS_DATA_ROOT:?COLOSSUS_DATA_ROOT is required}"
job_root="${COLOSSUS_JOB_ROOT:?COLOSSUS_JOB_ROOT is required}"
gpu_device="${COLOSSUS_GPU_CDI_DEVICE:?COLOSSUS_GPU_CDI_DEVICE is required}"

[[ -x "${remote}" ]] || {
  echo "Colossus Podman remote client is not executable: ${remote}" >&2
  exit 1
}
[[ $# -gt 0 ]] || {
  echo "Colossus Podman wrapper requires a command" >&2
  exit 1
}

if [[ "$1" != "run" ]]; then
  exec "${remote}" --remote "$@"
fi
shift

declare -a forwarded=()
while [[ $# -gt 0 ]]; do
  # Ray assumes rootless Podman. The host service is deliberately rootful and
  # rejects/does not need keep-id user namespaces. The Ubuntu 22.04 Podman
  # remote client also does not recognize Ray's cgroup-manager run option;
  # the host service already uses its configured manager.
  if [[ "$1" == "--userns=keep-id" || "$1" == "--cgroup-manager=cgroupfs" ]]; then
    shift
    continue
  fi
  forwarded+=("$1")
  shift
done

declare -a injected=(
  --network host
  --ipc host
  --volume "${data_root}:${data_root}"
  --volume "${job_root}:${job_root}"
)

if [[ -n "${COLOSSUS_FIPS_MASK:-}" ]]; then
  injected+=(--volume "${COLOSSUS_FIPS_MASK}:/proc/sys/crypto/fips_enabled:ro")
fi

# Ray first runs a Python import to discover default_worker.py in the requested
# image. Images such as groot-ci-train have a verbose NVIDIA entrypoint, whose
# banner would be mistaken for part of that path. Bypass only that probe's
# image entrypoint; Ray supplies an explicit Python entrypoint to real workers.
ray_worker=0
worker_probe=0
for argument in "${forwarded[@]}"; do
  if [[ "${argument}" == *"import ray._private.workers.default_worker"* ]]; then
    worker_probe=1
    injected+=(--entrypoint /usr/bin/env)
  elif [[ "${argument}" == */ray/_private/workers/default_worker.py ]]; then
    ray_worker=1
  fi
done

cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-}"
# Ray assigns CUDA_VISIBLE_DEVICES inside default_worker.py, after Podman has
# already created the container. Attach the node's sole physical GPU to every
# real Ray worker container and let Ray's resource assignment control process
# visibility. The import-only image probe remains device-free.
if [[ "${worker_probe}" != "1" ]] && {
  [[ "${ray_worker}" == "1" ]] ||
    [[ -n "${cuda_visible_devices}" && "${cuda_visible_devices}" != "NoDevFiles" ]]
}; then
  injected+=(
    --device "${gpu_device}"
    --env "CUDA_VISIBLE_DEVICES=0"
    --ulimit "memlock=-1"
    --ulimit "stack=67108864"
  )
  for name in \
    NCCL_IB_DISABLE \
    NCCL_IB_HCA \
    NCCL_IB_GID_INDEX \
    NCCL_SOCKET_IFNAME \
    NCCL_DEBUG \
    NCCL_DEBUG_SUBSYS \
    GLOO_SOCKET_IFNAME \
    RAY_ADDRESS
  do
    value="${!name:-}"
    if [[ -n "${value}" ]]; then
      injected+=(--env "${name}=${value}")
    fi
  done
  if [[ -n "${COLOSSUS_RDMA_DEVICES:-}" ]]; then
    injected+=(--env "COLOSSUS_RDMA_DEVICES=${COLOSSUS_RDMA_DEVICES}")
    IFS=: read -r -a rdma_devices <<< "${COLOSSUS_RDMA_DEVICES}"
    for device in "${rdma_devices[@]}"; do
      [[ -n "${device}" ]] && injected+=(--device "${device}:${device}")
    done
  fi
fi

exec "${remote}" --remote run "${injected[@]}" "${forwarded[@]}"
