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

# Ray 2.48 invokes a binary named `podman` for runtime_env.image_uri. The Ray
# control process lives in Docker, so this wrapper forwards to the host's
# rootful Podman service and adds the node-local mounts and devices required by
# One Launcher jobs.

set -euo pipefail

remote="${COLOSSUS_PODMAN_REMOTE:-/usr/bin/podman}"
data_root="${COLOSSUS_DATA_ROOT:?COLOSSUS_DATA_ROOT is required}"
job_root="${COLOSSUS_JOB_ROOT:?COLOSSUS_JOB_ROOT is required}"
gpu_devices="${COLOSSUS_GPU_CDI_DEVICES:-${COLOSSUS_GPU_CDI_DEVICE:-}}"
[[ -n "${gpu_devices}" ]] || {
  echo "COLOSSUS_GPU_CDI_DEVICES or COLOSSUS_GPU_CDI_DEVICE is required" >&2
  exit 1
}

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

if [[ -d "${data_root}/runtime/robocasa" ]]; then
  injected+=(--volume "${data_root}/runtime/robocasa:/opt/grootrobocasa/robocasa")
fi
if [[ -d "${data_root}/runtime/robosuite" ]]; then
  injected+=(--volume "${data_root}/runtime/robosuite:/opt/grootrobosuite/robosuite:ro")
fi
if [[ -d "${data_root}/ckpts" ]]; then
  injected+=(--volume "${data_root}/ckpts:/data/model_deps:ro")
fi

if [[ -n "${COLOSSUS_FIPS_MASK:-}" ]]; then
  injected+=(--volume "${COLOSSUS_FIPS_MASK}:/proc/sys/crypto/fips_enabled:ro")
fi

# Ray first runs a Python import to discover default_worker.py in the requested
# image. Images such as groot-ci-train have a verbose NVIDIA entrypoint, whose
# banner would be mistaken for part of that path. Bypass only that probe's
# image entrypoint; Ray supplies an explicit Python entrypoint to real workers.
ray_worker=0
worker_probe=0
job_expected_gpu_count=""
heterogeneous_evaluation=0
explicit_msc_config=0
explicit_aws_access_key=0
explicit_aws_secret_key=0
gear_data_credentials=""
has_auto_remove=0
for argument in "${forwarded[@]}"; do
  if [[ "${argument}" == *"import ray._private.workers.default_worker"* ]]; then
    worker_probe=1
    injected+=(--entrypoint /usr/bin/env)
  elif [[ "${argument}" == */ray/_private/workers/default_worker.py ]]; then
    ray_worker=1
  elif [[ "${argument}" == EXPECTED_NUM_GPUS=* ]]; then
    job_expected_gpu_count="${argument#EXPECTED_NUM_GPUS=}"
  elif [[ "${argument}" == GROOT_OL_COLOSSUS_HETEROGENEOUS_EVALUATION=1 ]]; then
    heterogeneous_evaluation=1
  elif [[ "${argument}" == MSC_CONFIG=* ]]; then
    explicit_msc_config=1
  elif [[ "${argument}" == AWS_ACCESS_KEY_ID=* ]]; then
    explicit_aws_access_key=1
  elif [[ "${argument}" == AWS_SECRET_ACCESS_KEY=* ]]; then
    explicit_aws_secret_key=1
  elif [[ "${argument}" == GEAR_DATA_CREDENTIALS=* ]]; then
    gear_data_credentials="${argument#GEAR_DATA_CREDENTIALS=}"
  elif [[ "${argument}" == "--rm" ]]; then
    has_auto_remove=1
  fi
done
if [[ "${has_auto_remove}" != "1" ]]; then
  # Ray 2.48's image_uri worker command omits --rm. Remove each runtime
  # container as soon as its worker exits instead of accumulating stale
  # containers indefinitely on a persistent station.
  injected+=(--rm)
fi
if [[ -n "${gear_data_credentials}" ]] && {
  [[ "${explicit_aws_access_key}" != "1" ]] || [[ "${explicit_aws_secret_key}" != "1" ]]
}; then
  # One Launcher also supports credentials stored only in
  # ~/.gear/data_credentials. Ray forwards that file as base64 so workers can
  # materialize it; derive missing AWS variables from the same payload without
  # putting credential values in the shared MSC profile or on the host.
  declare -a derived_aws_credentials=()
  while IFS= read -r -d '' credential; do
    derived_aws_credentials+=("${credential}")
  done < <(
    GEAR_DATA_CREDENTIALS="${gear_data_credentials}" python3 - <<'PY' 2>/dev/null
import base64
import json
import os
import sys

encoded = os.environ["GEAR_DATA_CREDENTIALS"]
credentials = json.loads(base64.b64decode(encoded, validate=True))
profile = credentials.get("default") if isinstance(credentials, dict) else None
if not isinstance(profile, dict):
    raise SystemExit(1)
for environment_name, profile_name in (
    ("AWS_ACCESS_KEY_ID", "aws_access_key_id"),
    ("AWS_SECRET_ACCESS_KEY", "aws_secret_access_key"),
):
    value = profile.get(profile_name)
    if not isinstance(value, str) or not value or any(char in value for char in "\x00\r\n"):
        raise SystemExit(1)
    sys.stdout.buffer.write(f"{environment_name}={value}".encode() + b"\x00")
PY
  )
  for credential in "${derived_aws_credentials[@]}"; do
    case "${credential}" in
      AWS_ACCESS_KEY_ID=*)
        [[ "${explicit_aws_access_key}" == "1" ]] || injected+=(--env "${credential}")
        ;;
      AWS_SECRET_ACCESS_KEY=*)
        [[ "${explicit_aws_secret_key}" == "1" ]] || injected+=(--env "${credential}")
        ;;
    esac
  done
fi
msc_config="${COLOSSUS_MSC_CONFIG:-${data_root}/config/msc/config.json}"
if [[ "${explicit_msc_config}" != "1" && -f "${msc_config}" ]]; then
  # The cluster bootstrap installs a non-secret profile whose credential values
  # are environment-variable references. Ray supplies those credential
  # variables per job; this only makes the shared profile discoverable inside
  # image_uri workers, including evaluation-only workers on a cold node.
  injected+=(--env "MSC_CONFIG=${msc_config}")
fi
if [[ "${worker_probe}" == "1" ]]; then
  # Ray captures this command's stdout as the default_worker.py path. On a
  # cold node, Podman otherwise prefixes that path with its implicit image-pull
  # progress and Ray later tries to execute the whole transcript as a filename.
  # --quiet suppresses only Podman's pull status, not the container's stdout.
  injected+=(--quiet)
fi

cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-}"
# Ray assigns CUDA_VISIBLE_DEVICES inside default_worker.py, after Podman has
# already created the container. Attach the node's ordered physical GPU set to
# every real Ray worker container and let Ray's logical resource assignment
# control process visibility. The import-only image probe remains device-free.
if [[ "${worker_probe}" != "1" ]] && {
  [[ "${ray_worker}" == "1" ]] ||
    [[ -n "${cuda_visible_devices}" && "${cuda_visible_devices}" != "NoDevFiles" ]]
}; then
  IFS=',' read -r -a ordered_gpu_devices <<< "${gpu_devices}"
  [[ "${#ordered_gpu_devices[@]}" -gt 0 ]] || {
    echo "No ordered CDI GPU devices were configured" >&2
    exit 1
  }
  if [[ "${ray_worker}" == "1" ]]; then
    # Ray's generic GPU allocator can return either logical slot from a
    # heterogeneous node, especially while replacing a failed actor. Preserve
    # the bootstrap-validated GB300 UUID separately so the training worker can
    # bind CUDA to physical slot 0 before it initializes torch. Evaluation
    # actors keep using Ray's normal slot assignment.
    gb300_gpu_uuid="${ordered_gpu_devices[0]#nvidia.com/gpu=}"
    [[ "${gb300_gpu_uuid}" == GPU-* ]] || {
      echo "Ordered GB300 CDI device does not contain a GPU UUID: ${ordered_gpu_devices[0]}" >&2
      exit 1
    }
    injected+=(--env "GROOT_OL_COLOSSUS_TRAINING_CUDA_DEVICE=${gb300_gpu_uuid}")
  fi
  declare -a selected_gpu_indices=()
  if [[ "${ray_worker}" != "1" && "${cuda_visible_devices}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    IFS=',' read -r -a selected_gpu_indices <<< "${cuda_visible_devices}"
  elif [[ "${ray_worker}" == "1" && "${heterogeneous_evaluation}" != "1" ]]; then
    # Training always uses one GB300 per station, even when the distributed
    # job requests several stations. Ray's generic GPU allocator is not aware
    # that logical slot 0 and slot 1 are different models, so preserve the
    # bootstrap-validated UUID and prevent default_worker.py from replacing it
    # with a generic logical ID before torch imports initialize CUDA.
    selected_gpu_indices=(0)
    injected+=(--env "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1")
  else
    # Evaluation-config jobs need both devices attached before default_worker
    # starts. Ray later narrows CUDA_VISIBLE_DEVICES to logical slot 0 for the
    # GB300 policy/trainer actor and slot 1 for the RTX simulator actor.
    selected_gpu_indices=("${!ordered_gpu_devices[@]}")
  fi
  cuda_device_uuids=""
  for index in "${selected_gpu_indices[@]}"; do
    [[ "${index}" =~ ^[0-9]+$ && "${index}" -lt "${#ordered_gpu_devices[@]}" ]] || {
      echo "Requested CUDA device index ${index} is outside the configured CDI set" >&2
      exit 1
    }
    [[ -n "${ordered_gpu_devices[$index]}" ]] || {
      echo "Empty CDI GPU device at ordered index ${index}" >&2
      exit 1
    }
    injected+=(--device "${ordered_gpu_devices[$index]}")
    gpu_uuid="${ordered_gpu_devices[$index]#nvidia.com/gpu=}"
    [[ "${gpu_uuid}" == GPU-* ]] || {
      echo "CDI GPU device does not contain a GPU UUID: ${ordered_gpu_devices[$index]}" >&2
      exit 1
    }
    cuda_device_uuids+="${cuda_device_uuids:+,}${gpu_uuid}"
  done
  injected+=(
    # UUID visibility makes CUDA enumerate the logical Ray slots in this exact
    # order even when /dev/nvidia* retains a different host-physical order.
    --env "CUDA_VISIBLE_DEVICES=${cuda_device_uuids}"
    --env "MUJOCO_GL=egl"
    --env "PYOPENGL_PLATFORM=egl"
    --env "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics"
    --env "PYTHONPATH=${data_root}/runtime/python:/opt/grootrobosuite:/opt/grootrobocasa:${PYTHONPATH:-}"
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
