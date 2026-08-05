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
# Reserve Colossus bare-metal GPU hosts and create or extend a persistent Ray cluster.

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_SETUP_SCRIPT="${SCRIPT_DIR}/setup_bare_metal_gpu_host.sh"
PODMAN_WRAPPER="${SCRIPT_DIR}/../colossus_ray_podman.sh"
PRUNE_SCRIPT="${SCRIPT_DIR}/../colossus_ray_prune.sh"
RAY_COMPAT_SCRIPT="${SCRIPT_DIR}/../../../groot/internal/infra/job/colossus_ray_compat/sitecustomize.py"

IMAGE="${COLOSSUS_RAY_CONTROL_IMAGE:-rayproject/ray:2.48.0-py312-gpu}"
GPU_TEST_IMAGE="${COLOSSUS_GPU_TEST_IMAGE:-docker.io/nvidia/cuda:13.0.1-base-ubuntu24.04}"
ARTIFACT_IMAGE="${COLOSSUS_ARTIFACT_IMAGE:-python:3.12-slim-bookworm}"
PROMETHEUS_IMAGE="${COLOSSUS_PROMETHEUS_IMAGE:-prom/prometheus:v3.5.0}"
GRAFANA_IMAGE="${COLOSSUS_GRAFANA_IMAGE:-grafana/grafana:12.0.2}"
COLOSSUS_POOL_ID="${COLOSSUS_POOL_ID:-gear-gb300-automation}"
COLOSSUS_NODE_COUNT="${COLOSSUS_NODE_COUNT:-}"
COLOSSUS_RESOURCE_GPU_COUNT="${COLOSSUS_RESOURCE_GPU_COUNT:-}"
COLOSSUS_EXISTING_LEASE_ID="${COLOSSUS_EXISTING_LEASE_ID:-}"
COLOSSUS_EXISTING_LEASE_NAME=""
COLOSSUS_LEASE_DURATION="${COLOSSUS_LEASE_DURATION:-7d}"
COLOSSUS_LEASE_JUSTIFICATION="${COLOSSUS_LEASE_JUSTIFICATION:-GR00T persistent Ray cluster}"
COLOSSUS_OS="${COLOSSUS_OS:-ubuntu-24.04-aarch64-standard-uefi}"
COLOSSUS_PROVISION_MODE="${COLOSSUS_PROVISION_MODE:-CLEAN}"
COLOSSUS_LEASE_READY_TIMEOUT="${COLOSSUS_LEASE_READY_TIMEOUT:-3600}"
COLOSSUS_LEASE_POLL_INTERVAL="${COLOSSUS_LEASE_POLL_INTERVAL:-15}"
KEEP_COLOSSUS_LEASES_ON_FAILURE="${KEEP_COLOSSUS_LEASES_ON_FAILURE:-0}"
BEST_EFFORT_HOSTS="${BEST_EFFORT_HOSTS:-0}"
COLOSSUS_LEASES_COMMITTED="0"
SSL_CERT="${SSL_CERT:-/etc/ssl/groot-ray/fullchain.pem}"
SSL_KEY="${SSL_KEY:-/etc/ssl/groot-ray/privkey.pem}"
LOCAL_SSL_CERT="${LOCAL_SSL_CERT:-}"
LOCAL_SSL_KEY="${LOCAL_SSL_KEY:-}"
HEAD_DOMAIN=""
HEAD_DOMAIN_EXPLICIT="0"
JOIN_RAY_ADDRESS=""
JOIN_RAY_HEAD_IP=""
JOIN_RAY_HEAD_PORT=""
RAY_AUTH_USER="${RAY_AUTH_USER:-gear}"
RAY_AUTH_PASSWORD=""
RAY_AUTH_PASSWORD_SOURCE=""
RAY_AUTH_PASSWORD_ENV=""
NVCR_TOKEN="${NVCR_TOKEN:-}"
NVCR_TOKEN_SOURCE=""
NVCR_TOKEN_ENV=""
SYNC_POST_TRAINING_ARTIFACTS="${SYNC_POST_TRAINING_ARTIFACTS:-1}"
GROOT_DATA_ROOT="${GROOT_DATA_ROOT:-/data/groot_ol}"
MODEL_DEPS_URI="${MODEL_DEPS_URI:-s3://GearCheckpoints/model_deps}"
POST_TRAINING_CHECKPOINT_URI="${POST_TRAINING_CHECKPOINT_URI:-${GROOT_OL_POST_TRAINING_CHECKPOINT_URI:-}}"
POST_TRAINING_CHECKPOINT_DESTINATION="${POST_TRAINING_CHECKPOINT_DESTINATION:-post_training_golden_run/checkpoint-20000}"
DATASET_ARCHIVE_URI="${DATASET_ARCHIVE_URI:-}"
DATASET_ARCHIVE_SHA256="${DATASET_ARCHIVE_SHA256:-}"
ROBOCASA_RUNTIME_ARCHIVE_URI="${ROBOCASA_RUNTIME_ARCHIVE_URI:-}"
ROBOCASA_RUNTIME_ARCHIVE_SHA256="${ROBOCASA_RUNTIME_ARCHIVE_SHA256:-}"
ARTIFACT_MAX_MIB_PER_SECOND="${ARTIFACT_MAX_MIB_PER_SECOND:-0}"
BUCKET_ACCESS_KEY_ID_ENV="${BUCKET_ACCESS_KEY_ID_ENV:-BUCKET_ACCESS_KEY_ID}"
BUCKET_SECRET_ACCESS_KEY_ENV="${BUCKET_SECRET_ACCESS_KEY_ENV:-BUCKET_SECRET_ACCESS_KEY}"
BUCKET_ENDPOINT_URL_ENV="${BUCKET_ENDPOINT_URL_ENV:-BUCKET_ENDPOINT_URL}"
BUCKET_REGION_ENV="${BUCKET_REGION_ENV:-BUCKET_REGION}"
BUCKET_ACCESS_KEY_ID_VALUE=""
BUCKET_SECRET_ACCESS_KEY_VALUE=""
BUCKET_ENDPOINT_URL_VALUE=""
BUCKET_REGION_VALUE=""
DRY_RUN="0"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-20}"
REBOOT_WAIT_TIMEOUT="${REBOOT_WAIT_TIMEOUT:-1800}"
REBOOT_POLL_INTERVAL="${REBOOT_POLL_INTERVAL:-10}"
HOST_SETUP_CONCURRENCY="${HOST_SETUP_CONCURRENCY:-8}"
DATA_INTERFACE="${DATA_INTERFACE:-}"
PREFERRED_ETHERNET_INTERFACE="${PREFERRED_ETHERNET_INTERFACE:-}"
STATIC_DATA_IP_PREFIX="${STATIC_DATA_IP_PREFIX:-}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-}"
NCCL_IB_HCA="${NCCL_IB_HCA:-}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-}"
REQUIRE_RDMA="${REQUIRE_RDMA:-0}"
ETHERNET_ONLY="${ETHERNET_ONLY:-0}"
TRANSPORT_VALIDATION_COMMAND="${TRANSPORT_VALIDATION_COMMAND:-}"
TRANSPORT_VALIDATION_TIMEOUT="${TRANSPORT_VALIDATION_TIMEOUT:-300}"
TRANSPORT_VALIDATION_IMAGE="${COLOSSUS_TRANSPORT_VALIDATION_IMAGE:-}"
STATE_OUTPUT="${COLOSSUS_RAY_STATE_OUTPUT:-}"
ACTIVE_RAY_HEAD_IP=""
REMOTE_TLS_STAGING_ACTIVE="0"
REMOTE_TLS_CERT_STAGING=""
REMOTE_TLS_KEY_STAGING=""

readonly HEAD_CONTAINER="groot-ray-head"
readonly WORKER_CONTAINER="groot-ray-worker"
readonly REMOTE_ROOT="/tmp/groot-ray-cluster"
readonly RAY_PORT="6379"
readonly TARGET_GPU_NAME="NVIDIA GB300"
readonly SECONDARY_GPU_NAME_PATTERN="RTX PRO 2000"
GPU_PROFILE="${GPU_PROFILE:-gb300}"

TMP_DIR=""
declare -a HOSTS=()
declare -a USERS=()
declare -a PASSWORDS=()
declare -a HOST_READY=()
declare -a GPU_COUNTS=()
declare -a NODE_IPS=()
declare -a DATA_IPS=()
declare -a DATA_INTERFACES=()
declare -a FAILED_HOSTS=()
declare -a HOST_SETUP_PIDS=()
declare -a JOINED_WORKER_INDICES=()
declare -a COLOSSUS_CANDIDATE_NAMES=()
declare -a COLOSSUS_CANDIDATE_FQDNS=()
declare -a COLOSSUS_EXCLUDED_RESOURCE_NAMES=()
declare -a COLOSSUS_LEASE_IDS=()
declare -a COLOSSUS_LEASE_NAMES=()

log() {
  printf '[setup_ray_cluster] %s\n' "$*"
}

colossus_cli() {
  # The Colossus CLI updates shared authentication/configuration state even
  # for read operations. Parallel auto-research host setup otherwise races
  # those updates and intermittently loses the configured API server.
  local lock_file="${COLOSSUS_CLI_LOCK_FILE:-/tmp/groot-colossus-cli.lock}"
  (
    flock --exclusive 9
    command colossus "$@"
  ) 9>"${lock_file}"
}

die() {
  printf '[setup_ray_cluster] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage:
  # Create a new Ray cluster:
  ci/scripts/colossus/setup_ray_cluster.sh \\
    --nodes 2 \\
    --ray-auth-password-file /path/to/password \\
    --nvcr-token-file /path/to/nvcr-token

  # Add workers to an existing Ray cluster without changing its head:
  ci/scripts/colossus/setup_ray_cluster.sh \\
    --nodes 2 \\
    --join-ray-address 10.0.0.10:6379 \\
    --nvcr-token-file /path/to/nvcr-token

Options:
  --nodes COUNT                 Number of Colossus hosts to reserve. The first lease is
                                the Ray head unless --join-ray-address is set.
  --resource-gpu-count COUNT    Require this physical GPU count in Colossus inventory.
  --exclude-resource NAME       Never reserve this Colossus resource; may be repeated.
  --best-effort-hosts           Continue with successfully reserved and SSH-ready hosts.
  --gpu-profile PROFILE         gb300 or gb300-rtx-pro-2000. Default: ${GPU_PROFILE}
  --existing-lease-id ID        Reuse this active Colossus lease instead of reserving a
                                host. Requires --nodes 1. The lease is never released by
                                setup, including when setup fails.
  --join-ray-address IP[:PORT]  Add every leased host as a worker to this existing Ray
                                control-plane address. Default port: ${RAY_PORT}.
                                The existing head and its nginx endpoint are not changed.
  --pool-id ID                  Colossus bare-metal pool ID. Default: ${COLOSSUS_POOL_ID}
  --lease-duration DURATION     Colossus lease duration. Default: ${COLOSSUS_LEASE_DURATION}
  --lease-justification TEXT    Colossus lease justification. Default: ${COLOSSUS_LEASE_JUSTIFICATION}
  --colossus-os OS              OS requested for each lease. Default: ${COLOSSUS_OS}
  --colossus-provision-mode MODE
                                Colossus provisioning mode. Default: ${COLOSSUS_PROVISION_MODE}
  --lease-ready-timeout SECS    Overall wait for all leases to become active. Default: ${COLOSSUS_LEASE_READY_TIMEOUT}
  --lease-poll-interval SECS    Seconds between lease readiness checks. Default: ${COLOSSUS_LEASE_POLL_INTERVAL}
  --keep-leases-on-failure      Do not release newly created leases when setup fails.
  --image IMAGE                 Public Ray control-plane image. Default: ${IMAGE}
  --gpu-test-image IMAGE        Public CUDA image for the GB300 container gate.
                                Default: ${GPU_TEST_IMAGE}
  --artifact-image IMAGE        Public Python image used to stage artifacts.
                                Default: ${ARTIFACT_IMAGE}
  --ssl-cert PATH               TLS certificate path on the head host. Default: ${SSL_CERT}
  --ssl-key PATH                TLS private key path on the head host. Default: ${SSL_KEY}
  --ssl-cert-local PATH         Copy this local certificate to --ssl-cert on the head.
  --ssl-key-local PATH          Copy this local key to --ssl-key on the head.
  --head-domain DOMAIN          Public HTTPS domain. Default: first leased host FQDN.
  --ray-auth-user USER          nginx basic-auth username for /ray/. Default: ${RAY_AUTH_USER}
  --ray-auth-password-file PATH Read basic-auth password from a local file.
  --ray-auth-password-env VAR   Read basic-auth password from environment variable VAR.
  --ray-auth-password-stdin     Read basic-auth password from stdin.
  --nvcr-token-file PATH        Read NVCR token from a local file.
  --nvcr-token-env VAR          Read NVCR token from environment variable VAR.
  --nvcr-token-stdin            Read NVCR token from stdin.
  --groot-data-root PATH        Host/container root for model deps, checkpoints, and caches.
                                Default: ${GROOT_DATA_ROOT}
  --model-deps-uri URI          S3 root containing model_deps. Default: ${MODEL_DEPS_URI}
  --post-training-checkpoint-uri URI
                                S3 prefix for the base post-training checkpoint to mirror to
                                ${GROOT_DATA_ROOT}/checkpoints/post_training_golden_run/checkpoint-20000.
                                Can also be set with GROOT_OL_POST_TRAINING_CHECKPOINT_URI.
  --post-training-checkpoint-destination PATH
                                Relative directory under data-root/checkpoints.
  --dataset-archive-uri URI     Optional S3 tar archive extracted under data-root/datasets.
  --dataset-archive-sha256 HEX  Required SHA-256 when --dataset-archive-uri is set.
  --robocasa-runtime-archive-uri URI
                                Optional S3 tar archive extracted under data-root/runtime.
  --robocasa-runtime-archive-sha256 HEX
                                Required SHA-256 when the runtime archive is set.
  --artifact-max-mib-per-second RATE
                                Limit each host's artifact download rate. Zero disables
                                throttling. Default: ${ARTIFACT_MAX_MIB_PER_SECOND}.
  --bucket-access-key-id-env VAR
                                Env var containing S3 access key. Default: ${BUCKET_ACCESS_KEY_ID_ENV}
  --bucket-secret-access-key-env VAR
                                Env var containing S3 secret key. Default: ${BUCKET_SECRET_ACCESS_KEY_ENV}
  --bucket-endpoint-url-env VAR Env var containing S3 endpoint URL. Default: ${BUCKET_ENDPOINT_URL_ENV}
  --bucket-region-env VAR       Env var containing S3 region. Default: ${BUCKET_REGION_ENV}
  --skip-post-training-artifacts
                                Do not mirror model deps/checkpoint artifacts.
  --ssh-connect-timeout SECS    SSH connect timeout. Default: ${SSH_CONNECT_TIMEOUT}
  --reboot-wait-timeout SECS    Overall seconds to wait for a reboot. Default: ${REBOOT_WAIT_TIMEOUT}
  --reboot-poll-interval SECS   Seconds between SSH reboot polls. Default: ${REBOOT_POLL_INTERVAL}
  --host-setup-concurrency N    Maximum hosts to prepare concurrently. Default: ${HOST_SETUP_CONCURRENCY}
  --data-interface IFACE       Data-plane netdev used by Ray, NCCL bootstrap, and Gloo.
  --prefer-ethernet-interface IFACE
                                Prefer this connected non-mlx5 Ethernet netdev per host;
                                fall back to the host's routed management interface when
                                it is absent or has no carrier.
  --static-data-ip-prefix A.B.C
                                Configure the selected data interface as
                                A.B.C.<host-derived>/24.
  --nccl-socket-ifname IFACE   NCCL socket netdev. Default: --data-interface value.
  --gloo-socket-ifname IFACE   Gloo socket netdev. Default: --data-interface value.
  --nccl-ib-hca HCA[:PORT]     Verbs HCA/port list for NCCL, for example mlx5_3:1.
                                Matching uverbs devices are derived from sysfs and passed to Podman.
  --nccl-ib-gid-index INDEX    RoCE GID index used by NCCL, for example 3.
  --require-rdma               Fail unless the selected HCA/GID and container RDMA path are usable.
  --allow-socket-transport     Permit legacy TCP-only operation. This is the default.
  --ethernet-only              Require the Ray data interface to use a non-mlx5 Ethernet
                                driver and set NCCL_IB_DISABLE=1.
  --transport-validation-command COMMAND
                                Run COMMAND in a head-host Podman GPU container after all GPUs
                                join Ray; a nonzero exit prevents cluster readiness.
  --transport-validation-image IMAGE
                                Image for COMMAND. Default: --gpu-test-image value.
  --transport-validation-timeout SECS
                                Bound the validation command. Default: ${TRANSPORT_VALIDATION_TIMEOUT}
  --dry-run                     Print the plan without changing any host.
                                Pool discovery is read-only; no lease is created.
  --state-output PATH           Write non-secret cluster/host state as JSON on success.
  -h, --help                    Show this help.

GPU policy:
  gb300 exposes only the ${TARGET_GPU_NAME}. gb300-rtx-pro-2000 exposes an
  ordered pair: Ray/CUDA slot 0 is GB300 and slot 1 is RTX PRO 2000.

If both --ray-auth-password-stdin and --nvcr-token-stdin are used, pass two lines:
  first line = Ray basic-auth password, second line = NVCR token.
EOF
}

terminate_host_setup_jobs() {
  local pid
  local -a pids=("${HOST_SETUP_PIDS[@]}")

  HOST_SETUP_PIDS=()
  set +m 2>/dev/null || true
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      # Host setup jobs are started in their own process groups. Signal the
      # whole group so an in-flight ssh/scp child cannot outlive the wrapper.
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}

forget_host_setup_pid() {
  local completed_pid="$1"
  local pid
  local -a remaining=()

  for pid in "${HOST_SETUP_PIDS[@]}"; do
    if [[ "${pid}" != "${completed_pid}" ]]; then
      remaining+=("${pid}")
    fi
  done
  HOST_SETUP_PIDS=("${remaining[@]}")
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  terminate_host_setup_jobs
  cleanup_remote_tls_staging
  if [[ "${status}" -ne 0 && -n "${JOIN_RAY_ADDRESS}" && "${#JOINED_WORKER_INDICES[@]}" -gt 0 ]]; then
    stop_joined_worker_containers
  fi
  if [[ "${status}" -ne 0 && "${COLOSSUS_LEASES_COMMITTED}" != "1" && "${#COLOSSUS_LEASE_IDS[@]}" -gt 0 ]]; then
    if [[ "${KEEP_COLOSSUS_LEASES_ON_FAILURE}" == "1" ]]; then
      log "setup failed; keeping Colossus leases: ${COLOSSUS_LEASE_IDS[*]}"
    else
      release_colossus_leases
    fi
  fi
  if [[ -n "${TMP_DIR}" && -d "${TMP_DIR}" ]]; then
    rm -rf "${TMP_DIR}" || true
  fi
  return "${status}"
}

require_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "${value}" ]]; then
    die "${option} requires a value"
  fi
}

read_secret_file() {
  local path="$1"
  if [[ ! -r "${path}" ]]; then
    die "cannot read secret file: ${path}"
  fi
  tr -d '\r\n' < "${path}"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --nodes)
        require_value "$1" "${2:-}"
        COLOSSUS_NODE_COUNT="$2"
        shift 2
        ;;
      --resource-gpu-count)
        require_value "$1" "${2:-}"
        COLOSSUS_RESOURCE_GPU_COUNT="$2"
        shift 2
        ;;
      --exclude-resource)
        require_value "$1" "${2:-}"
        COLOSSUS_EXCLUDED_RESOURCE_NAMES+=("$2")
        shift 2
        ;;
      --best-effort-hosts)
        BEST_EFFORT_HOSTS="1"
        shift
        ;;
      --gpu-profile)
        require_value "$1" "${2:-}"
        GPU_PROFILE="$2"
        shift 2
        ;;
      --existing-lease-id)
        require_value "$1" "${2:-}"
        COLOSSUS_EXISTING_LEASE_ID="$2"
        shift 2
        ;;
      --join-ray-address)
        require_value "$1" "${2:-}"
        JOIN_RAY_ADDRESS="$2"
        shift 2
        ;;
      --pool-id)
        require_value "$1" "${2:-}"
        COLOSSUS_POOL_ID="$2"
        shift 2
        ;;
      --lease-duration)
        require_value "$1" "${2:-}"
        COLOSSUS_LEASE_DURATION="$2"
        shift 2
        ;;
      --lease-justification)
        require_value "$1" "${2:-}"
        COLOSSUS_LEASE_JUSTIFICATION="$2"
        shift 2
        ;;
      --colossus-os)
        require_value "$1" "${2:-}"
        COLOSSUS_OS="$2"
        shift 2
        ;;
      --colossus-provision-mode)
        require_value "$1" "${2:-}"
        COLOSSUS_PROVISION_MODE="$2"
        shift 2
        ;;
      --lease-ready-timeout)
        require_value "$1" "${2:-}"
        COLOSSUS_LEASE_READY_TIMEOUT="$2"
        shift 2
        ;;
      --lease-poll-interval)
        require_value "$1" "${2:-}"
        COLOSSUS_LEASE_POLL_INTERVAL="$2"
        shift 2
        ;;
      --keep-leases-on-failure)
        KEEP_COLOSSUS_LEASES_ON_FAILURE="1"
        shift
        ;;
      --image)
        require_value "$1" "${2:-}"
        IMAGE="$2"
        shift 2
        ;;
      --gpu-test-image)
        require_value "$1" "${2:-}"
        GPU_TEST_IMAGE="$2"
        shift 2
        ;;
      --artifact-image)
        require_value "$1" "${2:-}"
        ARTIFACT_IMAGE="$2"
        shift 2
        ;;
      --ssl-cert)
        require_value "$1" "${2:-}"
        SSL_CERT="$2"
        shift 2
        ;;
      --ssl-key)
        require_value "$1" "${2:-}"
        SSL_KEY="$2"
        shift 2
        ;;
      --ssl-cert-local)
        require_value "$1" "${2:-}"
        LOCAL_SSL_CERT="$2"
        shift 2
        ;;
      --ssl-key-local)
        require_value "$1" "${2:-}"
        LOCAL_SSL_KEY="$2"
        shift 2
        ;;
      --head-domain)
        require_value "$1" "${2:-}"
        HEAD_DOMAIN="$2"
        HEAD_DOMAIN_EXPLICIT="1"
        shift 2
        ;;
      --ray-auth-user)
        require_value "$1" "${2:-}"
        RAY_AUTH_USER="$2"
        shift 2
        ;;
      --ray-auth-password-file)
        require_value "$1" "${2:-}"
        RAY_AUTH_PASSWORD="$(read_secret_file "$2")"
        RAY_AUTH_PASSWORD_SOURCE="file"
        shift 2
        ;;
      --ray-auth-password-env)
        require_value "$1" "${2:-}"
        RAY_AUTH_PASSWORD_ENV="$2"
        RAY_AUTH_PASSWORD_SOURCE="env"
        shift 2
        ;;
      --ray-auth-password-stdin)
        RAY_AUTH_PASSWORD_SOURCE="stdin"
        shift
        ;;
      --nvcr-token-file)
        require_value "$1" "${2:-}"
        NVCR_TOKEN="$(read_secret_file "$2")"
        NVCR_TOKEN_SOURCE="file"
        shift 2
        ;;
      --nvcr-token-env)
        require_value "$1" "${2:-}"
        NVCR_TOKEN_ENV="$2"
        NVCR_TOKEN_SOURCE="env"
        shift 2
        ;;
      --nvcr-token-stdin)
        NVCR_TOKEN_SOURCE="stdin"
        shift
        ;;
      --groot-data-root)
        require_value "$1" "${2:-}"
        GROOT_DATA_ROOT="$2"
        shift 2
        ;;
      --model-deps-uri)
        require_value "$1" "${2:-}"
        MODEL_DEPS_URI="$2"
        shift 2
        ;;
      --post-training-checkpoint-uri)
        require_value "$1" "${2:-}"
        POST_TRAINING_CHECKPOINT_URI="$2"
        shift 2
        ;;
      --post-training-checkpoint-destination)
        require_value "$1" "${2:-}"
        POST_TRAINING_CHECKPOINT_DESTINATION="$2"
        shift 2
        ;;
      --dataset-archive-uri)
        require_value "$1" "${2:-}"
        DATASET_ARCHIVE_URI="$2"
        shift 2
        ;;
      --dataset-archive-sha256)
        require_value "$1" "${2:-}"
        DATASET_ARCHIVE_SHA256="$2"
        shift 2
        ;;
      --robocasa-runtime-archive-uri)
        require_value "$1" "${2:-}"
        ROBOCASA_RUNTIME_ARCHIVE_URI="$2"
        shift 2
        ;;
      --robocasa-runtime-archive-sha256)
        require_value "$1" "${2:-}"
        ROBOCASA_RUNTIME_ARCHIVE_SHA256="$2"
        shift 2
        ;;
      --artifact-max-mib-per-second)
        require_value "$1" "${2:-}"
        ARTIFACT_MAX_MIB_PER_SECOND="$2"
        shift 2
        ;;
      --bucket-access-key-id-env)
        require_value "$1" "${2:-}"
        BUCKET_ACCESS_KEY_ID_ENV="$2"
        shift 2
        ;;
      --bucket-secret-access-key-env)
        require_value "$1" "${2:-}"
        BUCKET_SECRET_ACCESS_KEY_ENV="$2"
        shift 2
        ;;
      --bucket-endpoint-url-env)
        require_value "$1" "${2:-}"
        BUCKET_ENDPOINT_URL_ENV="$2"
        shift 2
        ;;
      --bucket-region-env)
        require_value "$1" "${2:-}"
        BUCKET_REGION_ENV="$2"
        shift 2
        ;;
      --skip-post-training-artifacts)
        SYNC_POST_TRAINING_ARTIFACTS="0"
        shift
        ;;
      --ssh-connect-timeout)
        require_value "$1" "${2:-}"
        SSH_CONNECT_TIMEOUT="$2"
        shift 2
        ;;
      --reboot-wait-timeout)
        require_value "$1" "${2:-}"
        REBOOT_WAIT_TIMEOUT="$2"
        shift 2
        ;;
      --reboot-poll-interval)
        require_value "$1" "${2:-}"
        REBOOT_POLL_INTERVAL="$2"
        shift 2
        ;;
      --host-setup-concurrency)
        require_value "$1" "${2:-}"
        HOST_SETUP_CONCURRENCY="$2"
        shift 2
        ;;
      --data-interface)
        require_value "$1" "${2:-}"
        DATA_INTERFACE="$2"
        shift 2
        ;;
      --prefer-ethernet-interface)
        require_value "$1" "${2:-}"
        PREFERRED_ETHERNET_INTERFACE="$2"
        shift 2
        ;;
      --static-data-ip-prefix)
        require_value "$1" "${2:-}"
        STATIC_DATA_IP_PREFIX="$2"
        shift 2
        ;;
      --nccl-socket-ifname)
        require_value "$1" "${2:-}"
        NCCL_SOCKET_IFNAME="$2"
        shift 2
        ;;
      --gloo-socket-ifname)
        require_value "$1" "${2:-}"
        GLOO_SOCKET_IFNAME="$2"
        shift 2
        ;;
      --nccl-ib-hca)
        require_value "$1" "${2:-}"
        NCCL_IB_HCA="$2"
        shift 2
        ;;
      --nccl-ib-gid-index)
        require_value "$1" "${2:-}"
        NCCL_IB_GID_INDEX="$2"
        shift 2
        ;;
      --require-rdma)
        REQUIRE_RDMA="1"
        shift
        ;;
      --allow-socket-transport)
        REQUIRE_RDMA="0"
        shift
        ;;
      --ethernet-only)
        ETHERNET_ONLY="1"
        REQUIRE_RDMA="0"
        shift
        ;;
      --transport-validation-command)
        require_value "$1" "${2:-}"
        TRANSPORT_VALIDATION_COMMAND="$2"
        shift 2
        ;;
      --transport-validation-image)
        require_value "$1" "${2:-}"
        TRANSPORT_VALIDATION_IMAGE="$2"
        shift 2
        ;;
      --transport-validation-timeout)
        require_value "$1" "${2:-}"
        TRANSPORT_VALIDATION_TIMEOUT="$2"
        shift 2
        ;;
      --dry-run)
        DRY_RUN="1"
        shift
        ;;
      --state-output)
        require_value "$1" "${2:-}"
        STATE_OUTPUT="$2"
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
}

require_cmd() {
  local missing=()
  local cmd
  for cmd in "$@"; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
      missing+=("${cmd}")
    fi
  done
  if [[ "${#missing[@]}" -gt 0 ]]; then
    die "missing required local command(s): ${missing[*]}"
  fi
}

shell_quote() {
  local value="$1"
  printf "'%s'" "${value//\'/\'\\\'\'}"
}

join_quoted() {
  local value
  local out=""
  for value in "$@"; do
    out+=" $(shell_quote "${value}")"
  done
  printf '%s' "${out}"
}

validate_args() {
  [[ -r "${RAY_COMPAT_SCRIPT}" ]] \
    || die "missing Colossus Ray Python compatibility shim: ${RAY_COMPAT_SCRIPT}"
  case "${COLOSSUS_NODE_COUNT}" in
    ''|*[!0-9]*) die "--nodes must be a positive integer" ;;
  esac
  (( COLOSSUS_NODE_COUNT > 0 )) || die "--nodes must be greater than 0"
  if [[ -n "${COLOSSUS_RESOURCE_GPU_COUNT}" ]]; then
    case "${COLOSSUS_RESOURCE_GPU_COUNT}" in
      *[!0-9]*) die "--resource-gpu-count must be a positive integer" ;;
    esac
    (( COLOSSUS_RESOURCE_GPU_COUNT > 0 )) \
      || die "--resource-gpu-count must be greater than 0"
  fi
  case "${BEST_EFFORT_HOSTS}" in
    0|1) ;;
    *) die "BEST_EFFORT_HOSTS must be 0 or 1" ;;
  esac
  case "${GPU_PROFILE}" in
    gb300|gb300-rtx-pro-2000) ;;
    *) die "--gpu-profile must be gb300 or gb300-rtx-pro-2000" ;;
  esac
  local excluded_resource
  for excluded_resource in "${COLOSSUS_EXCLUDED_RESOURCE_NAMES[@]}"; do
    [[ "${excluded_resource}" =~ ^[A-Za-z0-9._-]+$ ]] \
      || die "--exclude-resource contains invalid characters: ${excluded_resource}"
  done
  if [[ -n "${COLOSSUS_EXISTING_LEASE_ID}" ]]; then
    [[ "${COLOSSUS_EXISTING_LEASE_ID}" =~ ^[A-Za-z0-9-]+$ ]] \
      || die "--existing-lease-id contains invalid characters"
    (( COLOSSUS_NODE_COUNT == 1 )) \
      || die "--existing-lease-id requires --nodes 1"
  fi
  if [[ -n "${JOIN_RAY_ADDRESS}" ]]; then
    if [[ "${JOIN_RAY_ADDRESS}" == *://* || "${JOIN_RAY_ADDRESS}" == *:*:* ]]; then
      die "--join-ray-address must be an internal IPv4 address with an optional port"
    fi
    JOIN_RAY_HEAD_IP="${JOIN_RAY_ADDRESS%%:*}"
    if [[ "${JOIN_RAY_ADDRESS}" == *:* ]]; then
      JOIN_RAY_HEAD_PORT="${JOIN_RAY_ADDRESS##*:}"
    else
      JOIN_RAY_HEAD_PORT="${RAY_PORT}"
    fi
    is_cluster_ip "${JOIN_RAY_HEAD_IP}" \
      || die "--join-ray-address must use a non-loopback cluster IPv4 address"
    case "${JOIN_RAY_HEAD_PORT}" in
      ''|*[!0-9]*) die "--join-ray-address port must be an integer" ;;
    esac
    (( JOIN_RAY_HEAD_PORT > 0 && JOIN_RAY_HEAD_PORT <= 65535 )) \
      || die "--join-ray-address port must be between 1 and 65535"
    JOIN_RAY_ADDRESS="${JOIN_RAY_HEAD_IP}:${JOIN_RAY_HEAD_PORT}"
  fi
  [[ -n "${COLOSSUS_POOL_ID}" ]] || die "--pool-id is required"
  case "${COLOSSUS_POOL_ID}" in
    *$'\n'*|*$'\r'*) die "--pool-id cannot contain newlines" ;;
  esac
  [[ "${COLOSSUS_LEASE_DURATION}" =~ ^[1-9][0-9]*([mhd])?$ ]] \
    || die "--lease-duration must look like 120, 120m, 4h, or 7d"
  [[ -n "${COLOSSUS_LEASE_JUSTIFICATION}" ]] || die "--lease-justification is required"
  case "${COLOSSUS_LEASE_JUSTIFICATION}" in
    *$'\n'*|*$'\r'*) die "--lease-justification cannot contain newlines" ;;
  esac
  [[ -n "${COLOSSUS_OS}" ]] || die "--colossus-os is required"
  case "${COLOSSUS_PROVISION_MODE}" in
    PRE_PROVISION|CLEAN|REUSE_SPECIFIED_OS|NOT_APPLICABLE|REUSE_ANY_OS) ;;
    *) die "--colossus-provision-mode is invalid: ${COLOSSUS_PROVISION_MODE}" ;;
  esac
  case "${COLOSSUS_LEASE_READY_TIMEOUT}" in
    ''|*[!0-9]*) die "--lease-ready-timeout must be a positive integer" ;;
  esac
  case "${COLOSSUS_LEASE_POLL_INTERVAL}" in
    ''|*[!0-9]*) die "--lease-poll-interval must be a positive integer" ;;
  esac
  (( COLOSSUS_LEASE_READY_TIMEOUT > 0 )) || die "--lease-ready-timeout must be greater than 0"
  (( COLOSSUS_LEASE_POLL_INTERVAL > 0 )) || die "--lease-poll-interval must be greater than 0"
  case "${KEEP_COLOSSUS_LEASES_ON_FAILURE}" in
    0|1) ;;
    *) die "KEEP_COLOSSUS_LEASES_ON_FAILURE must be 0 or 1" ;;
  esac
  [[ -x "${HOST_SETUP_SCRIPT}" ]] || die "missing executable host setup script: ${HOST_SETUP_SCRIPT}"
  [[ -r "${PODMAN_WRAPPER}" ]] || die "missing Podman wrapper: ${PODMAN_WRAPPER}"
  [[ -r "${PRUNE_SCRIPT}" ]] || die "missing cleanup script: ${PRUNE_SCRIPT}"
  [[ -n "${IMAGE}" ]] || die "--image must be non-empty"
  [[ -n "${GPU_TEST_IMAGE}" ]] || die "--gpu-test-image must be non-empty"
  [[ -n "${ARTIFACT_IMAGE}" ]] || die "--artifact-image must be non-empty"
  if [[ -z "${JOIN_RAY_ADDRESS}" ]]; then
    [[ -n "${SSL_CERT}" ]] || die "--ssl-cert is required"
    [[ -n "${SSL_KEY}" ]] || die "--ssl-key is required"
    [[ -n "${RAY_AUTH_USER}" ]] || die "--ray-auth-user is required"

    case "${RAY_AUTH_USER}" in
      *:*|*$'\n'*|*$'\r'*) die "--ray-auth-user cannot contain ':' or newlines" ;;
    esac
    case "${SSL_CERT}${SSL_KEY}" in
      *[[:space:]]*) die "--ssl-cert and --ssl-key paths cannot contain whitespace" ;;
    esac
    if [[ -n "${LOCAL_SSL_CERT}" || -n "${LOCAL_SSL_KEY}" ]]; then
      [[ -r "${LOCAL_SSL_CERT}" ]] || die "cannot read local TLS certificate: ${LOCAL_SSL_CERT}"
      [[ -r "${LOCAL_SSL_KEY}" ]] || die "cannot read local TLS private key: ${LOCAL_SSL_KEY}"
    fi
  fi
  case "${GROOT_DATA_ROOT}" in
    ""|*[[:space:]]*) die "--groot-data-root must be non-empty and cannot contain whitespace" ;;
  esac
  case "${POST_TRAINING_CHECKPOINT_DESTINATION}" in
    ""|/*|*..*|*[!A-Za-z0-9._/-]*)
      die "--post-training-checkpoint-destination must be a safe relative path"
      ;;
  esac
  case "${SYNC_POST_TRAINING_ARTIFACTS}" in
    0|1) ;;
    *) die "SYNC_POST_TRAINING_ARTIFACTS must be 0 or 1" ;;
  esac
  case "${ARTIFACT_MAX_MIB_PER_SECOND}" in
    ''|*[!0-9]*) die "ARTIFACT_MAX_MIB_PER_SECOND must be a nonnegative integer" ;;
  esac
  if [[ "${SYNC_POST_TRAINING_ARTIFACTS}" == "1" ]]; then
    [[ "${MODEL_DEPS_URI}" == s3://* ]] || die "--model-deps-uri must be an s3:// URI"
    if [[ -n "${POST_TRAINING_CHECKPOINT_URI}" ]]; then
      [[ "${POST_TRAINING_CHECKPOINT_URI}" == s3://* ]] || die "--post-training-checkpoint-uri must be an s3:// URI"
    elif [[ "${DRY_RUN}" == "0" ]]; then
      die "provide --post-training-checkpoint-uri as an s3:// URI"
    fi
    if [[ -n "${DATASET_ARCHIVE_URI}" ]]; then
      [[ "${DATASET_ARCHIVE_URI}" == s3://* ]] \
        || die "--dataset-archive-uri must be an s3:// URI"
      [[ "${DATASET_ARCHIVE_SHA256}" =~ ^[[:xdigit:]]{64}$ ]] \
        || die "--dataset-archive-sha256 must contain 64 hexadecimal characters"
    elif [[ -n "${DATASET_ARCHIVE_SHA256}" ]]; then
      die "--dataset-archive-sha256 requires --dataset-archive-uri"
    fi
    if [[ -n "${ROBOCASA_RUNTIME_ARCHIVE_URI}" ]]; then
      [[ "${ROBOCASA_RUNTIME_ARCHIVE_URI}" == s3://* ]] \
        || die "--robocasa-runtime-archive-uri must be an s3:// URI"
      [[ "${ROBOCASA_RUNTIME_ARCHIVE_SHA256}" =~ ^[[:xdigit:]]{64}$ ]] \
        || die "--robocasa-runtime-archive-sha256 must contain 64 hexadecimal characters"
    elif [[ -n "${ROBOCASA_RUNTIME_ARCHIVE_SHA256}" ]]; then
      die "--robocasa-runtime-archive-sha256 requires --robocasa-runtime-archive-uri"
    fi
  fi
  case "${SSH_CONNECT_TIMEOUT}" in
    ''|*[!0-9]*) die "--ssh-connect-timeout must be a positive integer" ;;
  esac
  case "${REBOOT_WAIT_TIMEOUT}" in
    ''|*[!0-9]*) die "--reboot-wait-timeout must be a positive integer" ;;
  esac
  case "${REBOOT_POLL_INTERVAL}" in
    ''|*[!0-9]*) die "--reboot-poll-interval must be a positive integer" ;;
  esac
  case "${HOST_SETUP_CONCURRENCY}" in
    ''|*[!0-9]*) die "--host-setup-concurrency must be a positive integer" ;;
  esac
  case "${TRANSPORT_VALIDATION_TIMEOUT}" in
    ''|*[!0-9]*) die "--transport-validation-timeout must be a positive integer" ;;
  esac
  (( SSH_CONNECT_TIMEOUT > 0 )) || die "--ssh-connect-timeout must be greater than 0"
  (( REBOOT_WAIT_TIMEOUT > 0 )) || die "--reboot-wait-timeout must be greater than 0"
  (( REBOOT_POLL_INTERVAL > 0 )) || die "--reboot-poll-interval must be greater than 0"
  (( HOST_SETUP_CONCURRENCY > 0 )) || die "--host-setup-concurrency must be greater than 0"
  (( TRANSPORT_VALIDATION_TIMEOUT > 0 )) || die "--transport-validation-timeout must be greater than 0"

  case "${REQUIRE_RDMA}" in
    0|1) ;;
    *) die "REQUIRE_RDMA must be 0 or 1" ;;
  esac
  case "${ETHERNET_ONLY}" in
    0|1) ;;
    *) die "ETHERNET_ONLY must be 0 or 1" ;;
  esac
  if [[ "${ETHERNET_ONLY}" == "1" && "${REQUIRE_RDMA}" == "1" ]]; then
    die "--ethernet-only cannot be combined with --require-rdma"
  fi
  local interface
  for interface in \
    "${DATA_INTERFACE}" \
    "${PREFERRED_ETHERNET_INTERFACE}" \
    "${NCCL_SOCKET_IFNAME}" \
    "${GLOO_SOCKET_IFNAME}"
  do
    if [[ -n "${interface}" && ! "${interface}" =~ ^[[:alnum:]_.:-]+$ ]]; then
      die "network interface names may contain only letters, digits, '.', '_', ':', and '-'"
    fi
  done
  if [[ -n "${DATA_INTERFACE}" && -n "${PREFERRED_ETHERNET_INTERFACE}" ]]; then
    die "--data-interface and --prefer-ethernet-interface are mutually exclusive"
  fi
  if [[ -n "${STATIC_DATA_IP_PREFIX}" ]]; then
    [[ -n "${DATA_INTERFACE}" || -n "${PREFERRED_ETHERNET_INTERFACE}" ]] \
      || die "--static-data-ip-prefix requires --data-interface or --prefer-ethernet-interface"
    python3 - "${STATIC_DATA_IP_PREFIX}" <<'PY' \
      || die "--static-data-ip-prefix must contain three IPv4 octets"
import ipaddress
import sys

prefix = sys.argv[1]
ipaddress.ip_network(f"{prefix}.0/24")
if len(prefix.split(".")) != 3:
    raise SystemExit(1)
PY
  fi
  if [[ -n "${NCCL_IB_HCA}" && ! "${NCCL_IB_HCA}" =~ ^mlx5_[0-9]+(:[0-9]+)?(,mlx5_[0-9]+(:[0-9]+)?)*$ ]]; then
    die "--nccl-ib-hca must be a comma-separated mlx5_N[:PORT] list"
  fi
  if [[ -n "${NCCL_IB_GID_INDEX}" && ! "${NCCL_IB_GID_INDEX}" =~ ^[0-9]+$ ]]; then
    die "--nccl-ib-gid-index must be a nonnegative integer"
  fi

  if [[ -z "${NCCL_SOCKET_IFNAME}" ]]; then
    NCCL_SOCKET_IFNAME="${DATA_INTERFACE}"
  fi
  if [[ -z "${GLOO_SOCKET_IFNAME}" ]]; then
    GLOO_SOCKET_IFNAME="${DATA_INTERFACE}"
  fi
  if [[ "${REQUIRE_RDMA}" == "1" ]]; then
    [[ -n "${DATA_INTERFACE}" ]] || die "--data-interface is required with --require-rdma"
    [[ -n "${NCCL_SOCKET_IFNAME}" ]] || die "--nccl-socket-ifname is required with --require-rdma"
    [[ -n "${GLOO_SOCKET_IFNAME}" ]] || die "--gloo-socket-ifname is required with --require-rdma"
    [[ -n "${NCCL_IB_HCA}" ]] || die "--nccl-ib-hca is required with --require-rdma"
    [[ -n "${NCCL_IB_GID_INDEX}" ]] || die "--nccl-ib-gid-index is required with --require-rdma"
  elif [[ -n "${NCCL_IB_HCA}" || -n "${NCCL_IB_GID_INDEX}" ]]; then
    die "NCCL RDMA HCA/GID settings require --require-rdma"
  fi

  if [[ "${DRY_RUN}" == "0" ]]; then
    if [[ -z "${JOIN_RAY_ADDRESS}" ]]; then
      [[ -n "${RAY_AUTH_PASSWORD_SOURCE}" ]] || die "provide a Ray auth password via --ray-auth-password-file/env/stdin"
    fi
    [[ -n "${NVCR_TOKEN_SOURCE}" || -n "${NVCR_TOKEN}" ]] || die "provide an NVCR token via --nvcr-token-file/env/stdin or NVCR_TOKEN"
  fi
  case "${STATE_OUTPUT}" in
    *$'\n'*|*$'\r'*) die "--state-output cannot contain newlines" ;;
  esac
}

resolve_secret_sources() {
  if [[ "${RAY_AUTH_PASSWORD_SOURCE}" == "env" ]]; then
    RAY_AUTH_PASSWORD="${!RAY_AUTH_PASSWORD_ENV:-}"
    [[ -n "${RAY_AUTH_PASSWORD}" ]] || die "environment variable ${RAY_AUTH_PASSWORD_ENV} is empty"
  fi

  if [[ "${NVCR_TOKEN_SOURCE}" == "env" ]]; then
    NVCR_TOKEN="${!NVCR_TOKEN_ENV:-}"
    [[ -n "${NVCR_TOKEN}" ]] || die "environment variable ${NVCR_TOKEN_ENV} is empty"
  fi

  if [[ -n "${NVCR_TOKEN}" && -z "${NVCR_TOKEN_SOURCE}" ]]; then
    NVCR_TOKEN_SOURCE="env"
  fi

  if [[ "${RAY_AUTH_PASSWORD_SOURCE}" == "stdin" && "${NVCR_TOKEN_SOURCE}" == "stdin" ]]; then
    IFS= read -r RAY_AUTH_PASSWORD || die "failed to read Ray auth password from stdin"
    IFS= read -r NVCR_TOKEN || die "failed to read NVCR token from stdin"
  elif [[ "${RAY_AUTH_PASSWORD_SOURCE}" == "stdin" ]]; then
    RAY_AUTH_PASSWORD="$(tr -d '\r\n')"
  elif [[ "${NVCR_TOKEN_SOURCE}" == "stdin" ]]; then
    NVCR_TOKEN="$(tr -d '\r\n')"
  fi

  if [[ "${DRY_RUN}" == "0" ]]; then
    if [[ -z "${JOIN_RAY_ADDRESS}" ]]; then
      [[ -n "${RAY_AUTH_PASSWORD}" ]] || die "Ray auth password is empty"
    fi
    [[ -n "${NVCR_TOKEN}" ]] || die "NVCR token is empty"
  fi

  if [[ "${SYNC_POST_TRAINING_ARTIFACTS}" == "1" ]]; then
    BUCKET_ACCESS_KEY_ID_VALUE="${!BUCKET_ACCESS_KEY_ID_ENV:-}"
    BUCKET_SECRET_ACCESS_KEY_VALUE="${!BUCKET_SECRET_ACCESS_KEY_ENV:-}"
    BUCKET_ENDPOINT_URL_VALUE="${!BUCKET_ENDPOINT_URL_ENV:-}"
    BUCKET_REGION_VALUE="${!BUCKET_REGION_ENV:-us-east-1}"
    if [[ "${DRY_RUN}" == "0" ]]; then
      [[ -n "${BUCKET_ACCESS_KEY_ID_VALUE}" ]] || die "environment variable ${BUCKET_ACCESS_KEY_ID_ENV} is empty"
      [[ -n "${BUCKET_SECRET_ACCESS_KEY_VALUE}" ]] || die "environment variable ${BUCKET_SECRET_ACCESS_KEY_ENV} is empty"
      [[ -n "${BUCKET_ENDPOINT_URL_VALUE}" ]] || die "environment variable ${BUCKET_ENDPOINT_URL_ENV} is empty"
      [[ -n "${BUCKET_REGION_VALUE}" ]] || die "environment variable ${BUCKET_REGION_ENV} is empty"
    fi
  fi
}

validate_colossus_pool_access() {
  local access_path="${TMP_DIR}/colossus-pool-access.json"

  if ! colossus_cli bm pool access --pool-name "${COLOSSUS_POOL_ID}" --json > "${access_path}"; then
    die "failed to check reserve permission for Colossus pool ${COLOSSUS_POOL_ID}"
  fi
  if ! python3 - "${access_path}" "${COLOSSUS_POOL_ID}" <<'PY'
import json
import sys

path, pool_id = sys.argv[1:]
with open(path) as f:
    data = json.load(f)
if not isinstance(data, dict) or data.get(pool_id) is not True:
    raise SystemExit(f"current Colossus identity cannot reserve from pool {pool_id}")
PY
  then
    die "Colossus pool access check failed"
  fi
}

write_colossus_lease_list() {
  local status="$1"
  local path="$2"

  if colossus_cli bm lease list --status "${status}" --json > "${path}"; then
    return 0
  fi
  # Colossus CLI 2.3 exits 1 for an empty result while still writing valid [] JSON.
  python3 - "${path}" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
if data != []:
    raise SystemExit("lease query failed with a non-empty response")
PY
}

discover_colossus_candidates() {
  local resources_path="${TMP_DIR}/colossus-resources.json"
  local active_path="${TMP_DIR}/colossus-active-leases.json"
  local pending_path="${TMP_DIR}/colossus-pending-leases.json"
  local assignments
  local excluded_names

  excluded_names="$(IFS=,; printf '%s' "${COLOSSUS_EXCLUDED_RESOURCE_NAMES[*]}")"

  validate_colossus_pool_access
  if ! colossus_cli bm resource list \
    --filters "poolId=${COLOSSUS_POOL_ID}" status=AVAILABLE gpuTag=GB300 \
    --authorized-to-reserve \
    --json > "${resources_path}"; then
    die "failed to list available resources in Colossus pool ${COLOSSUS_POOL_ID}"
  fi
  if ! write_colossus_lease_list ACTIVE "${active_path}"; then
    die "failed to list active Colossus leases"
  fi
  if ! write_colossus_lease_list PENDING "${pending_path}"; then
    die "failed to list pending Colossus leases"
  fi

  if ! assignments="$(
    python3 - \
      "${resources_path}" \
      "${active_path}" \
      "${pending_path}" \
      "${COLOSSUS_POOL_ID}" \
      "${COLOSSUS_OS}" \
      "${COLOSSUS_RESOURCE_GPU_COUNT}" \
      "${excluded_names}" <<'PY'
import json
import re
import shlex
import sys

(
    resources_path,
    active_path,
    pending_path,
    pool_id,
    requested_os,
    gpu_count_text,
    excluded_names_text,
) = sys.argv[1:]
required_gpu_count = int(gpu_count_text) if gpu_count_text else None
excluded_names = set(filter(None, excluded_names_text.split(",")))


def load_records(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise SystemExit(f"{path}: expected a JSON object or array")


busy_names = set()
for path in (active_path, pending_path):
    for record in load_records(path):
        details = record.get("lease_details") or record
        status = str(details.get("status", "")).upper()
        name = details.get("resourceName")
        if not name:
            name = (record.get("entity_details") or {}).get("name")
        if status in {"ACTIVE", "PENDING"} and isinstance(name, str) and name:
            busy_names.add(name)

candidates = {}
for resource in load_records(resources_path):
    if resource.get("poolId") != pool_id:
        continue
    if str(resource.get("status", "")).upper() != "AVAILABLE":
        continue
    if resource.get("gpuTag") != "GB300":
        continue
    if required_gpu_count is not None and resource.get("gpuCount") != required_gpu_count:
        continue
    name = resource.get("name")
    if (
        not isinstance(name, str)
        or re.fullmatch(r"[A-Za-z0-9._-]+", name) is None
        or name in busy_names
        or name in excluded_names
    ):
        continue
    supported_os = resource.get("supportedOs")
    if isinstance(supported_os, list) and supported_os and requested_os not in supported_os:
        continue
    fqdn = resource.get("fqdn")
    if not isinstance(fqdn, str):
        fqdn = ""
    candidates[name] = fqdn

print("COLOSSUS_CANDIDATE_NAMES=()")
print("COLOSSUS_CANDIDATE_FQDNS=()")
for name in sorted(candidates):
    print(f"COLOSSUS_CANDIDATE_NAMES+=({shlex.quote(name)})")
    print(f"COLOSSUS_CANDIDATE_FQDNS+=({shlex.quote(candidates[name])})")
PY
  )"; then
    die "failed to parse Colossus resource discovery"
  fi
  eval "${assignments}"

  if (( ${#COLOSSUS_CANDIDATE_NAMES[@]} < COLOSSUS_NODE_COUNT )); then
    if [[ "${BEST_EFFORT_HOSTS}" == "1" && "${#COLOSSUS_CANDIDATE_NAMES[@]}" -gt 0 ]]; then
      log "pool ${COLOSSUS_POOL_ID} has only ${#COLOSSUS_CANDIDATE_NAMES[@]} compatible host(s); continuing best-effort"
      COLOSSUS_NODE_COUNT="${#COLOSSUS_CANDIDATE_NAMES[@]}"
    else
      die "pool ${COLOSSUS_POOL_ID} has ${#COLOSSUS_CANDIDATE_NAMES[@]} unleased compatible host(s), but --nodes requested ${COLOSSUS_NODE_COUNT}"
    fi
  fi
  log "found ${#COLOSSUS_CANDIDATE_NAMES[@]} unleased compatible host(s) in Colossus pool ${COLOSSUS_POOL_ID}"
}

prepare_dry_run_hosts() {
  HOSTS=()
  USERS=()
  PASSWORDS=()
  DATA_IPS=()
  DATA_INTERFACES=()

  local idx host
  for ((idx = 0; idx < COLOSSUS_NODE_COUNT; idx++)); do
    host="${COLOSSUS_CANDIDATE_FQDNS[$idx]}"
    if [[ -z "${host}" ]]; then
      host="<assigned-after-provisioning>"
    fi
    HOSTS+=("${host}")
    USERS+=("<assigned-by-colossus>")
    PASSWORDS+=("")
    DATA_IPS+=("")
    DATA_INTERFACES+=("")
  done

  if [[ -z "${JOIN_RAY_ADDRESS}" && -z "${HEAD_DOMAIN}" ]]; then
    HEAD_DOMAIN="${HOSTS[0]}"
  fi
}

load_existing_colossus_host() {
  local include_credentials="$1"
  local existing_path="${TMP_DIR}/colossus-existing-lease.json"
  local assignments
  local -a command=(
    colossus_cli bm lease list
    --lease-id "${COLOSSUS_EXISTING_LEASE_ID}"
  )

  if [[ "${include_credentials}" == "1" ]]; then
    command+=(--show-creds)
  fi
  command+=(--json)
  "${command[@]}" > "${existing_path}" \
    || die "could not read existing Colossus lease ${COLOSSUS_EXISTING_LEASE_ID}"
  chmod 0600 "${existing_path}"
  if ! assignments="$(
    python3 - \
      "${existing_path}" \
      "${COLOSSUS_EXISTING_LEASE_ID}" \
      "${include_credentials}" <<'PY'
import json
import re
import shlex
import sys

path, expected_id, include_credentials = sys.argv[1:]
with open(path) as f:
    data = json.load(f)
records = data if isinstance(data, list) else [data]
record = next(
    (
        item
        for item in records
        if isinstance(item, dict)
        and (item.get("lease_details") or item).get("id") == expected_id
    ),
    None,
)
if record is None:
    raise SystemExit("existing lease record is unavailable")
details = record.get("lease_details") or record
entity = record.get("entity_details") or {}
secrets = record.get("lease_secrets") or {}
if str(details.get("status", "")).upper() != "ACTIVE":
    raise SystemExit("existing lease is not active")
name = details.get("resourceName") or entity.get("name")
host = entity.get("fqdn")
if (
    not isinstance(name, str)
    or re.fullmatch(r"[A-Za-z0-9._-]+", name) is None
    or not isinstance(host, str)
    or not host
):
    raise SystemExit("existing lease resource identity is incomplete")
print(f"COLOSSUS_EXISTING_LEASE_NAME={shlex.quote(name)}")
print(f"HOSTS+=({shlex.quote(host)})")
print("DATA_IPS+=('')")
print("DATA_INTERFACES+=('')")
if include_credentials == "1":
    user = secrets.get("local_accountname")
    password = secrets.get("local_accountpassword")
    if not all(isinstance(value, str) and value for value in (user, password)):
        raise SystemExit("existing lease SSH credentials are incomplete")
    print(f"USERS+=({shlex.quote(user)})")
    print(f"PASSWORDS+=({shlex.quote(password)})")
else:
    print("USERS+=('<assigned-by-colossus>')")
    print("PASSWORDS+=('')")
PY
  )"; then
    die "could not load active existing lease ${COLOSSUS_EXISTING_LEASE_ID}"
  fi
  eval "${assignments}"
  if [[ -z "${JOIN_RAY_ADDRESS}" && -z "${HEAD_DOMAIN}" ]]; then
    HEAD_DOMAIN="${HOSTS[0]}"
  fi
  log "reusing Colossus lease ${COLOSSUS_EXISTING_LEASE_ID} (${COLOSSUS_EXISTING_LEASE_NAME})"
}

parse_created_lease_id() {
  local path="$1"
  python3 - "${path}" <<'PY'
import json
import re
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
if isinstance(data, list):
    if len(data) != 1:
        raise SystemExit("expected exactly one created lease")
    data = data[0]
if not isinstance(data, dict):
    raise SystemExit("expected a created lease object")
details = data.get("lease_details") or data
lease_id = details.get("id") or data.get("lease_id")
if not lease_id:
    response = data.get("json_output") or {}
    request_url = response.get("requestUrl") or data.get("lease_url")
    if isinstance(request_url, str):
        lease_id = request_url.rstrip("/").rsplit("/", 1)[-1]
if (
    not isinstance(lease_id, str)
    or re.fullmatch(r"[A-Za-z0-9-]+", lease_id) is None
):
    raise SystemExit("created lease response has no lease id")
print(lease_id)
PY
}

append_colossus_lease_host() {
  local lease_id="$1"
  local expected_name="$2"
  local credentials_path="${TMP_DIR}/colossus-lease-${lease_id}-credentials.json"
  local assignments

  if ! colossus_cli bm lease list \
    --lease-id "${lease_id}" \
    --show-creds \
    --json > "${credentials_path}"; then
    return 1
  fi
  chmod 0600 "${credentials_path}"
  if ! assignments="$(
    python3 - "${credentials_path}" "${lease_id}" "${expected_name}" <<'PY'
import json
import shlex
import sys

path, expected_id, expected_name = sys.argv[1:]
with open(path) as f:
    data = json.load(f)
records = data if isinstance(data, list) else [data]
record = next(
    (
        item
        for item in records
        if isinstance(item, dict)
        and (item.get("lease_details") or item).get("id") == expected_id
    ),
    None,
)
if record is None:
    raise SystemExit("lease record is unavailable")
details = record.get("lease_details") or record
entity = record.get("entity_details") or {}
secrets = record.get("lease_secrets") or {}
if str(details.get("status", "")).upper() != "ACTIVE":
    raise SystemExit("lease is not active")
name = details.get("resourceName") or entity.get("name")
if name != expected_name:
    raise SystemExit(f"lease resource mismatch: expected {expected_name}, got {name}")
host = entity.get("fqdn")
user = secrets.get("local_accountname")
password = secrets.get("local_accountpassword")
if not all(isinstance(value, str) and value for value in (host, user, password)):
    raise SystemExit("active lease credentials are incomplete")
print(f"HOSTS+=({shlex.quote(host)})")
print(f"USERS+=({shlex.quote(user)})")
print(f"PASSWORDS+=({shlex.quote(password)})")
print("DATA_IPS+=('')")
print("DATA_INTERFACES+=('')")
PY
  )"; then
    rm -f "${credentials_path}"
    return 1
  fi
  rm -f "${credentials_path}"
  eval "${assignments}"
}

colossus_lease_status() {
  local lease_id="$1"
  local status_path="${TMP_DIR}/colossus-lease-${lease_id}-status.json"

  if ! colossus_cli bm lease list --lease-id "${lease_id}" --json > "${status_path}"; then
    return 1
  fi
  python3 - "${status_path}" "${lease_id}" <<'PY'
import json
import sys

path, expected_id = sys.argv[1:]
with open(path) as f:
    data = json.load(f)
records = data if isinstance(data, list) else [data]
for record in records:
    if not isinstance(record, dict):
        continue
    details = record.get("lease_details") or record
    if details.get("id") == expected_id:
        print(str(details.get("status", "")).upper())
        break
PY
}

wait_for_colossus_lease() {
  local lease_id="$1"
  local resource_name="$2"
  local deadline="$3"
  local status=""

  while (( SECONDS < deadline )); do
    status="$(colossus_lease_status "${lease_id}" || true)"
    case "${status}" in
      ACTIVE)
        if append_colossus_lease_host "${lease_id}" "${resource_name}"; then
          log "${resource_name}: Colossus lease ${lease_id} is active and credentials are ready"
          return 0
        fi
        log "${resource_name}: lease is active; waiting for SSH credentials"
        ;;
      PENDING|"")
        log "${resource_name}: waiting for Colossus lease ${lease_id} (${status:-status unavailable})"
        ;;
      EXPIRED)
        log "${resource_name}: Colossus lease ${lease_id} expired during provisioning"
        return 1
        ;;
      *)
        log "${resource_name}: waiting for Colossus lease ${lease_id} (status ${status})"
        ;;
    esac
    sleep "${COLOSSUS_LEASE_POLL_INTERVAL}"
  done
  log "${resource_name}: timed out after ${COLOSSUS_LEASE_READY_TIMEOUT}s waiting for Colossus lease ${lease_id}"
  return 1
}

reserve_colossus_hosts() {
  local idx name lease_id status
  local create_path error_path
  local deadline
  local -a credentialed_lease_ids=() credentialed_lease_names=()
  local -a pending_lease_ids=() pending_lease_names=()
  local -a next_pending_lease_ids=() next_pending_lease_names=()
  local -a unready_lease_ids=() unready_lease_names=()

  HOSTS=()
  USERS=()
  PASSWORDS=()
  DATA_IPS=()
  DATA_INTERFACES=()
  COLOSSUS_LEASE_IDS=()
  COLOSSUS_LEASE_NAMES=()

  for idx in "${!COLOSSUS_CANDIDATE_NAMES[@]}"; do
    if (( ${#COLOSSUS_LEASE_IDS[@]} >= COLOSSUS_NODE_COUNT )); then
      break
    fi
    name="${COLOSSUS_CANDIDATE_NAMES[$idx]}"
    create_path="${TMP_DIR}/colossus-create-${name}.json"
    error_path="${TMP_DIR}/colossus-create-${name}.stderr"
    log "${name}: requesting ${COLOSSUS_LEASE_DURATION} lease from ${COLOSSUS_POOL_ID}"
    if ! colossus_cli bm lease create \
      --duration "${COLOSSUS_LEASE_DURATION}" \
      --lease-justification "${COLOSSUS_LEASE_JUSTIFICATION}; ${name}" \
      --provision-mode "${COLOSSUS_PROVISION_MODE}" \
      --os "${COLOSSUS_OS}" \
      --filters "name=${name}" "poolId=${COLOSSUS_POOL_ID}" status=AVAILABLE \
      --json > "${create_path}" 2> "${error_path}"; then
      log "${name}: reservation lost or failed; trying another pool host"
      if [[ -s "${error_path}" ]]; then
        tail -n 3 "${error_path}" | sed "s/^/[setup_ray_cluster] ${name}: /"
      fi
      continue
    fi
    if ! lease_id="$(parse_created_lease_id "${create_path}")"; then
      die "Colossus reported a successful reservation for ${name} without a lease id; inspect the service user's leases"
    fi
    COLOSSUS_LEASE_IDS+=("${lease_id}")
    COLOSSUS_LEASE_NAMES+=("${name}")
    log "${name}: created Colossus lease ${lease_id}"
  done

  if (( ${#COLOSSUS_LEASE_IDS[@]} < COLOSSUS_NODE_COUNT )); then
    if [[ "${BEST_EFFORT_HOSTS}" == "1" && "${#COLOSSUS_LEASE_IDS[@]}" -gt 0 ]]; then
      log "created ${#COLOSSUS_LEASE_IDS[@]} of ${COLOSSUS_NODE_COUNT} requested leases; continuing best-effort"
      COLOSSUS_NODE_COUNT="${#COLOSSUS_LEASE_IDS[@]}"
    else
      die "created ${#COLOSSUS_LEASE_IDS[@]} of ${COLOSSUS_NODE_COUNT} requested Colossus leases"
    fi
  fi

  # Poll the whole reservation batch round-robin against one shared deadline.
  # Waiting on a broken first station must not prevent later stations whose
  # credentials are already ready from joining a best-effort cluster.
  pending_lease_ids=("${COLOSSUS_LEASE_IDS[@]}")
  pending_lease_names=("${COLOSSUS_LEASE_NAMES[@]}")
  deadline=$((SECONDS + COLOSSUS_LEASE_READY_TIMEOUT))
  while (( ${#pending_lease_ids[@]} > 0 && SECONDS < deadline )); do
    next_pending_lease_ids=()
    next_pending_lease_names=()
    for idx in "${!pending_lease_ids[@]}"; do
      lease_id="${pending_lease_ids[$idx]}"
      name="${pending_lease_names[$idx]}"
      status="$(colossus_lease_status "${lease_id}" || true)"
      if [[ "${status}" == "ACTIVE" ]] \
          && append_colossus_lease_host "${lease_id}" "${name}"; then
        credentialed_lease_ids+=("${lease_id}")
        credentialed_lease_names+=("${name}")
        log "${name}: Colossus lease ${lease_id} is active and credentials are ready"
        continue
      fi
      if [[ "${status}" == "EXPIRED" ]]; then
        log "${name}: Colossus lease ${lease_id} expired during provisioning"
        unready_lease_ids+=("${lease_id}")
        unready_lease_names+=("${name}")
        continue
      fi
      log "${name}: waiting for Colossus lease ${lease_id} (${status:-status unavailable}; credentials pending)"
      next_pending_lease_ids+=("${lease_id}")
      next_pending_lease_names+=("${name}")
    done
    pending_lease_ids=("${next_pending_lease_ids[@]}")
    pending_lease_names=("${next_pending_lease_names[@]}")
    if (( ${#pending_lease_ids[@]} > 0 && SECONDS < deadline )); then
      sleep "${COLOSSUS_LEASE_POLL_INTERVAL}"
    fi
  done
  unready_lease_ids+=("${pending_lease_ids[@]}")
  unready_lease_names+=("${pending_lease_names[@]}")

  if (( ${#unready_lease_ids[@]} > 0 )) && [[ "${BEST_EFFORT_HOSTS}" != "1" ]]; then
    die "Colossus lease ${unready_lease_ids[0]} for ${unready_lease_names[0]} did not become credential-ready"
  fi
  for idx in "${!unready_lease_ids[@]}"; do
    lease_id="${unready_lease_ids[$idx]}"
    name="${unready_lease_names[$idx]}"
    log "${name}: ignoring lease that did not become credential-ready"
    if colossus_cli bm lease delete \
      --lease-id "${lease_id}" \
      --force \
      --release-mode CLEAN \
      --json >/dev/null; then
      log "${name}: released unready Colossus lease ${lease_id}"
    else
      log "${name}: WARNING: failed to release unready Colossus lease ${lease_id}"
    fi
  done
  COLOSSUS_LEASE_IDS=("${credentialed_lease_ids[@]}")
  COLOSSUS_LEASE_NAMES=("${credentialed_lease_names[@]}")
  COLOSSUS_NODE_COUNT="${#COLOSSUS_LEASE_IDS[@]}"
  (( COLOSSUS_NODE_COUNT > 0 )) \
    || die "no reserved host became credential-ready"
  if [[ -z "${JOIN_RAY_ADDRESS}" && -z "${HEAD_DOMAIN}" ]]; then
    HEAD_DOMAIN="${HOSTS[0]}"
  fi
}

release_colossus_leases() {
  local idx lease_id name

  log "releasing ${#COLOSSUS_LEASE_IDS[@]} Colossus lease(s) created by this failed setup"
  for idx in "${!COLOSSUS_LEASE_IDS[@]}"; do
    lease_id="${COLOSSUS_LEASE_IDS[$idx]}"
    name="${COLOSSUS_LEASE_NAMES[$idx]:-unknown}"
    if colossus_cli bm lease delete \
      --lease-id "${lease_id}" \
      --force \
      --release-mode CLEAN \
      --json >/dev/null; then
      log "${name}: released Colossus lease ${lease_id}"
    else
      log "${name}: WARNING: failed to release Colossus lease ${lease_id}"
    fi
  done
  COLOSSUS_LEASE_IDS=()
  COLOSSUS_LEASE_NAMES=()
}

print_dry_run() {
  log "dry-run only; no Colossus lease will be created and no host will be changed"
  if [[ -n "${COLOSSUS_EXISTING_LEASE_ID}" ]]; then
    log "host source: existing Colossus lease ${COLOSSUS_EXISTING_LEASE_ID}"
  else
    log "Colossus pool: ${COLOSSUS_POOL_ID}"
  fi
  log "requested nodes: ${COLOSSUS_NODE_COUNT}"
  log "required physical GPU count: ${COLOSSUS_RESOURCE_GPU_COUNT:-<any>}"
  log "best-effort hosts: ${BEST_EFFORT_HOSTS}"
  log "GPU profile: ${GPU_PROFILE}"
  if [[ "${#COLOSSUS_EXCLUDED_RESOURCE_NAMES[@]}" -gt 0 ]]; then
    log "excluded resources: ${COLOSSUS_EXCLUDED_RESOURCE_NAMES[*]}"
  fi
  log "lease duration: ${COLOSSUS_LEASE_DURATION}"
  log "lease OS: ${COLOSSUS_OS}"
  log "lease provision mode: ${COLOSSUS_PROVISION_MODE}"
  log "Ray control image: ${IMAGE}"
  log "GPU test image: ${GPU_TEST_IMAGE}"
  log "artifact image: ${ARTIFACT_IMAGE}"
  if [[ -n "${JOIN_RAY_ADDRESS}" ]]; then
    log "cluster mode: join existing"
    log "existing Ray address: ${JOIN_RAY_ADDRESS}"
    log "existing Ray head and nginx configuration: unchanged"
  else
    log "cluster mode: create new"
    log "head domain: ${HEAD_DOMAIN}"
    log "ssl cert on head: ${SSL_CERT}"
    log "ssl key on head: ${SSL_KEY}"
    log "local TLS material: ${LOCAL_SSL_CERT:+configured}"
    log "nginx endpoint: https://${HEAD_DOMAIN}/ray/"
    log "ray auth user: ${RAY_AUTH_USER}"
  fi
  log "host setup concurrency: ${HOST_SETUP_CONCURRENCY}"
  log "GPU isolation/profile: ${GPU_PROFILE}"
  log "data interface: ${DATA_INTERFACE:-<automatic>}"
  log "preferred non-mlx5 Ethernet interface: ${PREFERRED_ETHERNET_INTERFACE:-<disabled>}"
  log "static data IP prefix: ${STATIC_DATA_IP_PREFIX:-<disabled>}"
  log "NCCL socket interface: ${NCCL_SOCKET_IFNAME:-<automatic>}"
  log "Gloo socket interface: ${GLOO_SOCKET_IFNAME:-<automatic>}"
  log "RDMA required: ${REQUIRE_RDMA}"
  log "Ethernet only: ${ETHERNET_ONLY}"
  if [[ "${REQUIRE_RDMA}" == "1" ]]; then
    log "NCCL IB HCA: ${NCCL_IB_HCA}"
    log "NCCL IB GID index: ${NCCL_IB_GID_INDEX}"
  fi
  if [[ -n "${TRANSPORT_VALIDATION_COMMAND}" ]]; then
    [[ -n "${TRANSPORT_VALIDATION_IMAGE:-${GPU_TEST_IMAGE}}" ]] \
      || die "--transport-validation-image must be non-empty"
    log "transport validation command: configured (timeout ${TRANSPORT_VALIDATION_TIMEOUT}s)"
    log "transport validation image: ${TRANSPORT_VALIDATION_IMAGE:-${GPU_TEST_IMAGE}}"
  else
    log "transport validation command: disabled"
  fi
  log "groot data root: ${GROOT_DATA_ROOT}"
  if [[ "${SYNC_POST_TRAINING_ARTIFACTS}" == "1" ]]; then
    log "post-training artifacts: enabled"
    log "model deps source: ${MODEL_DEPS_URI}"
    log "post-training checkpoint source: ${POST_TRAINING_CHECKPOINT_URI:-<required for real run>}"
    log "post-training checkpoint destination: ${POST_TRAINING_CHECKPOINT_DESTINATION}"
    log "dataset archive: ${DATASET_ARCHIVE_URI:-<disabled>}"
    log "RoboCasa runtime archive: ${ROBOCASA_RUNTIME_ARCHIVE_URI:-<disabled>}"
    log "bucket credential env vars: ${BUCKET_ACCESS_KEY_ID_ENV}, ${BUCKET_SECRET_ACCESS_KEY_ENV}, ${BUCKET_ENDPOINT_URL_ENV}, ${BUCKET_REGION_ENV}"
  else
    log "post-training artifacts: disabled"
  fi
  log "managed containers: ${HEAD_CONTAINER}, ${WORKER_CONTAINER}"
  local i role
  for i in "${!HOSTS[@]}"; do
    role="worker"
    if [[ -z "${JOIN_RAY_ADDRESS}" && "${i}" == "0" ]]; then
      role="head"
    fi
    if [[ -n "${COLOSSUS_EXISTING_LEASE_ID}" ]]; then
      log "existing lease host[$i]: ${role} ${COLOSSUS_EXISTING_LEASE_NAME} (${HOSTS[$i]}) data_ip=<automatic>"
    else
      log "reservation candidate[$i]: ${role} ${COLOSSUS_CANDIDATE_NAMES[$i]} (${HOSTS[$i]}) data_ip=<automatic>"
    fi
  done
}

ssh_opts() {
  printf '%s\n' \
    "-o" "StrictHostKeyChecking=accept-new" \
    "-o" "UserKnownHostsFile=${TMP_DIR}/ssh-known-hosts" \
    "-o" "ConnectTimeout=${SSH_CONNECT_TIMEOUT}" \
    "-o" "ServerAliveInterval=30" \
    "-o" "ServerAliveCountMax=4"
}

remote_ssh() {
  local host="$1"
  local user="$2"
  local password="$3"
  local -a opts
  shift 3
  mapfile -t opts < <(ssh_opts)
  SSHPASS="${password}" sshpass -e ssh "${opts[@]}" "${user}@${host}" "$@"
}

remote_scp_to() {
  local host="$1"
  local user="$2"
  local password="$3"
  local local_path="$4"
  local remote_path="$5"
  local -a opts
  mapfile -t opts < <(ssh_opts)
  SSHPASS="${password}" sshpass -e scp "${opts[@]}" "${local_path}" "${user}@${host}:${remote_path}"
}

remote_sudo_bash_lc() {
  local host="$1"
  local user="$2"
  local password="$3"
  local command="$4"
  printf '%s\n' "${password}" \
    | remote_ssh "${host}" "${user}" "${password}" "sudo -S -p '' bash -lc $(shell_quote "${command}")"
}

cleanup_remote_tls_staging() {
  if [[ "${REMOTE_TLS_STAGING_ACTIVE}" != "1" ]]; then
    return
  fi

  if [[ "${#HOSTS[@]}" -gt 0 && "${#USERS[@]}" -gt 0 && "${#PASSWORDS[@]}" -gt 0 ]]; then
    remote_ssh "${HOSTS[0]}" "${USERS[0]}" "${PASSWORDS[0]}" \
      "rm -f -- $(shell_quote "${REMOTE_TLS_CERT_STAGING}") $(shell_quote "${REMOTE_TLS_KEY_STAGING}")" \
      >/dev/null 2>&1 || true
  fi
  REMOTE_TLS_STAGING_ACTIVE="0"
  REMOTE_TLS_CERT_STAGING=""
  REMOTE_TLS_KEY_STAGING=""
}

remote_sudo_capture() {
  remote_sudo_bash_lc "$@"
}

is_cluster_ip() {
  local ip="$1"
  [[ "${ip}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || return 1

  case "${ip}" in
    0.0.0.0|127.*|169.254.*|172.17.*)
      return 1
      ;;
  esac

  return 0
}

select_cluster_ip() {
  local ip
  while read -r ip; do
    if is_cluster_ip "${ip}"; then
      printf '%s\n' "${ip}"
      return 0
    fi
  done
  return 1
}

remote_cluster_ip() {
  local host="$1"
  local user="$2"
  local password="$3"
  local command="$4"

  remote_sudo_capture "${host}" "${user}" "${password}" "${command}" \
    | tr '[:space:]' '\n' \
    | select_cluster_ip
}

remote_sudo_script() {
  local host="$1"
  local user="$2"
  local password="$3"
  local script_path="$4"
  shift 4
  local quoted_args
  quoted_args="$(join_quoted "$@")"
  printf '%s\n' "${password}" \
    | remote_ssh "${host}" "${user}" "${password}" "sudo -S -p '' bash $(shell_quote "${script_path}")${quoted_args}"
}

remote_boot_id() {
  local host="$1"
  local user="$2"
  local password="$3"
  remote_ssh "${host}" "${user}" "${password}" "cat /proc/sys/kernel/random/boot_id" 2>/dev/null \
    | head -n 1 \
    | tr -d '[:space:]'
}

wait_for_remote_sudo() {
  local host="$1"
  local user="$2"
  local password="$3"
  local deadline="$4"

  while (( SECONDS < deadline )); do
    if remote_sudo_bash_lc "${host}" "${user}" "${password}" "true" >/dev/null 2>&1; then
      return 0
    fi
    sleep "${REBOOT_POLL_INTERVAL}"
  done

  log "${host}: timed out waiting ${REBOOT_WAIT_TIMEOUT}s for sudo after reboot"
  return 1
}

wait_for_reboot() {
  local host="$1"
  local user="$2"
  local password="$3"
  local old_boot_id="$4"
  local deadline=$((SECONDS + REBOOT_WAIT_TIMEOUT))
  local new_boot_id=""

  log "${host}: polling SSH until reboot completes"
  while (( SECONDS < deadline )); do
    if new_boot_id="$(remote_boot_id "${host}" "${user}" "${password}")" \
      && [[ -n "${new_boot_id}" && "${new_boot_id}" != "${old_boot_id}" ]]; then
      log "${host}: SSH returned after reboot"
      wait_for_remote_sudo "${host}" "${user}" "${password}" "${deadline}"
      return
    fi
    sleep "${REBOOT_POLL_INTERVAL}"
  done

  log "${host}: timed out waiting ${REBOOT_WAIT_TIMEOUT}s for reboot"
  return 1
}

reboot_host_and_wait() {
  local host="$1"
  local user="$2"
  local password="$3"
  local old_boot_id

  old_boot_id="$(remote_boot_id "${host}" "${user}" "${password}")" || {
    log "${host}: could not read boot ID before reboot"
    return 1
  }
  [[ -n "${old_boot_id}" ]] || {
    log "${host}: boot ID was empty before reboot"
    return 1
  }

  log "${host}: rebooting to apply NVIDIA driver/module options"
  remote_sudo_bash_lc "${host}" "${user}" "${password}" \
    "nohup sh -c 'sleep 1; systemctl reboot' >/dev/null 2>&1 &" >/dev/null || true

  wait_for_reboot "${host}" "${user}" "${password}" "${old_boot_id}"
}

write_remote_scripts() {
  cat > "${TMP_DIR}/remote_ray_container.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

role="$1"
image="$2"
head_address="$3"
node_ip="$4"
gpu_count="$5"
data_root="$6"
data_interface="$7"
nccl_socket_ifname="$8"
gloo_socket_ifname="$9"
nccl_ib_hca="${10}"
nccl_ib_gid_index="${11}"
require_rdma="${12}"
target_gpu_name="${13}"
gpu_test_image="${14}"
gpu_profile="${15}"
secondary_gpu_name_pattern="${16}"
ethernet_only="${17}"
head_domain="${18}"

head_container="groot-ray-head"
worker_container="groot-ray-worker"
host_ray_tmp="${data_root}/ray"
# Keep Ray's temporary path identical in the control container and on the
# host. Ray propagates this literal path into nested image_uri workers, whose
# Podman bind sources are resolved by the host daemon rather than by the
# control container.
container_ray_tmp="${host_ray_tmp}"
job_root="/data/groot-ol-jobs"
podman_socket="/run/podman/podman.sock"
podman_auth="/root/.config/containers/auth.json"
host_podman_wrapper="/usr/local/libexec/groot-ray/podman"

case "${require_rdma}" in
  0|1) ;;
  *) echo "invalid require_rdma=${require_rdma}" >&2; exit 1 ;;
esac
case "${ethernet_only}" in
  0|1) ;;
  *) echo "invalid ethernet_only=${ethernet_only}" >&2; exit 1 ;;
esac
case "${gpu_profile}" in
  gb300)
    [[ "${gpu_count}" == "1" ]] || {
      echo "expected one exposed GPU for profile ${gpu_profile}, got ${gpu_count}" >&2
      exit 1
    }
    ;;
  gb300-rtx-pro-2000)
    [[ "${gpu_count}" == "2" ]] || {
      echo "expected two exposed GPUs for profile ${gpu_profile}, got ${gpu_count}" >&2
      exit 1
    }
    ;;
  *) echo "invalid gpu_profile=${gpu_profile}" >&2; exit 1 ;;
esac
[[ -n "${target_gpu_name}" ]] || {
  echo "target GPU model must be non-empty" >&2
  exit 1
}

if ! gpu_inventory="$(timeout --kill-after=5s 20s nvidia-smi \
  --query-gpu=uuid,name --format=csv,noheader)"; then
  echo "could not query host GPU UUIDs and model names" >&2
  exit 1
fi
gb300_gpu_uuid="$(printf '%s\n' "${gpu_inventory}" \
  | awk -F',' -v target="${target_gpu_name}" '
        {
          uuid=$1; name=$2
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", uuid)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
          if (name == target) {
            selected=uuid
            count++
          }
        }
        END { if (count == 1 && selected ~ /^GPU-[[:xdigit:]-]+$/) print selected }
      ')"
[[ -n "${gb300_gpu_uuid}" ]] || {
  echo "expected exactly one ${target_gpu_name} GPU in the host inventory" >&2
  exit 1
}
rtx_gpu_uuid=""
if [[ "${gpu_profile}" == "gb300-rtx-pro-2000" ]]; then
  rtx_gpu_uuid="$(printf '%s\n' "${gpu_inventory}" \
    | awk -F',' -v target="${secondary_gpu_name_pattern}" '
          {
            uuid=$1; name=$2
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", uuid)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
            if (index(name, target) != 0) {
              selected=uuid
              count++
            }
          }
          END { if (count == 1 && selected ~ /^GPU-[[:xdigit:]-]+$/) print selected }
        ')"
  [[ -n "${rtx_gpu_uuid}" && "${rtx_gpu_uuid}" != "${gb300_gpu_uuid}" ]] || {
    echo "expected exactly one distinct ${secondary_gpu_name_pattern} GPU" >&2
    exit 1
  }
fi
ray_num_gpus="${gpu_count}"

install -d -m 0755 /etc/cdi "${host_ray_tmp}" "${job_root}" "${data_root}/runtime/python"
install -m 0644 \
  /tmp/groot-ray-cluster/colossus_ray_sitecustomize.py \
  "${data_root}/runtime/python/sitecustomize.py"
nvidia-ctk cdi generate \
  --device-name-strategy=uuid \
  --output=/etc/cdi/nvidia.yaml >/dev/null
# Colossus currently ships Podman 4.9, whose CDI parser predates the 0.7
# additionalGids field emitted by current NVIDIA Container Toolkit releases.
# Rootful job containers do not need those supplemental DRM group IDs.
python3 - /etc/cdi/nvidia.yaml <<'PYTHON'
from pathlib import Path
import os
import sys
import tempfile

path = Path(sys.argv[1])
lines = path.read_text().splitlines(keepends=True)
output: list[str] = []
skip_indent: int | None = None
for line in lines:
    stripped = line.lstrip(" ")
    indent = len(line) - len(stripped)
    if skip_indent is not None:
        if stripped.strip() and indent > skip_indent:
            continue
        skip_indent = None
    if stripped.strip() == "additionalGids:":
        skip_indent = indent
        continue
    if line.startswith("cdiVersion:"):
        line = "cdiVersion: 0.6.0\n"
    output.append(line)

mode = path.stat().st_mode
with tempfile.NamedTemporaryFile(
    mode="w",
    dir=path.parent,
    prefix=f".{path.name}.",
    delete=False,
) as destination:
    destination.writelines(output)
    temporary_path = destination.name
os.chmod(temporary_path, mode)
os.replace(temporary_path, path)
PYTHON
nvidia-ctk cdi list | grep -Fxq "nvidia.com/gpu=${gb300_gpu_uuid}"
if [[ -n "${rtx_gpu_uuid}" ]]; then
  nvidia-ctk cdi list | grep -Fxq "nvidia.com/gpu=${rtx_gpu_uuid}"
fi
systemctl enable --now podman.socket
[[ -S "${podman_socket}" ]] || {
  echo "rootful Podman socket is unavailable: ${podman_socket}" >&2
  exit 1
}
[[ -r "${podman_auth}" ]] || {
  echo "rootful Podman registry auth is unavailable: ${podman_auth}" >&2
  exit 1
}

if [[ -z "${data_interface}" ]]; then
  data_interface="$(ip -o -4 addr show scope global \
    | awk -v node_ip="${node_ip}" '
        {
          address=$4
          sub(/\/.*/, "", address)
          if (address == node_ip) {
            print $2
            exit
          }
        }
      ')"
  [[ -n "${data_interface}" ]] || {
    echo "could not resolve the network interface for Ray node IP ${node_ip}" >&2
    exit 1
  }
fi
if [[ -z "${nccl_socket_ifname}" ]]; then
  nccl_socket_ifname="${data_interface}"
fi
if [[ -z "${gloo_socket_ifname}" ]]; then
  gloo_socket_ifname="${data_interface}"
fi
ip link show dev "${data_interface}" >/dev/null
ip -o -4 addr show dev "${data_interface}" scope global | grep -q '[[:space:]]inet[[:space:]]'
if [[ "${ethernet_only}" == "1" ]]; then
  interface_driver="$(basename "$(readlink -f "/sys/class/net/${data_interface}/device/driver" 2>/dev/null || true)")"
  [[ -n "${interface_driver}" ]] || {
    echo "could not resolve driver for Ethernet interface ${data_interface}" >&2
    exit 1
  }
  [[ "${interface_driver}" != "mlx5_core" ]] || {
    echo "Ethernet-only mode rejects ConnectX interface ${data_interface}" >&2
    exit 1
  }
fi

declare -a uverbs_devices=()
declare -A seen_uverbs=()
if [[ "${require_rdma}" == "1" ]]; then
  IFS=',' read -r -a hca_specs <<< "${nccl_ib_hca}"
  [[ "${#hca_specs[@]}" -gt 0 ]] || { echo "RDMA requires at least one HCA" >&2; exit 1; }
  for hca_spec in "${hca_specs[@]}"; do
    hca="${hca_spec%%:*}"
    verbs_dir="/sys/class/infiniband/${hca}/device/infiniband_verbs"
    [[ -d "${verbs_dir}" ]] || { echo "missing verbs sysfs directory for ${hca}: ${verbs_dir}" >&2; exit 1; }
    shopt -s nullglob
    matches=("${verbs_dir}"/uverbs*)
    shopt -u nullglob
    [[ "${#matches[@]}" -eq 1 ]] || {
      echo "expected exactly one uverbs device for ${hca}, found ${#matches[@]}" >&2
      exit 1
    }
    uverbs="/dev/infiniband/${matches[0]##*/}"
    [[ -c "${uverbs}" ]] || { echo "missing RDMA character device ${uverbs}" >&2; exit 1; }
    if [[ -z "${seen_uverbs[${uverbs}]:-}" ]]; then
      seen_uverbs[${uverbs}]=1
      uverbs_devices+=("${uverbs}")
    fi
  done
fi
rdma_devices=""
if [[ "${#uverbs_devices[@]}" -gt 0 ]]; then
  rdma_devices="$(IFS=:; printf '%s' "${uverbs_devices[*]}")"
fi

declare -a network_env_args=()
if [[ -n "${nccl_socket_ifname}" ]]; then
  network_env_args+=(--env "NCCL_SOCKET_IFNAME=${nccl_socket_ifname}")
fi
if [[ -n "${gloo_socket_ifname}" ]]; then
  network_env_args+=(--env "GLOO_SOCKET_IFNAME=${gloo_socket_ifname}")
fi
if [[ "${require_rdma}" == "1" ]]; then
  network_env_args+=(
    --env "NCCL_IB_DISABLE=0"
    --env "NCCL_IB_HCA=${nccl_ib_hca}"
    --env "NCCL_IB_GID_INDEX=${nccl_ib_gid_index}"
  )
else
  network_env_args+=(--env "NCCL_IB_DISABLE=1")
fi

declare -a container_fips_args=()
container_fips_mask=""
if [[ "$(head -c 1 /proc/sys/crypto/fips_enabled 2>/dev/null || true)" == "1" ]]; then
  # The public Ray control image and per-job training images are not
  # FIPS-certified. Mask the host kernel flag so their bundled crypto
  # libraries do not enter an unsupported mode during self-tests.
  container_fips_mask="/var/lib/groot-ray/fips-disabled"
  install -d -m 0755 "$(dirname "${container_fips_mask}")"
  printf '0\n' > "${container_fips_mask}"
  chmod 0444 "${container_fips_mask}"
  container_fips_args+=(
    -v "${container_fips_mask}:/proc/sys/crypto/fips_enabled:ro"
  )
fi

common_ray_args=(
  "--node-ip-address=${node_ip}"
  "--temp-dir=${container_ray_tmp}"
  "--node-manager-port=6380"
  "--object-manager-port=6381"
  "--runtime-env-agent-port=6382"
  "--dashboard-agent-grpc-port=6383"
  "--dashboard-agent-listen-port=52365"
  "--metrics-export-port=8080"
  "--min-worker-port=20000"
  # A large persistent cluster can host many concurrent one-node jobs.  Give
  # their drivers and workers enough distinct ports on the head raylet.
  "--max-worker-port=20999"
  "--num-gpus=${ray_num_gpus}"
  "--disable-usage-stats"
  "--block"
)
if [[ "${gpu_profile}" == "gb300-rtx-pro-2000" ]]; then
  # ray_start is passed through a second Bash parser below. Preserve literal
  # quotes around the JSON so brace expansion cannot split it into two args.
  common_ray_args+=("--resources='{\"GB300\":1,\"RTXPRO2000\":1}'")
fi

declare -a control_gpu_telemetry_args=()
if [[ "${gpu_profile}" == "gb300-rtx-pro-2000" ]]; then
  # Ray schedules the logical GPUs into sibling Podman containers. Give the
  # control container utility-only access to the physical devices so its
  # dashboard agent can read NVML telemetry without exposing CUDA compute.
  control_gpu_telemetry_args+=(
    --gpus 'all,"capabilities=utility"'
    --env "NVIDIA_DRIVER_CAPABILITIES=utility"
  )
fi

declare -a head_metrics_env_args=()
if [[ "${role}" == "head" ]]; then
  [[ -n "${head_domain}" ]] || {
    echo "Ray head requires a public dashboard domain" >&2
    exit 1
  }
  head_metrics_env_args+=(
    --env "RAY_PROMETHEUS_HOST=http://127.0.0.1:9090"
    --env "RAY_GRAFANA_HOST=http://127.0.0.1:3000"
    --env "RAY_GRAFANA_IFRAME_HOST=https://${head_domain}/ray/grafana"
  )
fi

if [[ "${role}" == "head" ]]; then
  container="${head_container}"
  ray_start="ray start --head --port=6379 --dashboard-host=127.0.0.1 --dashboard-port=8265 --ray-client-server-port=10001 --include-dashboard=True ${common_ray_args[*]}"
else
  container="${worker_container}"
  ray_start="ray start --address=${head_address} ${common_ray_args[*]}"
fi
ray_cmd="if [[ ! -x /usr/bin/podman ]]; then apt-get update && apt-get install -y --no-install-recommends podman && rm -rf /var/lib/apt/lists/*; fi; ray stop --force >/dev/null 2>&1 || true; exec ${ray_start}"

install -m 0755 \
  /tmp/groot-ray-cluster/colossus_ray_prune.sh \
  /usr/local/sbin/colossus-ray-prune
install -D -m 0755 \
  /tmp/groot-ray-cluster/colossus_ray_podman.sh \
  "${host_podman_wrapper}"
cat > /etc/systemd/system/colossus-ray-prune.service <<SYSTEMD
[Unit]
Description=Prune stale Colossus Ray runtime images and job checkouts
After=podman.socket

[Service]
Type=oneshot
Environment=COLOSSUS_JOB_ROOT=${job_root}
ExecStart=/usr/local/sbin/colossus-ray-prune
SYSTEMD
cat > /etc/systemd/system/colossus-ray-prune.timer <<'SYSTEMD'
[Unit]
Description=Daily Colossus Ray runtime cleanup

[Timer]
OnCalendar=daily
Persistent=true
RandomizedDelaySec=1h

[Install]
WantedBy=timers.target
SYSTEMD
systemctl daemon-reload
systemctl enable --now colossus-ray-prune.timer

docker rm -f "${head_container}" "${worker_container}" >/dev/null 2>&1 || true
# A reset starts a new Ray session. Remove stale sockets and old job/runtime-env
# logs, which may contain serialized credentials from older Ray configurations.
find "${host_ray_tmp}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
install -d -m 0755 "${host_ray_tmp}"
# Runtime-env INFO logs serialize env_vars, including job credentials. Keep
# those node-local rather than forwarding them into user-visible job logs.
docker run -d \
  --name "${container}" \
  --restart unless-stopped \
  --user root \
  --network host \
  --pid host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  "${control_gpu_telemetry_args[@]}" \
  "${head_metrics_env_args[@]}" \
  "${network_env_args[@]}" \
  "${container_fips_args[@]}" \
  -e "CONTAINER_HOST=unix://${podman_socket}" \
  -e "REGISTRY_AUTH_FILE=${podman_auth}" \
  -e "COLOSSUS_DATA_ROOT=${data_root}" \
  -e "COLOSSUS_JOB_ROOT=${job_root}" \
  -e "COLOSSUS_GPU_CDI_DEVICE=nvidia.com/gpu=${gb300_gpu_uuid}" \
  -e "COLOSSUS_GPU_CDI_DEVICES=nvidia.com/gpu=${gb300_gpu_uuid}${rtx_gpu_uuid:+,nvidia.com/gpu=${rtx_gpu_uuid}}" \
  -e "CUDA_VISIBLE_DEVICES=${gb300_gpu_uuid}${rtx_gpu_uuid:+,${rtx_gpu_uuid}}" \
  -e "COLOSSUS_RDMA_DEVICES=${rdma_devices}" \
  -e "COLOSSUS_FIPS_MASK=${container_fips_mask:-}" \
  -e "RAY_RUNTIME_ENV_LOG_TO_DRIVER_ENABLED=0" \
  -v "${podman_socket}:${podman_socket}" \
  -v "${podman_auth}:${podman_auth}:ro" \
  -v "${host_podman_wrapper}:/usr/local/bin/podman:ro" \
  -v "${host_ray_tmp}:${container_ray_tmp}" \
  -v "${data_root}:${data_root}" \
  -v "${job_root}:${job_root}" \
  -w /tmp \
  "${image}" \
  bash -lc "${ray_cmd}"

for _ in $(seq 1 300); do
  if docker exec "${container}" /usr/bin/podman --remote version >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
docker exec "${container}" /usr/bin/podman --remote version >/dev/null
docker exec "${container}" python -c \
  'import ray; assert ray.__version__ == "2.48.0", ray.__version__'

visible_gpu="$(
  timeout --kill-after=10s 120s docker exec \
    -e CUDA_VISIBLE_DEVICES=0 \
    "${container}" \
    podman run --rm "${gpu_test_image}" \
    nvidia-smi --query-gpu=uuid,name --format=csv,noheader
)"
visible_gpu_uuid="${visible_gpu%%,*}"
visible_gpu_uuid="${visible_gpu_uuid//[[:space:]]/}"
visible_gpu_name="${visible_gpu#*,}"
visible_gpu_name="$(printf '%s' "${visible_gpu_name}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
[[ "${visible_gpu_uuid}" == "${gb300_gpu_uuid}" ]] || {
  echo "Podman Ray job sees GPU ${visible_gpu_uuid}, expected ${gb300_gpu_uuid}" >&2
  exit 1
}
[[ "${visible_gpu_name}" == "${target_gpu_name}" ]] || {
  echo "Podman Ray job sees ${visible_gpu_name}, expected ${target_gpu_name}" >&2
  exit 1
}
if [[ "${gpu_profile}" == "gb300-rtx-pro-2000" ]]; then
  visible_gpu="$(
    timeout --kill-after=10s 120s docker exec \
      -e CUDA_VISIBLE_DEVICES=1 \
      "${container}" \
      podman run --rm "${gpu_test_image}" \
      nvidia-smi --query-gpu=uuid,name --format=csv,noheader
  )"
  visible_gpu_uuid="${visible_gpu%%,*}"
  visible_gpu_uuid="${visible_gpu_uuid//[[:space:]]/}"
  [[ "${visible_gpu_uuid}" == "${rtx_gpu_uuid}" ]] || {
    echo "Podman Ray logical GPU 1 sees ${visible_gpu_uuid}, expected ${rtx_gpu_uuid}" >&2
    exit 1
  }
fi

if [[ "${require_rdma}" == "1" ]]; then
  docker exec -e CUDA_VISIBLE_DEVICES=0 "${container}" \
    podman run --rm "${gpu_test_image}" sh -c \
    'IFS=:; for device in ${COLOSSUS_RDMA_DEVICES:?}; do test -c "${device}"; done'
fi
if [[ "${#container_fips_args[@]}" -gt 0 ]]; then
  docker exec "${container}" sh -c \
    'test "$(head -c 1 /proc/sys/crypto/fips_enabled)" = "0"'
fi
docker exec "${container}" python -c \
  'import ray, ssl; ssl.create_default_context(); assert ray.__version__ == "2.48.0"'
EOF

  cat > "${TMP_DIR}/remote_post_training_artifacts.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

image="$1"
data_root="$2"
env_file="$3"

install -d -m 0755 \
  "${data_root}" \
  "${data_root}/ckpts" \
  "${data_root}/checkpoints" \
  "${data_root}/datasets" \
  "${data_root}/hydrator_cache"

declare -a container_fips_args=()
container_fips_mask="/var/lib/groot-ray/fips-disabled"
if [[ "$(head -c 1 /proc/sys/crypto/fips_enabled 2>/dev/null || true)" == "1" ]]; then
  install -d -m 0755 "$(dirname "${container_fips_mask}")"
  printf '0\n' > "${container_fips_mask}"
  chmod 0444 "${container_fips_mask}"
  container_fips_args+=(
    -v "${container_fips_mask}:/proc/sys/crypto/fips_enabled:ro"
  )
fi

docker run --rm -i \
  --network host \
  -v "${data_root}:${data_root}" \
  "${container_fips_args[@]}" \
  -e "GROOT_ARTIFACT_DOWNLOAD_OWNER=$(hostname)" \
  --env-file "${env_file}" \
  "${image}" \
  sh -c 'python -m pip install --quiet --no-cache-dir boto3 && exec python -' <<'PY'
from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.parse import urlparse

import boto3
from botocore.config import Config


def log(message: str) -> None:
    print(f"[groot_artifacts] {message}", flush=True)


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Expected s3://bucket/key URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def s3_client():
    endpoint = required_env("AWS_ENDPOINT_URL")
    if endpoint.startswith("swift://"):
        endpoint = "https://pdx.s8k.io"
    return boto3.client(
        "s3",
        aws_access_key_id=required_env("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=required_env("AWS_SECRET_ACCESS_KEY"),
        endpoint_url=endpoint,
        region_name=os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-east-1",
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def download_object(s3, bucket: str, key: str, target: Path, expected_size: int | None) -> None:
    if target.exists():
        target_size = target.stat().st_size
        if (expected_size is None and target_size > 0) or target_size == expected_size:
            return
    target.parent.mkdir(parents=True, exist_ok=True)
    if expected_size == 0:
        # Hugging Face cache exports include real zero-byte lock objects. There
        # is no response body to resume, and a partial file does not exist yet.
        target.touch()
        return
    # /data may be shared by several Colossus hosts. Including the physical
    # host in the resumable partial prevents concurrent setup processes from
    # moving one another's temporary file out from underneath an active
    # download.
    owner = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in required_env("GROOT_ARTIFACT_DOWNLOAD_OWNER")
    )
    partial = target.with_name(f".{target.name}.{owner}.partial")
    resume_offset = partial.stat().st_size if partial.exists() else 0
    if expected_size is not None and resume_offset > expected_size:
        partial.write_bytes(b"")
        resume_offset = 0
    if expected_size is not None and resume_offset == expected_size:
        partial.replace(target)
        return

    request = {"Bucket": bucket, "Key": key}
    if resume_offset:
        request["Range"] = f"bytes={resume_offset}-"
        log(f"resuming s3://{bucket}/{key} at byte {resume_offset} -> {target}")
    else:
        log(f"downloading s3://{bucket}/{key} -> {target}")
    response = s3.get_object(**request)
    if resume_offset and not str(response.get("ContentRange", "")).startswith(
        f"bytes {resume_offset}-"
    ):
        raise OSError(f"S3 server did not honor resume offset {resume_offset} for {key}")
    max_mib_per_second = int(os.environ.get("GROOT_ARTIFACT_MAX_MIB_PER_SECOND", "0"))
    max_bytes_per_second = max_mib_per_second * 1024 * 1024
    started = time.monotonic()
    downloaded = 0
    with partial.open("ab" if resume_offset else "wb") as file:
        for chunk in response["Body"].iter_chunks(chunk_size=16 * 1024 * 1024):
            if chunk:
                file.write(chunk)
                downloaded += len(chunk)
                if max_bytes_per_second:
                    target_elapsed = downloaded / max_bytes_per_second
                    delay = target_elapsed - (time.monotonic() - started)
                    if delay > 0:
                        time.sleep(delay)
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise OSError(
            f"Downloaded size mismatch for s3://{bucket}/{key}: "
            f"expected {expected_size}, got {partial.stat().st_size}"
        )
    partial.replace(target)


def sync_prefix(s3, source_uri: str, target: Path, required_files: tuple[str, ...] = ()) -> None:
    bucket, prefix = parse_s3_uri(source_uri)
    prefix = prefix.rstrip("/") + "/"
    marker = target / ".groot_artifact_sync_complete"
    if marker.exists() and all((target / name).exists() for name in required_files):
        log(f"exists: {target}")
        return

    log(f"syncing s3://{bucket}/{prefix} -> {target}")
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):]
            if not rel or rel.endswith("/"):
                continue
            download_object(s3, bucket, key, target / rel, int(obj["Size"]))
            count += 1
    if count == 0:
        raise FileNotFoundError(f"No objects found under s3://{bucket}/{prefix}")
    missing = [name for name in required_files if not (target / name).exists()]
    if missing:
        raise FileNotFoundError(f"Artifact sync incomplete for {target}: missing {missing}")
    marker.write_text(json.dumps({"source": f"s3://{bucket}/{prefix}", "objects": count}) + "\n")


def sync_file(s3, source_uri: str, target: Path) -> None:
    bucket, key = parse_s3_uri(source_uri)
    response = s3.head_object(Bucket=bucket, Key=key)
    download_object(s3, bucket, key, target, int(response["ContentLength"]))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(64 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_target_lock_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.groot-artifact.lock")


def atomic_publish_directory(prepared: Path, target: Path) -> None:
    """Publish a complete directory without exposing a missing/partial target."""
    if not target.exists():
        prepared.replace(target)
        return
    if target.is_symlink() or not target.is_dir():
        raise OSError(f"Artifact target must be a real directory: {target}")

    # Linux renameat2(RENAME_EXCHANGE) swaps two populated directories in one
    # VFS operation. Afterward `prepared` names the old target, which the
    # caller removes while the target-scoped lock is still held.
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError("Atomic artifact directory exchange requires renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(prepared),
        -100,
        os.fsencode(target),
        2,  # RENAME_EXCHANGE
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            f"Could not atomically replace artifact directory {target}",
        )


def _stage_archive_locked(
    s3,
    source_uri: str,
    expected_sha256: str,
    target: Path,
) -> None:
    marker = target / ".groot_archive_source"
    expected_marker = f"{source_uri}\n{expected_sha256.lower()}\n"
    archive = target.parent / ".artifact-cache" / PurePosixPath(source_uri).name
    archive.parent.mkdir(parents=True, exist_ok=True)
    lock_path = archive.with_name(f".{archive.name}.lock")
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        # Recheck after taking the shared-filesystem lock because another host
        # may have completed extraction while this process was waiting.
        if marker.is_file() and marker.read_text() == expected_marker:
            log(f"exists: {target}")
            return
        sync_file(s3, source_uri, archive)
        actual_sha256 = sha256(archive)
        if actual_sha256.lower() != expected_sha256.lower():
            log(
                f"discarding corrupt cached archive {archive}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
            archive.unlink(missing_ok=True)
            sync_file(s3, source_uri, archive)
            actual_sha256 = sha256(archive)
        if actual_sha256.lower() != expected_sha256.lower():
            archive.unlink(missing_ok=True)
            raise OSError(
                f"Archive checksum mismatch for {source_uri}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        prepared = Path(
            tempfile.mkdtemp(
                prefix=f".{target.name}.extract-",
                dir=target.parent,
            )
        )
        try:
            log(f"extracting {archive} -> {prepared}")
            with tarfile.open(archive) as contents:
                contents.extractall(prepared, filter="data")
            (prepared / marker.name).write_text(expected_marker)
            atomic_publish_directory(prepared, target)
        finally:
            # After an exchange this name holds the old target. On extraction
            # failure it holds the incomplete new target. Neither may linger.
            if prepared.exists():
                shutil.rmtree(prepared)


def stage_archive(s3, source_uri: str, expected_sha256: str, target: Path) -> None:
    if not source_uri:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with artifact_target_lock_path(target).open("a+") as target_lock:
        fcntl.flock(target_lock.fileno(), fcntl.LOCK_EX)
        _stage_archive_locked(s3, source_uri, expected_sha256, target)


def patch_robocasa_python312(runtime: Path) -> None:
    replacements = {
        "robocasa/utils/visuals_utls.py": {
            "from dataclasses import dataclass": "from dataclasses import dataclass, field",
            "rgba_a: np.ndarray = np.array([0, 0, 0, 1])": (
                "rgba_a: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1]))"
            ),
            "rgba_b: np.ndarray = np.array([1, 1, 1, 1])": (
                "rgba_b: np.ndarray = field(default_factory=lambda: np.array([1, 1, 1, 1]))"
            ),
        },
        "robocasa/utils/scene/configs.py": {
            "from dataclasses import dataclass": "from dataclasses import dataclass, field",
            "x_range: np.ndarray = np.zeros(2)": (
                "x_range: np.ndarray = field(default_factory=lambda: np.zeros(2))"
            ),
            "y_range: np.ndarray = np.zeros(2)": (
                "y_range: np.ndarray = field(default_factory=lambda: np.zeros(2))"
            ),
            "rotation: np.ndarray = np.zeros(2)": (
                "rotation: np.ndarray = field(default_factory=lambda: np.zeros(2))"
            ),
            "reference_pos: np.ndarray = np.zeros(3)": (
                "reference_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))"
            ),
        },
    }
    for relative, substitutions in replacements.items():
        path = runtime / relative
        if not path.is_file():
            continue
        text = path.read_text()
        for source, replacement in substitutions.items():
            if source in text and replacement not in text:
                text = text.replace(source, replacement)
        path.write_text(text)


def install_robocasa_python_dependencies(runtime: Path) -> None:
    target = runtime / "python"
    marker = target / ".groot_robocasa_dependencies"
    expected = "mink==0.0.10\ndaqp==0.8.7\nmujoco==3.3.1\n"
    if marker.is_file() and marker.read_text() == expected:
        return
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-cache-dir",
            "--no-deps",
            "--target",
            str(target),
            "mink==0.0.10",
            "daqp==0.8.7",
            "mujoco==3.3.1",
        ],
        check=True,
    )
    marker.write_text(expected)


def stage_robocasa_runtime(
    s3,
    source_uri: str,
    expected_sha256: str,
    runtime: Path,
) -> None:
    """Stage and postprocess one runtime under a single target-scoped lock."""
    runtime.parent.mkdir(parents=True, exist_ok=True)
    with artifact_target_lock_path(runtime).open("a+") as target_lock:
        fcntl.flock(target_lock.fileno(), fcntl.LOCK_EX)
        # Both markers are rechecked only after the target lock is held, so
        # concurrent setup processes cannot extract, patch, or pip-install
        # into this runtime simultaneously.
        _stage_archive_locked(s3, source_uri, expected_sha256, runtime)
        patch_robocasa_python312(runtime)
        install_robocasa_python_dependencies(runtime)


def main() -> None:
    root = Path(required_env("GROOT_DATA_ROOT"))
    model_deps = required_env("GROOT_MODEL_DEPS_URI").rstrip("/")
    post_training_checkpoint = required_env("GROOT_POST_TRAINING_CHECKPOINT_URI")
    checkpoint_destination = required_env("GROOT_POST_TRAINING_CHECKPOINT_DESTINATION")
    s3 = s3_client()

    sync_prefix(
        s3,
        f"{model_deps}/cosmos25/cosmos_reason1",
        root / "ckpts/cosmos25/cosmos_reason1",
        required_files=("config.json", "model.safetensors.index.json", "tokenizer.json"),
    )
    sync_prefix(
        s3,
        f"{model_deps}/cosmos25/Cosmos-Predict2.5-2B-diffusers/transformer",
        root / "ckpts/cosmos25/Cosmos-Predict2.5-2B-diffusers/transformer",
        required_files=("config.json", "diffusion_pytorch_model.safetensors"),
    )
    sync_file(
        s3,
        f"{model_deps}/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
        root / "ckpts/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
    )
    sync_prefix(
        s3,
        post_training_checkpoint,
        root / "checkpoints" / checkpoint_destination,
    )
    stage_archive(
        s3,
        os.environ.get("GROOT_DATASET_ARCHIVE_URI", "").strip(),
        os.environ.get("GROOT_DATASET_ARCHIVE_SHA256", "").strip(),
        root
        / "datasets/robocasa_benchmark/new-robocasa/robocasa-datasets/v1.0/pretrain",
    )
    runtime_uri = os.environ.get("GROOT_ROBOCASA_RUNTIME_ARCHIVE_URI", "").strip()
    if runtime_uri:
        stage_robocasa_runtime(
            s3,
            runtime_uri,
            os.environ.get("GROOT_ROBOCASA_RUNTIME_ARCHIVE_SHA256", "").strip(),
            root / "runtime",
        )
    log("post-training artifacts are ready")


main()
PY
EOF

  cat > "${TMP_DIR}/remote_ray_metrics.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

domain="$1"
prometheus_image="$2"
grafana_image="$3"
data_root="$4"

host_ray_tmp="${data_root}/ray"
container_ray_tmp="${host_ray_tmp}"
prometheus_container="groot-ray-prometheus"
grafana_container="groot-ray-grafana"

for _ in $(seq 1 120); do
  session_target="$(readlink "${host_ray_tmp}/session_latest" 2>/dev/null || true)"
  session_name="${session_target##*/}"
  prometheus_config="${host_ray_tmp}/${session_name}/metrics/prometheus/prometheus.yml"
  grafana_config="${host_ray_tmp}/${session_name}/metrics/grafana/grafana.ini"
  if [[ -n "${session_name}" && -r "${prometheus_config}" && -r "${grafana_config}" ]]; then
    break
  fi
  sleep 1
done
[[ -r "${prometheus_config}" ]] || {
  echo "missing Ray-generated Prometheus config: ${prometheus_config}" >&2
  exit 1
}
[[ -r "${grafana_config}" ]] || {
  echo "missing Ray-generated Grafana config: ${grafana_config}" >&2
  exit 1
}

docker volume create groot-ray-prometheus-data >/dev/null
docker volume create groot-ray-grafana-data >/dev/null
docker rm -f "${prometheus_container}" "${grafana_container}" >/dev/null 2>&1 || true

docker run -d \
  --name "${prometheus_container}" \
  --restart unless-stopped \
  --network host \
  --log-opt max-size=50m \
  --log-opt max-file=3 \
  -v "${host_ray_tmp}:${container_ray_tmp}:ro" \
  -v groot-ray-prometheus-data:/prometheus \
  "${prometheus_image}" \
  --config.file="${container_ray_tmp}/session_latest/metrics/prometheus/prometheus.yml" \
  --storage.tsdb.path=/prometheus \
  --web.enable-lifecycle \
  --web.listen-address=127.0.0.1:9090 >/dev/null

docker run -d \
  --name "${grafana_container}" \
  --restart unless-stopped \
  --network host \
  --log-opt max-size=50m \
  --log-opt max-file=3 \
  -e "GF_SERVER_HTTP_ADDR=127.0.0.1" \
  -e "GF_SERVER_ROOT_URL=https://${domain}/ray/grafana/" \
  -e "GF_SERVER_SERVE_FROM_SUB_PATH=true" \
  -e "GF_SECURITY_ALLOW_EMBEDDING=true" \
  -e "GF_AUTH_ANONYMOUS_ENABLED=true" \
  -e "GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer" \
  -e "GF_PATHS_PROVISIONING=${container_ray_tmp}/session_latest/metrics/grafana/provisioning" \
  -v "${host_ray_tmp}:${container_ray_tmp}:ro" \
  -v groot-ray-grafana-data:/var/lib/grafana \
  "${grafana_image}" \
  --config="${container_ray_tmp}/session_latest/metrics/grafana/grafana.ini" >/dev/null

for _ in $(seq 1 120); do
  if curl --fail --silent --show-error http://127.0.0.1:9090/-/ready >/dev/null \
      && curl --fail --silent --show-error http://127.0.0.1:3000/api/health >/dev/null; then
    exit 0
  fi
  sleep 2
done

docker logs --tail 200 "${prometheus_container}" >&2 || true
docker logs --tail 200 "${grafana_container}" >&2 || true
exit 1
EOF

  cat > "${TMP_DIR}/remote_nginx_ray.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

domain="$1"
ssl_cert="$2"
ssl_key="$3"
auth_user="$4"
password_file="$5"

if [[ ! -r "${ssl_cert}" ]]; then
  echo "missing readable ssl cert: ${ssl_cert}" >&2
  exit 1
fi
if [[ ! -r "${ssl_key}" ]]; then
  echo "missing readable ssl key: ${ssl_key}" >&2
  exit 1
fi
if [[ ! -r "${password_file}" ]]; then
  echo "missing readable Ray auth password file" >&2
  exit 1
fi

DEBIAN_FRONTEND=noninteractive apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y nginx openssl curl

install -d -m 0755 /etc/nginx/sites-available /etc/nginx/sites-enabled
# SHA-512 crypt remains usable on FIPS-enabled Colossus hosts; APR1 relies on
# MD5 and produces credentials that nginx cannot validate in FIPS mode.
hash="$(openssl passwd -6 -stdin < "${password_file}")"
printf '%s:%s\n' "${auth_user}" "${hash}" > /etc/nginx/htpasswd-groot-ray
chown root:www-data /etc/nginx/htpasswd-groot-ray
chmod 0640 /etc/nginx/htpasswd-groot-ray

cat > /etc/nginx/sites-available/groot-ray.conf <<NGINX
server {
    listen 80;
    server_name ${domain};

    location = /ray {
        return 301 https://\$host/ray/;
    }

    location /ray/ {
        return 301 https://\$host\$request_uri;
    }
}

server {
    listen 443 ssl;
    server_name ${domain};

    ssl_certificate ${ssl_cert};
    ssl_certificate_key ${ssl_key};

    client_max_body_size 2g;

    location = /ray {
        return 301 /ray/;
    }

    location = /ray/grafana {
        return 301 /ray/grafana/;
    }

    location /ray/grafana/ {
        auth_basic "GR00T Ray";
        auth_basic_user_file /etc/nginx/htpasswd-groot-ray;

        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header Authorization "";
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Existing heads may predate RAY_GRAFANA_IFRAME_HOST. Rewrite only the
    # health response's loopback iframe URL so the browser stays on the
    # authenticated, same-origin Grafana proxy. New heads already return this
    # public URL directly from Ray.
    location = /ray/api/grafana_health {
        auth_basic "GR00T Ray";
        auth_basic_user_file /etc/nginx/htpasswd-groot-ray;

        proxy_pass http://127.0.0.1:8265/api/grafana_health;
        proxy_set_header Host \$host;
        proxy_set_header Accept-Encoding "";
        sub_filter_once on;
        sub_filter_types application/json;
        sub_filter 'http://localhost:3000' 'https://${domain}/ray/grafana';
        sub_filter 'http://127.0.0.1:3000' 'https://${domain}/ray/grafana';
    }

    location /ray/ {
        auth_basic "GR00T Ray";
        auth_basic_user_file /etc/nginx/htpasswd-groot-ray;

        proxy_pass http://127.0.0.1:8265/;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}
NGINX

ln -sf /etc/nginx/sites-available/groot-ray.conf /etc/nginx/sites-enabled/groot-ray.conf
nginx -t
systemctl enable --now nginx
systemctl reload nginx || systemctl restart nginx

if command -v ufw >/dev/null 2>&1 && ufw status | grep -q '^Status: active'; then
  ufw allow 443/tcp comment 'GR00T Ray HTTPS'
fi

auth_code=""
for _ in $(seq 1 30); do
  auth_code="$(curl -k -s -o /dev/null -w '%{http_code}' "https://${domain}/ray/api/version" --resolve "${domain}:443:127.0.0.1" || true)"
  if [[ "${auth_code}" == "401" ]]; then
    break
  fi
  sleep 1
done
if [[ "${auth_code}" != "401" ]]; then
  echo "expected unauthenticated /ray/api/version to return 401, got ${auth_code}" >&2
  exit 1
fi

curl_config="$(mktemp /tmp/groot-ray-curl.XXXXXX)"
chmod 0600 "${curl_config}"
{
  printf 'user = "%s:%s"\n' "${auth_user}" "$(cat "${password_file}")"
  printf 'insecure\n'
  printf 'silent\n'
  printf 'show-error\n'
  printf 'fail\n'
} > "${curl_config}"
authenticated=0
for _ in $(seq 1 30); do
  if curl --config "${curl_config}" "https://${domain}/ray/api/version" --resolve "${domain}:443:127.0.0.1" >/dev/null; then
    authenticated=1
    break
  fi
  sleep 1
done
if [[ "${authenticated}" != "1" ]]; then
  echo "authenticated /ray/api/version check did not become ready" >&2
  exit 1
fi
rm -f "${curl_config}" "${password_file}"
EOF

  chmod +x \
    "${TMP_DIR}/remote_ray_container.sh" \
    "${TMP_DIR}/remote_post_training_artifacts.sh" \
    "${TMP_DIR}/remote_ray_metrics.sh" \
    "${TMP_DIR}/remote_nginx_ray.sh"
}

create_local_secret_files() {
  printf '%s' "${NVCR_TOKEN}" > "${TMP_DIR}/nvcr-token"
  printf '%s' "${RAY_AUTH_PASSWORD}" > "${TMP_DIR}/ray-auth-password"
  chmod 0600 "${TMP_DIR}/nvcr-token" "${TMP_DIR}/ray-auth-password"
  if [[ "${SYNC_POST_TRAINING_ARTIFACTS}" == "1" ]]; then
    local bucket_endpoint_url="${BUCKET_ENDPOINT_URL_VALUE}"
    if [[ "${bucket_endpoint_url}" == swift://* ]]; then
      bucket_endpoint_url="https://pdx.s8k.io"
    fi
    local gear_data_credentials
    gear_data_credentials="$(python3 - \
      "${BUCKET_ACCESS_KEY_ID_VALUE}" \
      "${BUCKET_SECRET_ACCESS_KEY_VALUE}" \
      "${bucket_endpoint_url}" \
      "${BUCKET_REGION_VALUE}" <<'PY'
import base64
import json
import sys

access_key, secret_key, endpoint_url, region = sys.argv[1:]
profile = {
    "aws_access_key_id": access_key,
    "aws_secret_access_key": secret_key,
    "endpoint_url": endpoint_url,
    "region": region,
}
# PDX is a path-style S3-compatible endpoint. Do not alias it as the reserved
# "aws" profile: DataSourceResolver treats that profile as regional AWS and
# sends requests to the wrong endpoint.
print(base64.b64encode(json.dumps({"default": profile}).encode()).decode())
PY
)"
    {
      printf 'AWS_ACCESS_KEY_ID=%s\n' "${BUCKET_ACCESS_KEY_ID_VALUE}"
      printf 'AWS_SECRET_ACCESS_KEY=%s\n' "${BUCKET_SECRET_ACCESS_KEY_VALUE}"
      printf 'AWS_ENDPOINT_URL=%s\n' "${bucket_endpoint_url}"
      printf 'AWS_DEFAULT_REGION=%s\n' "${BUCKET_REGION_VALUE}"
      printf 'AWS_REGION=%s\n' "${BUCKET_REGION_VALUE}"
      printf 'GEAR_DATA_CREDENTIALS=%s\n' "${gear_data_credentials}"
      printf 'GROOT_DATA_ROOT=%s\n' "${GROOT_DATA_ROOT}"
      printf 'GROOT_MODEL_DEPS_URI=%s\n' "${MODEL_DEPS_URI}"
      printf 'GROOT_POST_TRAINING_CHECKPOINT_URI=%s\n' "${POST_TRAINING_CHECKPOINT_URI}"
      printf 'GROOT_POST_TRAINING_CHECKPOINT_DESTINATION=%s\n' "${POST_TRAINING_CHECKPOINT_DESTINATION}"
      printf 'GROOT_DATASET_ARCHIVE_URI=%s\n' "${DATASET_ARCHIVE_URI}"
      printf 'GROOT_DATASET_ARCHIVE_SHA256=%s\n' "${DATASET_ARCHIVE_SHA256}"
      printf 'GROOT_ROBOCASA_RUNTIME_ARCHIVE_URI=%s\n' "${ROBOCASA_RUNTIME_ARCHIVE_URI}"
      printf 'GROOT_ROBOCASA_RUNTIME_ARCHIVE_SHA256=%s\n' "${ROBOCASA_RUNTIME_ARCHIVE_SHA256}"
      printf 'GROOT_ARTIFACT_MAX_MIB_PER_SECOND=%s\n' "${ARTIFACT_MAX_MIB_PER_SECOND}"
    } > "${TMP_DIR}/data-credentials.env"
    chmod 0600 "${TMP_DIR}/data-credentials.env"
    python3 - \
      "${bucket_endpoint_url}" \
      "${BUCKET_REGION_VALUE}" \
      > "${TMP_DIR}/msc-config.json" <<'PY'
import json
import sys

endpoint_url, region = sys.argv[1:]
config = {
    "profiles": {
        "GearCheckpoints": {
            "storage_provider": {
                "type": "s8k",
                "options": {
                    "region_name": region,
                    "base_path": "GearCheckpoints",
                    "endpoint_url": endpoint_url,
                    "rust_client": {},
                },
            },
            "credentials_provider": {
                "type": "S3Credentials",
                "options": {
                    "access_key": "${AWS_ACCESS_KEY_ID}",
                    "secret_key": "${AWS_SECRET_ACCESS_KEY}",
                },
            },
        }
    }
}
json.dump(config, sys.stdout, indent=2)
sys.stdout.write("\n")
PY
    # This profile contains no credential values: MSC expands the two
    # environment references from each job's existing runtime credentials.
    chmod 0644 "${TMP_DIR}/msc-config.json"
  else
    : > "${TMP_DIR}/data-credentials.env"
    chmod 0600 "${TMP_DIR}/data-credentials.env"
    rm -f "${TMP_DIR}/msc-config.json"
  fi
}

remote_prepare_tmp() {
  local host="$1"
  local user="$2"
  local password="$3"
  remote_ssh "${host}" "${user}" "${password}" "rm -rf $(shell_quote "${REMOTE_ROOT}") && mkdir -p $(shell_quote "${REMOTE_ROOT}") && chmod 700 $(shell_quote "${REMOTE_ROOT}") && printf '%s\n' $(shell_quote "${REMOTE_ROOT}")"
}

copy_common_remote_files() {
  local host="$1"
  local user="$2"
  local password="$3"
  remote_scp_to "${host}" "${user}" "${password}" "${HOST_SETUP_SCRIPT}" "${REMOTE_ROOT}/setup_bare_metal_gpu_host.sh" || return 1
  remote_scp_to "${host}" "${user}" "${password}" "${PODMAN_WRAPPER}" "${REMOTE_ROOT}/colossus_ray_podman.sh" || return 1
  remote_scp_to "${host}" "${user}" "${password}" "${PRUNE_SCRIPT}" "${REMOTE_ROOT}/colossus_ray_prune.sh" || return 1
  remote_scp_to "${host}" "${user}" "${password}" "${RAY_COMPAT_SCRIPT}" "${REMOTE_ROOT}/colossus_ray_sitecustomize.py" || return 1
  remote_scp_to "${host}" "${user}" "${password}" "${TMP_DIR}/remote_ray_container.sh" "${REMOTE_ROOT}/remote_ray_container.sh" || return 1
  remote_scp_to "${host}" "${user}" "${password}" "${TMP_DIR}/remote_post_training_artifacts.sh" "${REMOTE_ROOT}/remote_post_training_artifacts.sh" || return 1
  remote_scp_to "${host}" "${user}" "${password}" "${TMP_DIR}/remote_ray_metrics.sh" "${REMOTE_ROOT}/remote_ray_metrics.sh" || return 1
  remote_scp_to "${host}" "${user}" "${password}" "${TMP_DIR}/nvcr-token" "${REMOTE_ROOT}/nvcr-token" || return 1
  remote_scp_to "${host}" "${user}" "${password}" "${TMP_DIR}/data-credentials.env" "${REMOTE_ROOT}/data-credentials.env" || return 1
  remote_ssh "${host}" "${user}" "${password}" "chmod 700 $(shell_quote "${REMOTE_ROOT}") && chmod 600 $(shell_quote "${REMOTE_ROOT}/nvcr-token") $(shell_quote "${REMOTE_ROOT}/data-credentials.env") && chmod +x $(shell_quote "${REMOTE_ROOT}/setup_bare_metal_gpu_host.sh") $(shell_quote "${REMOTE_ROOT}/colossus_ray_podman.sh") $(shell_quote "${REMOTE_ROOT}/colossus_ray_prune.sh") $(shell_quote "${REMOTE_ROOT}/remote_ray_container.sh") $(shell_quote "${REMOTE_ROOT}/remote_post_training_artifacts.sh") $(shell_quote "${REMOTE_ROOT}/remote_ray_metrics.sh")" || return 1
  if [[ -s "${TMP_DIR}/msc-config.json" ]]; then
    remote_scp_to "${host}" "${user}" "${password}" \
      "${TMP_DIR}/msc-config.json" "${REMOTE_ROOT}/msc-config.json" || return 1
    remote_sudo_bash_lc "${host}" "${user}" "${password}" \
      "install -d -m 0755 $(shell_quote "${GROOT_DATA_ROOT}/config/msc") && install -m 0444 $(shell_quote "${REMOTE_ROOT}/msc-config.json") $(shell_quote "${GROOT_DATA_ROOT}/config/msc/config.json")" || return 1
  fi
}

run_host_setup() {
  local idx="$1"
  local host="${HOSTS[$idx]}"
  local user="${USERS[$idx]}"
  local password="${PASSWORDS[$idx]}"
  local -a setup_args=(
    "--nvidia-test-image" "${GPU_TEST_IMAGE}"
    "--nvcr-token-file" "${REMOTE_ROOT}/nvcr-token"
  )

  if [[ "${REQUIRE_RDMA}" == "1" ]]; then
    setup_args+=(
      "--validate-rdma"
      "--rdma-interface" "${DATA_INTERFACE}"
      "--rdma-hca" "${NCCL_IB_HCA}"
      "--rdma-gid-index" "${NCCL_IB_GID_INDEX}"
    )
  fi

  remote_sudo_script "${host}" "${user}" "${password}" \
    "${REMOTE_ROOT}/setup_bare_metal_gpu_host.sh" \
    "${setup_args[@]}"
}

sync_post_training_artifacts() {
  if [[ "${SYNC_POST_TRAINING_ARTIFACTS}" != "1" ]]; then
    return 0
  fi

  local idx="$1"
  local host="${HOSTS[$idx]}"
  local user="${USERS[$idx]}"
  local password="${PASSWORDS[$idx]}"

  log "${host}: syncing Colossus post-training artifacts under ${GROOT_DATA_ROOT}"
  remote_sudo_script "${host}" "${user}" "${password}" \
    "${REMOTE_ROOT}/remote_post_training_artifacts.sh" \
    "${ARTIFACT_IMAGE}" "${GROOT_DATA_ROOT}" "${REMOTE_ROOT}/data-credentials.env"
}

configure_static_data_interface() {
  local idx="$1"
  local host="${HOSTS[$idx]}"
  local user="${USERS[$idx]}"
  local password="${PASSWORDS[$idx]}"
  local resource_name="${host%%.*}"
  local interface="${DATA_INTERFACE:-${PREFERRED_ETHERNET_INTERFACE}}"
  local numeric_suffix last_octet data_ip quoted_interface

  DATA_INTERFACES[$idx]=""
  DATA_IPS[$idx]=""
  [[ -n "${interface}" ]] || return 0
  quoted_interface="$(shell_quote "${interface}")"

  if [[ -n "${PREFERRED_ETHERNET_INTERFACE}" ]]; then
    # ConnectX management is the only cabled interface on much of the current
    # fleet. Prefer normal Ethernet whenever it is actually usable, but do not
    # reject a host merely because that optional port is uncabled.
    if ! remote_sudo_bash_lc "${host}" "${user}" "${password}" \
      "test -d /sys/class/net/${quoted_interface}; ip link set dev ${quoted_interface} up; sleep 2; test \"\$(cat /sys/class/net/${quoted_interface}/carrier)\" = 1; driver=\$(basename \"\$(readlink -f /sys/class/net/${quoted_interface}/device/driver)\"); test -n \"\${driver}\"; test \"\${driver}\" != mlx5_core"; then
      log "${host}: ${interface} is absent, uncabled, or mlx5; falling back to routed management TCP"
      return 0
    fi
    log "${host}: selected connected non-mlx5 Ethernet interface ${interface}"
  fi

  DATA_INTERFACES[$idx]="${interface}"
  [[ -n "${STATIC_DATA_IP_PREFIX}" ]] || return 0

  [[ "${resource_name}" =~ -([0-9]+)$ ]] || {
    log "${host}: cannot derive a static data IP from the resource name"
    return 1
  }
  numeric_suffix="$((10#${BASH_REMATCH[1]}))"
  last_octet="$((numeric_suffix % 250 + 1))"
  data_ip="${STATIC_DATA_IP_PREFIX}.${last_octet}"
  log "${host}: configuring ${interface} as ${data_ip}/24"
  remote_sudo_bash_lc "${host}" "${user}" "${password}" \
    "ip link set dev ${quoted_interface} up; sleep 2; test \"\$(cat /sys/class/net/${quoted_interface}/carrier)\" = 1; ip address replace $(shell_quote "${data_ip}/24") dev ${quoted_interface}; install -d -m 0755 /etc/netplan; printf '%s\n' 'network:' '  version: 2' '  ethernets:' '    ${interface}:' '      dhcp4: false' '      optional: true' '      addresses:' '        - ${data_ip}/24' > /etc/netplan/90-groot-ray-ethernet.yaml; chmod 0600 /etc/netplan/90-groot-ray-ethernet.yaml"
  DATA_IPS[$idx]="${data_ip}"
}

remote_nvidia_numa_param() {
  local host="$1"
  local user="$2"
  local password="$3"
  remote_sudo_capture "${host}" "${user}" "${password}" \
    "awk -F': ' '\$1 == \"EnableUserNUMAManagement\" {print \$2}' /proc/driver/nvidia/params 2>/dev/null || true" \
    | head -n 1 \
    | tr -d '[:space:]'
}

remove_remote_nvcr_token() {
  local host="$1"
  local user="$2"
  local password="$3"
  remote_sudo_bash_lc "${host}" "${user}" "${password}" "rm -f $(shell_quote "${REMOTE_ROOT}/nvcr-token")"
}

ensure_nvidia_ready_with_reboot() {
  local idx="$1"
  local host="${HOSTS[$idx]}"
  local user="${USERS[$idx]}"
  local password="${PASSWORDS[$idx]}"
  local live_numa

  live_numa="$(remote_nvidia_numa_param "${host}" "${user}" "${password}" || true)"
  if [[ "${live_numa}" == "0" ]]; then
    return 0
  fi

  log "${host}: NVIDIA driver/module options are not active yet (EnableUserNUMAManagement=${live_numa:-missing}); rebooting before Ray bootstrap"
  reboot_host_and_wait "${host}" "${user}" "${password}" || return 1

  log "${host}: refreshing remote setup files after reboot"
  remote_prepare_tmp "${host}" "${user}" "${password}" >/dev/null || return 1
  copy_common_remote_files "${host}" "${user}" "${password}" || return 1

  log "${host}: re-running host setup validation after reboot"
  run_host_setup "${idx}" || {
    log "${host}: host setup failed after reboot"
    return 1
  }

  live_numa="$(remote_nvidia_numa_param "${host}" "${user}" "${password}" || true)"
  if [[ "${live_numa}" != "0" ]]; then
    log "${host}: NVIDIA driver is still not loaded with EnableUserNUMAManagement=0 after reboot"
    return 1
  fi
}

setup_one_host() {
  local idx="$1"
  local host="${HOSTS[$idx]}"
  local user="${USERS[$idx]}"
  local password="${PASSWORDS[$idx]}"

  log "${host}: preparing GPU/Docker/image prerequisites"
  remote_prepare_tmp "${host}" "${user}" "${password}" >/dev/null || {
    log "${host}: failed to prepare remote setup directory"
    return 1
  }
  copy_common_remote_files "${host}" "${user}" "${password}" || {
    log "${host}: failed to copy remote setup files"
    return 1
  }

  if ! run_host_setup "${idx}"; then
    log "${host}: host setup failed"
    return 1
  fi

  if ! ensure_nvidia_ready_with_reboot "${idx}"; then
    return 1
  fi

  if ! configure_static_data_interface "${idx}"; then
    log "${host}: normal Ethernet data interface is unavailable"
    return 1
  fi

  if ! sync_post_training_artifacts "${idx}"; then
    return 1
  fi

  local live_numa
  live_numa="$(remote_nvidia_numa_param "${host}" "${user}" "${password}" || true)"
  if [[ "${live_numa}" != "0" ]]; then
    log "${host}: NVIDIA driver is not loaded with EnableUserNUMAManagement=0 after reboot handling"
    return 1
  fi

  local gpu_count gpu_inventory
  gpu_inventory="$(remote_sudo_capture "${host}" "${user}" "${password}" \
    "timeout --kill-after=5s 20s nvidia-smi --query-gpu=uuid,name --format=csv,noheader")" || {
    log "${host}: failed to query GPU UUIDs and model names"
    return 1
  }
  if ! printf '%s\n' "${gpu_inventory}" \
    | awk -F',' \
        -v target="${TARGET_GPU_NAME}" \
        -v secondary="${SECONDARY_GPU_NAME_PATTERN}" \
        -v profile="${GPU_PROFILE}" '
        {
          uuid=$1; name=$2
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", uuid)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
          if (uuid ~ /^GPU-[[:xdigit:]-]+$/) total++
          if (name == target && uuid ~ /^GPU-[[:xdigit:]-]+$/) {
            primary_count++
          }
          if (index(name, secondary) != 0 && uuid ~ /^GPU-[[:xdigit:]-]+$/) {
            secondary_count++
          }
        }
        END {
          if (profile == "gb300-rtx-pro-2000") {
            exit !(total == 2 && primary_count == 1 && secondary_count == 1)
          }
          exit !(primary_count == 1)
        }
      '; then
    log "${host}: GPU inventory does not match profile ${GPU_PROFILE}"
    return 1
  fi
  if [[ "${GPU_PROFILE}" == "gb300-rtx-pro-2000" ]]; then
    gpu_count="2"
  else
    gpu_count="1"
  fi

  remote_sudo_bash_lc "${host}" "${user}" "${password}" \
    "docker pull $(shell_quote "${IMAGE}") >/dev/null && docker image inspect $(shell_quote "${IMAGE}") >/dev/null" || {
    log "${host}: Ray control image is unavailable"
    return 1
  }

  HOST_READY[$idx]="1"
  GPU_COUNTS[$idx]="${gpu_count}"
  log "${host}: ready with ${gpu_count} GPU(s) for profile ${GPU_PROFILE}"
}

host_setup_child_cleanup() {
  local idx="$1"
  local semaphore_fd="$2"
  local status="$3"
  local host="${HOSTS[$idx]}"
  local user="${USERS[$idx]}"
  local password="${PASSWORDS[$idx]}"

  trap - EXIT
  if ! remove_remote_nvcr_token "${host}" "${user}" "${password}" >/dev/null 2>&1; then
    log "${host}: failed to remove remote NVCR token"
    if [[ "${status}" -eq 0 ]]; then
      status=1
    fi
  fi
  printf '.' >&"${semaphore_fd}" || true
  exit "${status}"
}

collect_host_setup_results() {
  local idx pid host result_path log_path gpu_count data_ip data_interface wait_status
  while [[ "$#" -gt 0 ]]; do
    idx="$1"
    pid="$2"
    shift 2
    host="${HOSTS[$idx]}"
    result_path="${TMP_DIR}/host-setup-${idx}.result"
    log_path="${TMP_DIR}/host-setup-${idx}.log"

    if wait "${pid}"; then
      wait_status=0
    else
      wait_status=$?
    fi
    forget_host_setup_pid "${pid}"

    if [[ "${wait_status}" -eq 0 && -r "${result_path}" ]]; then
      IFS='|' read -r gpu_count data_ip data_interface < "${result_path}" || true
      if [[ "${gpu_count}" =~ ^[0-9]+$ && "${gpu_count}" -gt 0 ]]; then
        HOST_READY[$idx]="1"
        GPU_COUNTS[$idx]="${gpu_count}"
        DATA_IPS[$idx]="${data_ip:-}"
        DATA_INTERFACES[$idx]="${data_interface:-}"
        log "${host}: ready with ${gpu_count} GPU(s), data interface ${data_interface:-routed-management}"
        rm -f "${result_path}" "${log_path}"
        continue
      fi
    fi

    HOST_READY[$idx]="0"
    FAILED_HOSTS+=("${host}")
    log "${host}: setup failed; this host will not be added to Ray"
    if [[ -s "${log_path}" ]]; then
      log "${host}: failure log tail follows"
      tail -n 20 "${log_path}" | sed "s/^/[setup_ray_cluster] ${host}: /"
    fi
  done
}

setup_hosts() {
  local idx result_path log_path slot child_pid
  local concurrency="${HOST_SETUP_CONCURRENCY}"
  local semaphore_path="${TMP_DIR}/host-setup-semaphore"
  local semaphore_fd
  local -a host_jobs=()

  if (( concurrency > ${#HOSTS[@]} )); then
    concurrency="${#HOSTS[@]}"
  fi

  mkfifo "${semaphore_path}"
  exec {semaphore_fd}<>"${semaphore_path}"
  rm -f "${semaphore_path}"
  for ((slot = 0; slot < concurrency; slot++)); do
    printf '.' >&"${semaphore_fd}"
  done

  log "preparing ${#HOSTS[@]} host(s) with concurrency ${concurrency}"
  for idx in "${!HOSTS[@]}"; do
    IFS= read -r -n 1 -u "${semaphore_fd}" _
    result_path="${TMP_DIR}/host-setup-${idx}.result"
    log_path="${TMP_DIR}/host-setup-${idx}.log"
    rm -f "${result_path}" "${log_path}"
    log "${HOSTS[$idx]}: host setup started"
    # Monitor mode gives each background setup its own process group. Cleanup
    # can then terminate both the wrapper and an in-flight ssh/scp descendant.
    set -m
    (
      trap 'host_setup_child_cleanup "${idx}" "${semaphore_fd}" "$?"' EXIT
      setup_one_host "${idx}"
      printf '%s|%s|%s\n' \
        "${GPU_COUNTS[$idx]}" \
        "${DATA_IPS[$idx]:-}" \
        "${DATA_INTERFACES[$idx]:-}" \
        > "${result_path}"
    ) > "${log_path}" 2>&1 &
    child_pid="$!"
    HOST_SETUP_PIDS+=("${child_pid}")
    host_jobs+=("${idx}" "${child_pid}")
    set +m
  done

  collect_host_setup_results "${host_jobs[@]}"
  exec {semaphore_fd}>&-
}

retain_ready_hosts() {
  local idx lease_id name
  local -a ready_hosts=() ready_users=() ready_passwords=() ready_data_ips=()
  local -a ready_data_interfaces=()
  local -a ready_gpu_counts=() ready_lease_ids=() ready_lease_names=()

  for idx in "${!HOSTS[@]}"; do
    if [[ "${HOST_READY[$idx]:-0}" == "1" ]]; then
      ready_hosts+=("${HOSTS[$idx]}")
      ready_users+=("${USERS[$idx]}")
      ready_passwords+=("${PASSWORDS[$idx]}")
      ready_data_ips+=("${DATA_IPS[$idx]:-}")
      ready_data_interfaces+=("${DATA_INTERFACES[$idx]:-}")
      ready_gpu_counts+=("${GPU_COUNTS[$idx]}")
      if [[ -n "${COLOSSUS_LEASE_IDS[$idx]:-}" ]]; then
        ready_lease_ids+=("${COLOSSUS_LEASE_IDS[$idx]}")
        ready_lease_names+=("${COLOSSUS_LEASE_NAMES[$idx]}")
      fi
      continue
    fi

    lease_id="${COLOSSUS_LEASE_IDS[$idx]:-}"
    name="${COLOSSUS_LEASE_NAMES[$idx]:-${HOSTS[$idx]}}"
    if [[ -n "${lease_id}" ]]; then
      log "${name}: releasing lease for a host that did not become SSH/setup ready"
      if ! colossus_cli bm lease delete \
        --lease-id "${lease_id}" \
        --force \
        --release-mode CLEAN \
        --json >/dev/null; then
        log "${name}: WARNING: failed to release Colossus lease ${lease_id}"
      fi
    fi
  done

  (( ${#ready_hosts[@]} > 0 )) || die "no reserved host became SSH/setup ready"
  HOSTS=("${ready_hosts[@]}")
  USERS=("${ready_users[@]}")
  PASSWORDS=("${ready_passwords[@]}")
  DATA_IPS=("${ready_data_ips[@]}")
  DATA_INTERFACES=("${ready_data_interfaces[@]}")
  GPU_COUNTS=("${ready_gpu_counts[@]}")
  HOST_READY=()
  for idx in "${!HOSTS[@]}"; do
    HOST_READY+=("1")
  done
  COLOSSUS_LEASE_IDS=("${ready_lease_ids[@]}")
  COLOSSUS_LEASE_NAMES=("${ready_lease_names[@]}")
  COLOSSUS_NODE_COUNT="${#HOSTS[@]}"
  FAILED_HOSTS=()
  if [[ -z "${JOIN_RAY_ADDRESS}" && "${HEAD_DOMAIN_EXPLICIT}" != "1" ]]; then
    HEAD_DOMAIN="${HOSTS[0]}"
  fi
}

resolve_configured_data_ip() {
  local idx="$1"
  local host="${HOSTS[$idx]}"
  local user="${USERS[$idx]}"
  local password="${PASSWORDS[$idx]}"
  local explicit_ip="${DATA_IPS[$idx]:-}"
  local data_interface="${DATA_INTERFACES[$idx]:-${DATA_INTERFACE}}"
  local ip=""
  local addr_command="ip -o -4 addr show scope global"

  if [[ -n "${data_interface}" ]]; then
    addr_command="ip -o -4 addr show dev $(shell_quote "${data_interface}") scope global"
  fi
  if [[ -n "${explicit_ip}" ]]; then
    if ! remote_sudo_capture "${host}" "${user}" "${password}" \
      "${addr_command} | awk '{split(\$4, parts, \"/\"); print parts[1]}'" \
      | grep -Fxq "${explicit_ip}"; then
      die "configured data IP ${explicit_ip} is not assigned to ${host}${data_interface:+ on ${data_interface}}"
    fi
    printf '%s\n' "${explicit_ip}"
    return 0
  fi
  if [[ -n "${data_interface}" ]]; then
    ip="$(remote_cluster_ip "${host}" "${user}" "${password}" \
      "${addr_command} | awk '{split(\$4, parts, \"/\"); print parts[1]}'" || true)"
    [[ -n "${ip}" ]] || die "could not resolve an IPv4 address on ${host}:${data_interface}"
    printf '%s\n' "${ip}"
    return 0
  fi
  return 1
}

resolve_head_ip() {
  local host="${HOSTS[0]}"
  local user="${USERS[0]}"
  local password="${PASSWORDS[0]}"
  local ip=""

  if [[ -n "${DATA_IPS[0]:-}" || -n "${DATA_INTERFACES[0]:-${DATA_INTERFACE}}" ]]; then
    resolve_configured_data_ip "0"
    return
  fi

  ip="$(remote_cluster_ip "${host}" "${user}" "${password}" "ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i=1; i<=NF; i++) if (\$i==\"src\") {print \$(i+1); exit}}' || true" || true)"
  if [[ -z "${ip}" ]]; then
    ip="$(remote_cluster_ip "${host}" "${user}" "${password}" "hostname -I" || true)"
  fi
  if [[ -z "${ip}" ]]; then
    ip="$(remote_cluster_ip "${host}" "${user}" "${password}" "getent ahostsv4 $(shell_quote "${HEAD_DOMAIN}") | awk '{print \$1}' || true" || true)"
  fi
  [[ -n "${ip}" ]] || die "could not resolve head IP on ${host}"
  printf '%s\n' "${ip}"
}

resolve_node_ip() {
  local idx="$1"
  local head_ip="$2"
  local host="${HOSTS[$idx]}"
  local user="${USERS[$idx]}"
  local password="${PASSWORDS[$idx]}"
  local ip=""
  if [[ -n "${DATA_IPS[$idx]:-}" || -n "${DATA_INTERFACES[$idx]:-${DATA_INTERFACE}}" ]]; then
    NODE_IPS[$idx]="$(resolve_configured_data_ip "${idx}")"
    return
  fi
  ip="$(remote_cluster_ip "${host}" "${user}" "${password}" "ip -4 route get $(shell_quote "${head_ip}") 2>/dev/null | awk '{for (i=1; i<=NF; i++) if (\$i==\"src\") {print \$(i+1); exit}}' || true" || true)"
  if [[ -z "${ip}" ]]; then
    ip="$(remote_cluster_ip "${host}" "${user}" "${password}" "ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i=1; i<=NF; i++) if (\$i==\"src\") {print \$(i+1); exit}}' || true" || true)"
  fi
  if [[ -z "${ip}" ]]; then
    ip="$(remote_cluster_ip "${host}" "${user}" "${password}" "hostname -I" || true)"
  fi
  [[ -n "${ip}" ]] || die "could not resolve node IP on ${host}"
  NODE_IPS[$idx]="${ip}"
}

expected_gpu_count() {
  local idx
  local total=0
  for idx in "${!HOSTS[@]}"; do
    if [[ "${HOST_READY[$idx]:-0}" == "1" ]]; then
      total=$((total + GPU_COUNTS[$idx]))
    fi
  done
  printf '%s\n' "${total}"
}

start_ray_container() {
  local idx="$1"
  local role="$2"
  local head_address="$3"
  local host="${HOSTS[$idx]}"
  local user="${USERS[$idx]}"
  local password="${PASSWORDS[$idx]}"
  local node_ip="${NODE_IPS[$idx]}"
  local gpu_count="${GPU_COUNTS[$idx]}"
  local data_interface="${DATA_INTERFACES[$idx]:-${DATA_INTERFACE}}"
  local nccl_socket_ifname="${NCCL_SOCKET_IFNAME:-${data_interface}}"
  local gloo_socket_ifname="${GLOO_SOCKET_IFNAME:-${data_interface}}"

  log "${host}: starting Ray ${role} container on ${node_ip}"
  remote_sudo_script "${host}" "${user}" "${password}" \
    "${REMOTE_ROOT}/remote_ray_container.sh" \
    "${role}" "${IMAGE}" "${head_address}" "${node_ip}" "${gpu_count}" "${GROOT_DATA_ROOT}" \
    "${data_interface}" "${nccl_socket_ifname}" "${gloo_socket_ifname}" \
    "${NCCL_IB_HCA}" "${NCCL_IB_GID_INDEX}" "${REQUIRE_RDMA}" "${TARGET_GPU_NAME}" \
    "${GPU_TEST_IMAGE}" "${GPU_PROFILE}" "${SECONDARY_GPU_NAME_PATTERN}" "${ETHERNET_ONLY}" \
    "${HEAD_DOMAIN}"
}

configure_ray_firewall() {
  local idx="$1"
  local route_target="$2"
  local host="${HOSTS[$idx]}"
  local user="${USERS[$idx]}"
  local password="${PASSWORDS[$idx]}"
  local data_interface="${DATA_INTERFACES[$idx]:-${DATA_INTERFACE}}"

  log "${host}: allowing fixed Ray TCP ports from its directly connected cluster subnet"
  remote_sudo_bash_lc "${host}" "${user}" "${password}" \
    "if command -v ufw >/dev/null 2>&1 && ufw status | grep -q '^Status: active'; then
      interface=$(shell_quote "${data_interface}")
      if [[ -z \"\${interface}\" ]]; then
        interface=\$(ip -4 route get $(shell_quote "${route_target}") | awk '{for (i=1; i<=NF; i++) if (\$i==\"dev\") {print \$(i+1); exit}}')
      fi
      cidr=\$(ip -4 route show dev \"\${interface}\" proto kernel scope link | awk '\$1 ~ /^[0-9]+\\./ {print \$1; exit}')
      [[ -n \"\${cidr}\" ]] || { echo \"could not resolve cluster CIDR on \${interface}\" >&2; exit 1; }
      for port in 6379 6380:6383 8080 10001 20000:20999 52365; do
        ufw allow from \"\${cidr}\" to any port \"\${port}\" proto tcp comment 'GR00T Ray cluster'
      done
    fi"
}

wait_for_head() {
  local host="${HOSTS[0]}"
  local user="${USERS[0]}"
  local password="${PASSWORDS[0]}"
  log "${host}: waiting for Ray head readiness"
  remote_sudo_bash_lc "${host}" "${user}" "${password}" \
    "for i in \$(seq 1 60); do docker exec ${HEAD_CONTAINER} ray status >/dev/null 2>&1 && exit 0; sleep 2; done; docker logs --tail 200 ${HEAD_CONTAINER}; exit 1"
}

wait_for_worker() {
  local idx="$1"
  local host="${HOSTS[$idx]}"
  local user="${USERS[$idx]}"
  local password="${PASSWORDS[$idx]}"
  log "${host}: waiting for Ray worker container"
  remote_sudo_bash_lc "${host}" "${user}" "${password}" \
    "for i in \$(seq 1 30); do docker ps --filter name=${WORKER_CONTAINER} --filter status=running --format '{{.Names}}' | grep -qx ${WORKER_CONTAINER} && exit 0; sleep 2; done; docker logs --tail 200 ${WORKER_CONTAINER}; exit 1"
}

stop_joined_worker_containers() {
  local idx host user password

  log "stopping worker containers started by this failed join attempt"
  for idx in "${JOINED_WORKER_INDICES[@]}"; do
    host="${HOSTS[$idx]}"
    user="${USERS[$idx]}"
    password="${PASSWORDS[$idx]}"
    if remote_sudo_bash_lc "${host}" "${user}" "${password}" \
      "docker rm -f ${WORKER_CONTAINER} >/dev/null 2>&1 || true"; then
      log "${host}: stopped ${WORKER_CONTAINER}"
    else
      log "${host}: WARNING: failed to stop ${WORKER_CONTAINER}"
    fi
  done
  JOINED_WORKER_INDICES=()
}

verify_existing_ray_head_reachable() {
  local idx host user password

  log "checking existing Ray head reachability at ${JOIN_RAY_ADDRESS}"
  for idx in "${!HOSTS[@]}"; do
    [[ "${HOST_READY[$idx]:-0}" == "1" ]] || continue
    host="${HOSTS[$idx]}"
    user="${USERS[$idx]}"
    password="${PASSWORDS[$idx]}"
    remote_sudo_bash_lc "${host}" "${user}" "${password}" \
      "timeout --kill-after=2s 10s bash -c '</dev/tcp/${JOIN_RAY_HEAD_IP}/${JOIN_RAY_HEAD_PORT}'" \
      || die "${host} cannot reach existing Ray head ${JOIN_RAY_ADDRESS}"
  done
}

wait_for_joined_workers() {
  local expected_ips
  local expected_node_gpus="1"
  local host="${HOSTS[0]}"
  local user="${USERS[0]}"
  local password="${PASSWORDS[0]}"

  expected_ips="$(IFS=,; printf '%s' "${NODE_IPS[*]}")"
  if [[ "${GPU_PROFILE}" == "gb300-rtx-pro-2000" ]]; then
    expected_node_gpus="2"
  fi
  log "${host}: waiting for ${#NODE_IPS[@]} new worker(s) to appear in ${JOIN_RAY_ADDRESS}"
  remote_sudo_bash_lc "${host}" "${user}" "${password}" \
    "for i in \$(seq 1 60); do joined=\$(timeout --kill-after=5s 15s docker exec -e RAY_ADDRESS=$(shell_quote "${JOIN_RAY_ADDRESS}") -e EXPECTED_NODE_IPS=$(shell_quote "${expected_ips}") -i ${WORKER_CONTAINER} python - <<'PY' 2>/dev/null | awk '/^[01]$/ {value=\$1} END {print value}' || true
import os
import ray

ray.init(address=os.environ['RAY_ADDRESS'], logging_level='ERROR')
expected = set(filter(None, os.environ['EXPECTED_NODE_IPS'].split(',')))
nodes = {
    node['NodeManagerAddress']: node
    for node in ray.nodes()
    if node.get('Alive')
}
ready = all(
    ip in nodes
    and float(nodes[ip].get('Resources', {}).get('GPU', 0)) >= ${expected_node_gpus}
    and (
        '${GPU_PROFILE}' != 'gb300-rtx-pro-2000'
        or (
            float(nodes[ip].get('Resources', {}).get('GB300', 0)) >= 1
            and float(nodes[ip].get('Resources', {}).get('RTXPRO2000', 0)) >= 1
        )
    )
    for ip in expected
)
print(1 if ready else 0)
ray.shutdown()
PY
); if [[ \"\${joined}\" == \"1\" ]]; then exit 0; fi; sleep 2; done; docker exec ${WORKER_CONTAINER} ray status --address=$(shell_quote "${JOIN_RAY_ADDRESS}"); exit 1"
}

wait_for_cluster_gpus() {
  local expected_gpus="$1"
  local host="${HOSTS[0]}"
  local user="${USERS[0]}"
  local password="${PASSWORDS[0]}"

  log "${host}: waiting for Ray cluster to report ${expected_gpus} profile-backed GPU(s)"
  remote_sudo_bash_lc "${host}" "${user}" "${password}" \
    "for i in \$(seq 1 60); do actual=\$(docker exec -i ${HEAD_CONTAINER} python - <<'PY' 2>/dev/null | awk '/^[0-9]+$/ {value=\$1} END {print value}' || true
import ray
ray.init(address='auto', logging_level='ERROR')
resources = ray.cluster_resources()
gpu = int(resources.get('GPU', 0))
if '${GPU_PROFILE}' == 'gb300-rtx-pro-2000':
    ready = (
        int(resources.get('GB300', 0)) * 2 == gpu
        and int(resources.get('RTXPRO2000', 0)) * 2 == gpu
    )
    print(gpu if ready else -1)
else:
    print(gpu)
PY
); if [[ \"\${actual}\" =~ ^[0-9]+$ && \"\${actual}\" -ge ${expected_gpus} ]]; then exit 0; fi; sleep 2; done; docker exec ${HEAD_CONTAINER} ray status; exit 1"
}

run_transport_validation() {
  if [[ -z "${TRANSPORT_VALIDATION_COMMAND}" ]]; then
    return
  fi

  local host="${HOSTS[0]}"
  local user="${USERS[0]}"
  local password="${PASSWORDS[0]}"
  local command container="${HEAD_CONTAINER}"
  command="timeout --kill-after=10s ${TRANSPORT_VALIDATION_TIMEOUT}s docker exec"
  command+=" -e CUDA_VISIBLE_DEVICES=0 -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET"
  if [[ -n "${JOIN_RAY_ADDRESS}" ]]; then
    container="${WORKER_CONTAINER}"
    command+=" -e RAY_ADDRESS=$(shell_quote "${JOIN_RAY_ADDRESS}")"
  fi
  command+=" ${container} podman run --rm"
  command+=" $(shell_quote "${TRANSPORT_VALIDATION_IMAGE:-${GPU_TEST_IMAGE}}")"
  command+=" bash -lc $(shell_quote "${TRANSPORT_VALIDATION_COMMAND}")"
  log "${host}: running configured transport validation command"
  remote_sudo_bash_lc "${host}" "${user}" "${password}" "${command}"
}

configure_nginx_on_head() {
  local host="${HOSTS[0]}"
  local user="${USERS[0]}"
  local password="${PASSWORDS[0]}"

  if [[ -n "${LOCAL_SSL_CERT}" ]]; then
    log "${host}: installing caller-provided TLS certificate and key"
    REMOTE_TLS_CERT_STAGING="${REMOTE_ROOT}/ray-tls-cert-${TMP_DIR##*/}.pem"
    REMOTE_TLS_KEY_STAGING="${REMOTE_ROOT}/ray-tls-key-${TMP_DIR##*/}.pem"
    # Arm local cleanup before the first upload so failures or caught signals
    # cannot strand caller-provided key material before the remote trap starts.
    REMOTE_TLS_STAGING_ACTIVE="1"
    remote_scp_to "${host}" "${user}" "${password}" \
      "${LOCAL_SSL_CERT}" "${REMOTE_TLS_CERT_STAGING}"
    remote_scp_to "${host}" "${user}" "${password}" \
      "${LOCAL_SSL_KEY}" "${REMOTE_TLS_KEY_STAGING}"
    remote_ssh "${host}" "${user}" "${password}" \
      "chmod 600 $(shell_quote "${REMOTE_TLS_CERT_STAGING}") $(shell_quote "${REMOTE_TLS_KEY_STAGING}")"
    remote_sudo_bash_lc "${host}" "${user}" "${password}" \
      "cleanup_tls_staging() { rm -f -- $(shell_quote "${REMOTE_TLS_CERT_STAGING}") $(shell_quote "${REMOTE_TLS_KEY_STAGING}"); }; trap cleanup_tls_staging EXIT; install -D -m 0644 $(shell_quote "${REMOTE_TLS_CERT_STAGING}") $(shell_quote "${SSL_CERT}") && install -D -m 0600 $(shell_quote "${REMOTE_TLS_KEY_STAGING}") $(shell_quote "${SSL_KEY}")"
    cleanup_remote_tls_staging
  fi

  log "${host}: configuring nginx basic-auth proxy at https://${HEAD_DOMAIN}/ray/"
  remote_scp_to "${host}" "${user}" "${password}" "${TMP_DIR}/remote_nginx_ray.sh" "${REMOTE_ROOT}/remote_nginx_ray.sh"
  remote_scp_to "${host}" "${user}" "${password}" "${TMP_DIR}/ray-auth-password" "${REMOTE_ROOT}/ray-auth-password"
  remote_ssh "${host}" "${user}" "${password}" "chmod +x $(shell_quote "${REMOTE_ROOT}/remote_nginx_ray.sh") && chmod 600 $(shell_quote "${REMOTE_ROOT}/ray-auth-password")"
  remote_sudo_script "${host}" "${user}" "${password}" \
    "${REMOTE_ROOT}/remote_nginx_ray.sh" \
    "${HEAD_DOMAIN}" "${SSL_CERT}" "${SSL_KEY}" "${RAY_AUTH_USER}" "${REMOTE_ROOT}/ray-auth-password"
}

start_ray_metrics_on_head() {
  local host="${HOSTS[0]}"
  local user="${USERS[0]}"
  local password="${PASSWORDS[0]}"

  log "${host}: starting localhost-only Prometheus and Grafana sidecars"
  remote_sudo_script "${host}" "${user}" "${password}" \
    "${REMOTE_ROOT}/remote_ray_metrics.sh" \
    "${HEAD_DOMAIN}" "${PROMETHEUS_IMAGE}" "${GRAFANA_IMAGE}" "${GROOT_DATA_ROOT}"
}

start_ray_cluster() {
  if [[ "${HOST_READY[0]:-0}" != "1" ]]; then
    die "head host ${HOSTS[0]} is not ready; refusing to promote another host"
  fi

  local head_ip
  head_ip="$(resolve_head_ip)"
  ACTIVE_RAY_HEAD_IP="${head_ip}"
  log "head Ray address: ${head_ip}:${RAY_PORT}"

  local idx
  for idx in "${!HOSTS[@]}"; do
    if [[ "${HOST_READY[$idx]:-0}" == "1" ]]; then
      resolve_node_ip "${idx}" "${head_ip}"
      if [[ "${idx}" == "0" ]]; then
        configure_ray_firewall "${idx}" "1.1.1.1"
      else
        configure_ray_firewall "${idx}" "${head_ip}"
      fi
    fi
  done

  start_ray_container "0" "head" "${head_ip}:${RAY_PORT}"
  wait_for_head
  configure_nginx_on_head
  start_ray_metrics_on_head

  for idx in "${!HOSTS[@]}"; do
    if [[ "${idx}" == "0" ]]; then
      continue
    fi
    if [[ "${HOST_READY[$idx]:-0}" == "1" ]]; then
      start_ray_container "${idx}" "worker" "${head_ip}:${RAY_PORT}"
      wait_for_worker "${idx}"
    fi
  done

  wait_for_cluster_gpus "$(expected_gpu_count)"
  run_transport_validation

  local host="${HOSTS[0]}"
  local user="${USERS[0]}"
  local password="${PASSWORDS[0]}"
  log "${host}: final Ray status"
  remote_sudo_bash_lc "${host}" "${user}" "${password}" "docker exec ${HEAD_CONTAINER} ray status"
}

start_ray_workers() {
  local idx

  ACTIVE_RAY_HEAD_IP="${JOIN_RAY_HEAD_IP}"

  for idx in "${!HOSTS[@]}"; do
    if [[ "${HOST_READY[$idx]:-0}" == "1" ]]; then
      resolve_node_ip "${idx}" "${JOIN_RAY_HEAD_IP}"
      configure_ray_firewall "${idx}" "${JOIN_RAY_HEAD_IP}"
    fi
  done
  verify_existing_ray_head_reachable

  for idx in "${!HOSTS[@]}"; do
    if [[ "${HOST_READY[$idx]:-0}" == "1" ]]; then
      # Track the attempt first so cleanup covers an ambiguous SSH failure
      # after the remote container has started.
      JOINED_WORKER_INDICES+=("${idx}")
      start_ray_container "${idx}" "worker" "${JOIN_RAY_ADDRESS}"
      wait_for_worker "${idx}"
    fi
  done

  wait_for_joined_workers
  run_transport_validation

  local host="${HOSTS[0]}"
  local user="${USERS[0]}"
  local password="${PASSWORDS[0]}"
  log "${host}: existing cluster status after adding workers"
  remote_sudo_bash_lc "${host}" "${user}" "${password}" \
    "docker exec ${WORKER_CONTAINER} ray status --address=$(shell_quote "${JOIN_RAY_ADDRESS}")"
}

write_cluster_state() {
  [[ -n "${STATE_OUTPUT}" ]] || return 0
  local hosts node_ips data_interfaces lease_ids lease_names
  hosts="$(IFS=,; printf '%s' "${HOSTS[*]}")"
  node_ips="$(IFS=,; printf '%s' "${NODE_IPS[*]}")"
  data_interfaces="$(IFS=,; printf '%s' "${DATA_INTERFACES[*]}")"
  lease_ids="$(IFS=,; printf '%s' "${COLOSSUS_LEASE_IDS[*]}")"
  lease_names="$(IFS=,; printf '%s' "${COLOSSUS_LEASE_NAMES[*]}")"
  python3 - \
    "${STATE_OUTPUT}" \
    "${ACTIVE_RAY_HEAD_IP}" \
    "${RAY_PORT}" \
    "${HEAD_DOMAIN}" \
    "${GPU_PROFILE}" \
    "${hosts}" \
    "${node_ips}" \
    "${data_interfaces}" \
    "${lease_ids}" \
    "${lease_names}" \
    "${COLOSSUS_EXISTING_LEASE_ID}" <<'PY'
import json
from pathlib import Path
import sys

(
    output,
    head_ip,
    ray_port,
    head_domain,
    gpu_profile,
    hosts,
    node_ips,
    data_interfaces,
    lease_ids,
    lease_names,
    existing_lease_id,
) = sys.argv[1:]


def split(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


payload = {
    "head_ip": head_ip,
    "ray_address": f"{head_ip}:{ray_port}",
    "head_domain": head_domain,
    "gpu_profile": gpu_profile,
    "hosts": split(hosts),
    "node_ips": split(node_ips),
    "data_interfaces": data_interfaces.split(","),
    "created_lease_ids": split(lease_ids),
    "created_lease_names": split(lease_names),
    "existing_lease_id": existing_lease_id,
}
path = Path(output)
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_name(f".{path.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
  chmod 0600 "${STATE_OUTPUT}"
  log "wrote cluster state to ${STATE_OUTPUT}"
}

main() {
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  parse_args "$@"
  validate_args
  require_cmd colossus flock python3
  TMP_DIR="$(mktemp -d)"
  if [[ -n "${COLOSSUS_EXISTING_LEASE_ID}" ]]; then
    load_existing_colossus_host "0"
  else
    discover_colossus_candidates
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    if [[ -z "${COLOSSUS_EXISTING_LEASE_ID}" ]]; then
      prepare_dry_run_hosts
    fi
    print_dry_run
    return 0
  fi

  resolve_secret_sources
  require_cmd sshpass ssh scp

  write_remote_scripts
  create_local_secret_files
  if [[ -n "${COLOSSUS_EXISTING_LEASE_ID}" ]]; then
    HOSTS=()
    USERS=()
    PASSWORDS=()
    DATA_IPS=()
    DATA_INTERFACES=()
    load_existing_colossus_host "1"
  else
    reserve_colossus_hosts
  fi
  setup_hosts
  if [[ "${#FAILED_HOSTS[@]}" -gt 0 ]]; then
    if [[ "${BEST_EFFORT_HOSTS}" == "1" && -z "${COLOSSUS_EXISTING_LEASE_ID}" ]]; then
      log "host setup failed for ${FAILED_HOSTS[*]}; retaining only ready hosts"
      retain_ready_hosts
    else
      die "host setup failed for ${FAILED_HOSTS[*]}; refusing to start a partial Ray cluster"
    fi
  fi
  if [[ -n "${JOIN_RAY_ADDRESS}" ]]; then
    start_ray_workers
  else
    start_ray_cluster
  fi
  write_cluster_state
  COLOSSUS_LEASES_COMMITTED="1"

  if [[ -n "${JOIN_RAY_ADDRESS}" ]]; then
    log "added ${#HOSTS[@]} Colossus worker(s) to existing Ray cluster ${JOIN_RAY_ADDRESS}"
    log "existing Ray head and endpoint were not reconfigured"
  else
    log "Ray endpoint: https://${HEAD_DOMAIN}/ray/"
  fi
  if [[ -n "${COLOSSUS_EXISTING_LEASE_ID}" ]]; then
    log "Colossus lease: ${COLOSSUS_EXISTING_LEASE_ID} (${COLOSSUS_EXISTING_LEASE_NAME}, reused)"
  fi
  local idx
  for idx in "${!COLOSSUS_LEASE_IDS[@]}"; do
    log "Colossus lease: ${COLOSSUS_LEASE_IDS[$idx]} (${COLOSSUS_LEASE_NAMES[$idx]})"
  done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
