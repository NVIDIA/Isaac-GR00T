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
# Build a persistent Ray cluster across bare-metal GR00T GPU hosts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_SETUP_SCRIPT="${SCRIPT_DIR}/setup_bare_metal_gpu_host.sh"

IMAGE="${GROOT_CI_TRAIN_IMAGE:-nvcr.io/nvidian/groot-ci-train:c7a32a6d}"
INVENTORY=""
SSL_CERT="${SSL_CERT:-/etc/ssl/groot-ray/fullchain.pem}"
SSL_KEY="${SSL_KEY:-/etc/ssl/groot-ray/privkey.pem}"
HEAD_DOMAIN=""
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
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-}"
NCCL_IB_HCA="${NCCL_IB_HCA:-}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-}"
REQUIRE_RDMA="${REQUIRE_RDMA:-0}"
TRANSPORT_VALIDATION_COMMAND="${TRANSPORT_VALIDATION_COMMAND:-}"
TRANSPORT_VALIDATION_TIMEOUT="${TRANSPORT_VALIDATION_TIMEOUT:-300}"

readonly HEAD_CONTAINER="groot-ray-head"
readonly WORKER_CONTAINER="groot-ray-worker"
readonly REMOTE_ROOT="/tmp/groot-ray-cluster"
readonly RAY_PORT="6379"
readonly TARGET_GPU_NAME="NVIDIA GB300"

TMP_DIR=""
declare -a HOSTS=()
declare -a USERS=()
declare -a PASSWORDS=()
declare -a HOST_READY=()
declare -a GPU_COUNTS=()
declare -a NODE_IPS=()
declare -a DATA_IPS=()
declare -a FAILED_HOSTS=()
declare -a HOST_SETUP_PIDS=()

log() {
  printf '[setup_ray_cluster] %s\n' "$*"
}

die() {
  printf '[setup_ray_cluster] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage:
  ci/scripts/setup_ray_cluster.sh \\
    --inventory hosts.csv \\
    --ray-auth-password-file /path/to/password \\
    --nvcr-token-file /path/to/nvcr-token

Options:
  --inventory PATH              CSV with host,user,password[,data_ip] rows. First row is Ray head.
  --image IMAGE                 Docker image to pull and run. Default: ${IMAGE}
  --ssl-cert PATH               TLS certificate path on the head host. Default: ${SSL_CERT}
  --ssl-key PATH                TLS private key path on the head host. Default: ${SSL_KEY}
  --head-domain DOMAIN          Public HTTPS domain. Default: first inventory host.
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
  --bucket-access-key-id-env VAR
                                Env var containing S3 access key. Default: ${BUCKET_ACCESS_KEY_ID_ENV}
  --bucket-secret-access-key-env VAR
                                Env var containing S3 secret key. Default: ${BUCKET_SECRET_ACCESS_KEY_ENV}
  --bucket-endpoint-url-env VAR Env var containing S3 endpoint URL. Default: ${BUCKET_ENDPOINT_URL_ENV}
  --bucket-region-env VAR       Env var containing S3 region. Default: ${BUCKET_REGION_ENV}
  --skip-post-training-artifacts
                                Do not mirror model deps/checkpoint artifacts or inject data creds.
  --ssh-connect-timeout SECS    SSH connect timeout. Default: ${SSH_CONNECT_TIMEOUT}
  --reboot-wait-timeout SECS    Overall seconds to wait for a reboot. Default: ${REBOOT_WAIT_TIMEOUT}
  --reboot-poll-interval SECS   Seconds between SSH reboot polls. Default: ${REBOOT_POLL_INTERVAL}
  --host-setup-concurrency N    Maximum hosts to prepare concurrently. Default: ${HOST_SETUP_CONCURRENCY}
  --data-interface IFACE       Data-plane netdev used by Ray, NCCL bootstrap, and Gloo.
  --nccl-socket-ifname IFACE   NCCL socket netdev. Default: --data-interface value.
  --gloo-socket-ifname IFACE   Gloo socket netdev. Default: --data-interface value.
  --nccl-ib-hca HCA[:PORT]     Verbs HCA/port list for NCCL, for example mlx5_3:1.
                                Matching uverbs devices are derived from sysfs and passed to Docker.
  --nccl-ib-gid-index INDEX    RoCE GID index used by NCCL, for example 3.
  --require-rdma               Fail unless the selected HCA/GID and container RDMA path are usable.
  --allow-socket-transport     Permit legacy TCP-only operation. This is the default.
  --transport-validation-command COMMAND
                                Run COMMAND in the head container after all GPUs join Ray; a
                                nonzero exit prevents the cluster from being declared ready.
  --transport-validation-timeout SECS
                                Bound the validation command. Default: ${TRANSPORT_VALIDATION_TIMEOUT}
  --dry-run                     Print the plan without changing any host.
  -h, --help                    Show this help.

GPU policy:
  Every host must contain exactly one ${TARGET_GPU_NAME}. Only that GPU is
  exposed to its Ray container and advertised as one generic GPU resource.

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
  if [[ -n "${TMP_DIR}" && -d "${TMP_DIR}" ]]; then
    rm -rf "${TMP_DIR}" || true
  fi
  return "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

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
      --inventory)
        require_value "$1" "${2:-}"
        INVENTORY="$2"
        shift 2
        ;;
      --image)
        require_value "$1" "${2:-}"
        IMAGE="$2"
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
      --head-domain)
        require_value "$1" "${2:-}"
        HEAD_DOMAIN="$2"
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
      --transport-validation-command)
        require_value "$1" "${2:-}"
        TRANSPORT_VALIDATION_COMMAND="$2"
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
  [[ -n "${INVENTORY}" ]] || die "--inventory is required"
  [[ -r "${INVENTORY}" ]] || die "cannot read inventory: ${INVENTORY}"
  [[ -x "${HOST_SETUP_SCRIPT}" ]] || die "missing executable host setup script: ${HOST_SETUP_SCRIPT}"
  [[ -n "${SSL_CERT}" ]] || die "--ssl-cert is required"
  [[ -n "${SSL_KEY}" ]] || die "--ssl-key is required"
  [[ -n "${RAY_AUTH_USER}" ]] || die "--ray-auth-user is required"

  case "${RAY_AUTH_USER}" in
    *:*|*$'\n'*|*$'\r'*) die "--ray-auth-user cannot contain ':' or newlines" ;;
  esac
  case "${SSL_CERT}${SSL_KEY}" in
    *[[:space:]]*) die "--ssl-cert and --ssl-key paths cannot contain whitespace" ;;
  esac
  case "${GROOT_DATA_ROOT}" in
    ""|*[[:space:]]*) die "--groot-data-root must be non-empty and cannot contain whitespace" ;;
  esac
  case "${SYNC_POST_TRAINING_ARTIFACTS}" in
    0|1) ;;
    *) die "SYNC_POST_TRAINING_ARTIFACTS must be 0 or 1" ;;
  esac
  if [[ "${SYNC_POST_TRAINING_ARTIFACTS}" == "1" ]]; then
    [[ "${MODEL_DEPS_URI}" == s3://* ]] || die "--model-deps-uri must be an s3:// URI"
    if [[ -n "${POST_TRAINING_CHECKPOINT_URI}" ]]; then
      [[ "${POST_TRAINING_CHECKPOINT_URI}" == s3://* ]] || die "--post-training-checkpoint-uri must be an s3:// URI"
    elif [[ "${DRY_RUN}" == "0" ]]; then
      die "provide --post-training-checkpoint-uri as an s3:// URI"
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
  local interface
  for interface in "${DATA_INTERFACE}" "${NCCL_SOCKET_IFNAME}" "${GLOO_SOCKET_IFNAME}"; do
    if [[ -n "${interface}" && ! "${interface}" =~ ^[[:alnum:]_.:-]+$ ]]; then
      die "network interface names may contain only letters, digits, '.', '_', ':', and '-'"
    fi
  done
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
    [[ -n "${RAY_AUTH_PASSWORD_SOURCE}" ]] || die "provide a Ray auth password via --ray-auth-password-file/env/stdin"
    [[ -n "${NVCR_TOKEN_SOURCE}" || -n "${NVCR_TOKEN}" ]] || die "provide an NVCR token via --nvcr-token-file/env/stdin or NVCR_TOKEN"
  fi
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
    [[ -n "${RAY_AUTH_PASSWORD}" ]] || die "Ray auth password is empty"
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

parse_inventory() {
  HOSTS=()
  USERS=()
  PASSWORDS=()
  DATA_IPS=()

  eval "$(python3 - "${INVENTORY}" <<'PY'
import csv
import ipaddress
import shlex
import sys

path = sys.argv[1]
first_record = True

with open(path, newline="") as f:
    reader = csv.reader(f)
    for lineno, row in enumerate(reader, start=1):
        if not row or all(not cell.strip() for cell in row):
            continue
        if row[0].lstrip().startswith("#"):
            continue
        normalized = [cell.strip().lower() for cell in row[:4]]
        if first_record and normalized[:3] == ["host", "user", "password"]:
            if len(normalized) == 4 and normalized[3] not in {"", "data_ip"}:
                raise SystemExit(f"{path}:{lineno}: fourth inventory header must be data_ip")
            first_record = False
            continue
        first_record = False
        if len(row) < 3:
            raise SystemExit(f"{path}:{lineno}: expected host,user,password")
        host = row[0].strip()
        user = row[1].strip()
        password = row[2]
        data_ip = row[3].strip() if len(row) >= 4 else ""
        if not host or not user or password == "":
            raise SystemExit(f"{path}:{lineno}: host, user, and password must be non-empty")
        if data_ip:
            try:
                address = ipaddress.ip_address(data_ip)
            except ValueError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid data_ip {data_ip!r}: {exc}") from exc
            if address.version != 4 or address.is_unspecified or address.is_loopback or address.is_link_local:
                raise SystemExit(f"{path}:{lineno}: data_ip must be a usable IPv4 address")
        print(f"HOSTS+=({shlex.quote(host)})")
        print(f"USERS+=({shlex.quote(user)})")
        print(f"PASSWORDS+=({shlex.quote(password)})")
        print(f"DATA_IPS+=({shlex.quote(data_ip)})")
PY
)"

  [[ "${#HOSTS[@]}" -gt 0 ]] || die "inventory has no hosts"
  if [[ -z "${HEAD_DOMAIN}" ]]; then
    HEAD_DOMAIN="${HOSTS[0]}"
  fi
}

print_dry_run() {
  log "dry-run only; no host will be changed"
  log "image: ${IMAGE}"
  log "head domain: ${HEAD_DOMAIN}"
  log "ssl cert on head: ${SSL_CERT}"
  log "ssl key on head: ${SSL_KEY}"
  log "nginx endpoint: https://${HEAD_DOMAIN}/ray/"
  log "ray auth user: ${RAY_AUTH_USER}"
  log "host setup concurrency: ${HOST_SETUP_CONCURRENCY}"
  log "GPU isolation: one NVIDIA GB300 per Ray node"
  log "data interface: ${DATA_INTERFACE:-<automatic>}"
  log "NCCL socket interface: ${NCCL_SOCKET_IFNAME:-<automatic>}"
  log "Gloo socket interface: ${GLOO_SOCKET_IFNAME:-<automatic>}"
  log "RDMA required: ${REQUIRE_RDMA}"
  if [[ "${REQUIRE_RDMA}" == "1" ]]; then
    log "NCCL IB HCA: ${NCCL_IB_HCA}"
    log "NCCL IB GID index: ${NCCL_IB_GID_INDEX}"
  fi
  if [[ -n "${TRANSPORT_VALIDATION_COMMAND}" ]]; then
    log "transport validation command: configured (timeout ${TRANSPORT_VALIDATION_TIMEOUT}s)"
  else
    log "transport validation command: disabled"
  fi
  log "groot data root: ${GROOT_DATA_ROOT}"
  if [[ "${SYNC_POST_TRAINING_ARTIFACTS}" == "1" ]]; then
    log "post-training artifacts: enabled"
    log "model deps source: ${MODEL_DEPS_URI}"
    log "post-training checkpoint source: ${POST_TRAINING_CHECKPOINT_URI:-<required for real run>}"
    log "bucket credential env vars: ${BUCKET_ACCESS_KEY_ID_ENV}, ${BUCKET_SECRET_ACCESS_KEY_ENV}, ${BUCKET_ENDPOINT_URL_ENV}, ${BUCKET_REGION_ENV}"
  else
    log "post-training artifacts: disabled"
  fi
  log "managed containers: ${HEAD_CONTAINER}, ${WORKER_CONTAINER}"
  local i role
  for i in "${!HOSTS[@]}"; do
    role="worker"
    if [[ "${i}" == "0" ]]; then
      role="head"
    fi
    log "inventory[$i]: ${role} ${USERS[$i]}@${HOSTS[$i]} data_ip=${DATA_IPS[$i]:-<automatic>}"
  done
}

ssh_opts() {
  printf '%s\n' \
    "-o" "StrictHostKeyChecking=accept-new" \
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
head_ip="$3"
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

head_container="groot-ray-head"
worker_container="groot-ray-worker"
volume="groot_ray_tmp"

case "${require_rdma}" in
  0|1) ;;
  *) echo "invalid require_rdma=${require_rdma}" >&2; exit 1 ;;
esac
[[ "${gpu_count}" == "1" ]] || {
  echo "expected exactly one GB300 GPU per Ray node, got gpu_count=${gpu_count}" >&2
  exit 1
}
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
declare -a docker_gpu_args=(--gpus "device=${gb300_gpu_uuid}")
ray_num_gpus="1"

if [[ -n "${data_interface}" ]]; then
  ip link show dev "${data_interface}" >/dev/null
  ip -o -4 addr show dev "${data_interface}" scope global | grep -q '[[:space:]]inet[[:space:]]'
fi

declare -a rdma_device_args=()
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
      rdma_device_args+=(--device "${uverbs}:${uverbs}")
    fi
  done
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
fi

declare -a container_fips_args=()
container_fips_mask="/var/lib/groot-ray/fips-disabled"
if [[ "$(head -c 1 /proc/sys/crypto/fips_enabled 2>/dev/null || true)" == "1" ]]; then
  # The GR00T training image is not FIPS-certified. Mask the host kernel flag
  # inside the container so its bundled crypto libraries do not enter an
  # unsupported FIPS mode and abort during their self-tests.
  install -d -m 0755 "$(dirname "${container_fips_mask}")"
  printf '0\n' > "${container_fips_mask}"
  chmod 0444 "${container_fips_mask}"
  container_fips_args+=(
    -v "${container_fips_mask}:/proc/sys/crypto/fips_enabled:ro"
  )
fi

common_ray_args=(
  "--node-ip-address=${node_ip}"
  "--node-manager-port=6380"
  "--object-manager-port=6381"
  "--runtime-env-agent-port=6382"
  "--dashboard-agent-grpc-port=6383"
  "--dashboard-agent-listen-port=52365"
  "--min-worker-port=20000"
  "--max-worker-port=20100"
  "--num-gpus=${ray_num_gpus}"
  "--disable-usage-stats"
  "--block"
)

if [[ "${role}" == "head" ]]; then
  container="${head_container}"
  ray_cmd="ray stop --force >/dev/null 2>&1 || true; exec ray start --head --port=6379 --dashboard-host=127.0.0.1 --dashboard-port=8265 --ray-client-server-port=10001 --include-dashboard=True ${common_ray_args[*]}"
else
  container="${worker_container}"
  ray_cmd="ray stop --force >/dev/null 2>&1 || true; exec ray start --address=${head_ip}:6379 ${common_ray_args[*]}"
fi

docker rm -f "${head_container}" "${worker_container}" >/dev/null 2>&1 || true
docker volume create "${volume}" >/dev/null
docker run -d \
  --name "${container}" \
  --restart unless-stopped \
  --network host \
  "${docker_gpu_args[@]}" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  "${rdma_device_args[@]}" \
  "${network_env_args[@]}" \
  "${container_fips_args[@]}" \
  -v "${volume}:/tmp/ray" \
  -v "${data_root}:${data_root}" \
  --env-file /tmp/groot-ray-cluster/data-credentials.env \
  -w /workspace \
  "${image}" \
  bash -lc "${ray_cmd}"

if ! visible_gpu_count="$(
  timeout --kill-after=5s 20s docker exec "${container}" nvidia-smi -L \
    | wc -l \
    | tr -d '[:space:]'
)"; then
  echo "failed to query GPU visibility in ${container}" >&2
  docker rm -f "${container}" >/dev/null 2>&1 || true
  exit 1
fi
if [[ "${visible_gpu_count}" != "${gpu_count}" ]]; then
  echo "${container} sees ${visible_gpu_count} GPU(s), expected ${gpu_count}" >&2
  docker rm -f "${container}" >/dev/null 2>&1 || true
  exit 1
fi
if ! visible_gpu_uuid="$(
  timeout --kill-after=5s 20s docker exec "${container}" nvidia-smi \
    --query-gpu=uuid --format=csv,noheader \
    | tr -d '[:space:]'
)"; then
  echo "failed to query the GPU UUID in ${container}" >&2
  docker rm -f "${container}" >/dev/null 2>&1 || true
  exit 1
fi
if [[ "${visible_gpu_uuid}" != "${gb300_gpu_uuid}" ]]; then
  echo "${container} sees GPU ${visible_gpu_uuid}, expected ${target_gpu_name} ${gb300_gpu_uuid}" >&2
  docker rm -f "${container}" >/dev/null 2>&1 || true
  exit 1
fi
if ! visible_gpu_name="$(
  timeout --kill-after=5s 20s docker exec "${container}" nvidia-smi \
    --query-gpu=name --format=csv,noheader \
    | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
)"; then
  echo "failed to query the GPU model in ${container}" >&2
  docker rm -f "${container}" >/dev/null 2>&1 || true
  exit 1
fi
if [[ "${visible_gpu_name}" != "${target_gpu_name}" ]]; then
  echo "${container} sees GPU model ${visible_gpu_name}, expected ${target_gpu_name}" >&2
  docker rm -f "${container}" >/dev/null 2>&1 || true
  exit 1
fi

if [[ "${require_rdma}" == "1" ]]; then
  for uverbs in "${uverbs_devices[@]}"; do
    docker exec "${container}" test -c "${uverbs}"
  done
  docker exec "${container}" python -c \
    'import ctypes; ctypes.CDLL("libibverbs.so.1")'
  docker exec "${container}" env | grep -Fxq "NCCL_IB_DISABLE=0"
  docker exec "${container}" env | grep -Fxq "NCCL_IB_HCA=${nccl_ib_hca}"
  docker exec "${container}" env | grep -Fxq "NCCL_IB_GID_INDEX=${nccl_ib_gid_index}"
  docker exec "${container}" env | grep -Fxq "NCCL_SOCKET_IFNAME=${nccl_socket_ifname}"
  docker exec "${container}" env | grep -Fxq "GLOO_SOCKET_IFNAME=${gloo_socket_ifname}"
fi
if [[ "${#container_fips_args[@]}" -gt 0 ]]; then
  docker exec "${container}" sh -c \
    'test "$(head -c 1 /proc/sys/crypto/fips_enabled)" = "0"'
fi
docker exec "${container}" python -c \
  'import ssl; ssl.create_default_context(); import av, cv2; from transformers import ProcessorMixin'
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
  --env-file "${env_file}" \
  "${image}" \
  python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path
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
    if target.exists() and target.stat().st_size > 0:
        if expected_size is None or target.stat().st_size == expected_size:
            return
    log(f"downloading s3://{bucket}/{key} -> {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    response = s3.get_object(Bucket=bucket, Key=key)
    with tmp.open("wb") as file:
        for chunk in response["Body"].iter_chunks(chunk_size=16 * 1024 * 1024):
            if chunk:
                file.write(chunk)
    tmp.replace(target)
    if expected_size is not None and target.stat().st_size != expected_size:
        raise OSError(
            f"Downloaded size mismatch for s3://{bucket}/{key}: "
            f"expected {expected_size}, got {target.stat().st_size}"
        )


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


def main() -> None:
    root = Path(required_env("GROOT_DATA_ROOT"))
    model_deps = required_env("GROOT_MODEL_DEPS_URI").rstrip("/")
    post_training_checkpoint = required_env("GROOT_POST_TRAINING_CHECKPOINT_URI")
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
        root / "checkpoints/post_training_golden_run/checkpoint-20000",
    )
    log("post-training artifacts are ready")


main()
PY
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
    } > "${TMP_DIR}/data-credentials.env"
    chmod 0600 "${TMP_DIR}/data-credentials.env"
  else
    : > "${TMP_DIR}/data-credentials.env"
    chmod 0600 "${TMP_DIR}/data-credentials.env"
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
  remote_scp_to "${host}" "${user}" "${password}" "${TMP_DIR}/remote_ray_container.sh" "${REMOTE_ROOT}/remote_ray_container.sh" || return 1
  remote_scp_to "${host}" "${user}" "${password}" "${TMP_DIR}/remote_post_training_artifacts.sh" "${REMOTE_ROOT}/remote_post_training_artifacts.sh" || return 1
  remote_scp_to "${host}" "${user}" "${password}" "${TMP_DIR}/nvcr-token" "${REMOTE_ROOT}/nvcr-token" || return 1
  remote_scp_to "${host}" "${user}" "${password}" "${TMP_DIR}/data-credentials.env" "${REMOTE_ROOT}/data-credentials.env" || return 1
  remote_ssh "${host}" "${user}" "${password}" "chmod 700 $(shell_quote "${REMOTE_ROOT}") && chmod 600 $(shell_quote "${REMOTE_ROOT}/nvcr-token") $(shell_quote "${REMOTE_ROOT}/data-credentials.env") && chmod +x $(shell_quote "${REMOTE_ROOT}/setup_bare_metal_gpu_host.sh") $(shell_quote "${REMOTE_ROOT}/remote_ray_container.sh") $(shell_quote "${REMOTE_ROOT}/remote_post_training_artifacts.sh")" || return 1
}

run_host_setup() {
  local idx="$1"
  local host="${HOSTS[$idx]}"
  local user="${USERS[$idx]}"
  local password="${PASSWORDS[$idx]}"
  local -a setup_args=(
    "--validate-groot-image"
    "--nvidia-test-image" "${IMAGE}"
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
    "${IMAGE}" "${GROOT_DATA_ROOT}" "${REMOTE_ROOT}/data-credentials.env"
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
    | awk -F',' -v target="${TARGET_GPU_NAME}" '
        {
          uuid=$1; name=$2
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", uuid)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
          if (name == target && uuid ~ /^GPU-[[:xdigit:]-]+$/) {
            count++
          }
        }
        END { exit !(count == 1) }
      '; then
    log "${host}: expected exactly one healthy ${TARGET_GPU_NAME}"
    return 1
  fi
  gpu_count="1"

  remote_sudo_bash_lc "${host}" "${user}" "${password}" \
    "docker image inspect $(shell_quote "${IMAGE}") >/dev/null" || {
    log "${host}: prepared image is unavailable"
    return 1
  }

  HOST_READY[$idx]="1"
  GPU_COUNTS[$idx]="${gpu_count}"
  log "${host}: ready with one isolated ${TARGET_GPU_NAME}"
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
  local idx pid host result_path log_path gpu_count wait_status
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
      gpu_count="$(tr -d '[:space:]' < "${result_path}")"
      if [[ "${gpu_count}" =~ ^[0-9]+$ && "${gpu_count}" -gt 0 ]]; then
        HOST_READY[$idx]="1"
        GPU_COUNTS[$idx]="${gpu_count}"
        log "${host}: ready with ${gpu_count} GPU(s)"
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
      printf '%s\n' "${GPU_COUNTS[$idx]}" > "${result_path}"
    ) > "${log_path}" 2>&1 &
    child_pid="$!"
    HOST_SETUP_PIDS+=("${child_pid}")
    host_jobs+=("${idx}" "${child_pid}")
    set +m
  done

  collect_host_setup_results "${host_jobs[@]}"
  exec {semaphore_fd}>&-
}

resolve_configured_data_ip() {
  local idx="$1"
  local host="${HOSTS[$idx]}"
  local user="${USERS[$idx]}"
  local password="${PASSWORDS[$idx]}"
  local explicit_ip="${DATA_IPS[$idx]:-}"
  local ip=""
  local addr_command="ip -o -4 addr show scope global"

  if [[ -n "${DATA_INTERFACE}" ]]; then
    addr_command="ip -o -4 addr show dev $(shell_quote "${DATA_INTERFACE}") scope global"
  fi
  if [[ -n "${explicit_ip}" ]]; then
    if ! remote_sudo_capture "${host}" "${user}" "${password}" \
      "${addr_command} | awk '{split(\$4, parts, \"/\"); print parts[1]}'" \
      | grep -Fxq "${explicit_ip}"; then
      die "configured data IP ${explicit_ip} is not assigned to ${host}${DATA_INTERFACE:+ on ${DATA_INTERFACE}}"
    fi
    printf '%s\n' "${explicit_ip}"
    return 0
  fi
  if [[ -n "${DATA_INTERFACE}" ]]; then
    ip="$(remote_cluster_ip "${host}" "${user}" "${password}" \
      "${addr_command} | awk '{split(\$4, parts, \"/\"); print parts[1]}'" || true)"
    [[ -n "${ip}" ]] || die "could not resolve an IPv4 address on ${host}:${DATA_INTERFACE}"
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

  if [[ -n "${DATA_IPS[0]:-}" || -n "${DATA_INTERFACE}" ]]; then
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
  if [[ -n "${DATA_IPS[$idx]:-}" || -n "${DATA_INTERFACE}" ]]; then
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
  local head_ip="$3"
  local host="${HOSTS[$idx]}"
  local user="${USERS[$idx]}"
  local password="${PASSWORDS[$idx]}"
  local node_ip="${NODE_IPS[$idx]}"
  local gpu_count="${GPU_COUNTS[$idx]}"

  log "${host}: starting Ray ${role} container on ${node_ip}"
  remote_sudo_script "${host}" "${user}" "${password}" \
    "${REMOTE_ROOT}/remote_ray_container.sh" \
    "${role}" "${IMAGE}" "${head_ip}" "${node_ip}" "${gpu_count}" "${GROOT_DATA_ROOT}" \
    "${DATA_INTERFACE}" "${NCCL_SOCKET_IFNAME}" "${GLOO_SOCKET_IFNAME}" \
    "${NCCL_IB_HCA}" "${NCCL_IB_GID_INDEX}" "${REQUIRE_RDMA}" "${TARGET_GPU_NAME}"
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

wait_for_cluster_gpus() {
  local expected_gpus="$1"
  local host="${HOSTS[0]}"
  local user="${USERS[0]}"
  local password="${PASSWORDS[0]}"

  log "${host}: waiting for Ray cluster to report ${expected_gpus} GB300-backed GPU(s)"
  remote_sudo_bash_lc "${host}" "${user}" "${password}" \
    "for i in \$(seq 1 60); do actual=\$(docker exec -i ${HEAD_CONTAINER} python - <<'PY' 2>/dev/null | awk '/^[0-9]+$/ {value=\$1} END {print value}' || true
import ray
ray.init(address='auto', logging_level='ERROR')
resources = ray.cluster_resources()
print(int(resources.get('GPU', 0)))
PY
); if [[ \"\${actual}\" =~ ^[0-9]+$ && \"\${actual}\" -eq ${expected_gpus} ]]; then exit 0; fi; sleep 2; done; docker exec ${HEAD_CONTAINER} ray status; exit 1"
}

run_transport_validation() {
  if [[ -z "${TRANSPORT_VALIDATION_COMMAND}" ]]; then
    return
  fi

  local host="${HOSTS[0]}"
  local user="${USERS[0]}"
  local password="${PASSWORDS[0]}"
  local command
  command="timeout --kill-after=10s ${TRANSPORT_VALIDATION_TIMEOUT}s docker exec"
  command+=" -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET"
  command+=" ${HEAD_CONTAINER} bash -lc $(shell_quote "${TRANSPORT_VALIDATION_COMMAND}")"
  log "${host}: running configured transport validation command"
  remote_sudo_bash_lc "${host}" "${user}" "${password}" "${command}"
}

configure_nginx_on_head() {
  local host="${HOSTS[0]}"
  local user="${USERS[0]}"
  local password="${PASSWORDS[0]}"

  log "${host}: configuring nginx basic-auth proxy at https://${HEAD_DOMAIN}/ray/"
  remote_scp_to "${host}" "${user}" "${password}" "${TMP_DIR}/remote_nginx_ray.sh" "${REMOTE_ROOT}/remote_nginx_ray.sh"
  remote_scp_to "${host}" "${user}" "${password}" "${TMP_DIR}/ray-auth-password" "${REMOTE_ROOT}/ray-auth-password"
  remote_ssh "${host}" "${user}" "${password}" "chmod +x $(shell_quote "${REMOTE_ROOT}/remote_nginx_ray.sh") && chmod 600 $(shell_quote "${REMOTE_ROOT}/ray-auth-password")"
  remote_sudo_script "${host}" "${user}" "${password}" \
    "${REMOTE_ROOT}/remote_nginx_ray.sh" \
    "${HEAD_DOMAIN}" "${SSL_CERT}" "${SSL_KEY}" "${RAY_AUTH_USER}" "${REMOTE_ROOT}/ray-auth-password"
}

start_ray_cluster() {
  if [[ "${HOST_READY[0]:-0}" != "1" ]]; then
    die "head host ${HOSTS[0]} is not ready; refusing to promote another host"
  fi

  local head_ip
  head_ip="$(resolve_head_ip)"
  log "head Ray address: ${head_ip}:${RAY_PORT}"

  local idx
  for idx in "${!HOSTS[@]}"; do
    if [[ "${HOST_READY[$idx]:-0}" == "1" ]]; then
      resolve_node_ip "${idx}" "${head_ip}"
    fi
  done

  start_ray_container "0" "head" "${head_ip}"
  wait_for_head
  configure_nginx_on_head

  for idx in "${!HOSTS[@]}"; do
    if [[ "${idx}" == "0" ]]; then
      continue
    fi
    if [[ "${HOST_READY[$idx]:-0}" == "1" ]]; then
      start_ray_container "${idx}" "worker" "${head_ip}"
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

main() {
  parse_args "$@"
  validate_args
  require_cmd python3
  parse_inventory

  if [[ "${DRY_RUN}" == "1" ]]; then
    print_dry_run
    return 0
  fi

  resolve_secret_sources
  require_cmd sshpass ssh scp

  TMP_DIR="$(mktemp -d)"
  write_remote_scripts
  create_local_secret_files
  setup_hosts
  if [[ "${#FAILED_HOSTS[@]}" -gt 0 ]]; then
    die "host setup failed for ${FAILED_HOSTS[*]}; refusing to start a partial Ray cluster"
  fi
  start_ray_cluster

  log "Ray endpoint: https://${HEAD_DOMAIN}/ray/"
}

main "$@"
