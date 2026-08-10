#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Assemble the heterogeneous Colossus pool used by GR00T auto-research.
# Reuse eligible service-user leases first, then reserve every available
# two-GPU station that fits under the service user's lease limit.

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAY_SETUP="${SCRIPT_DIR}/setup_ray_cluster.sh"

readonly POOL_ID="${COLOSSUS_POOL_ID:-gear-gb300-automation}"
readonly LEASE_DURATION="${COLOSSUS_LEASE_DURATION:-180d}"
readonly LEASE_JUSTIFICATION="GR00T auto research Ray cluster"
readonly CHECKPOINT_URI="${GROOT_OL_POST_TRAINING_CHECKPOINT_URI:-s3://GearCheckpoints/trinity-ckpt-repository/amaddukuri_n2_posttrain_0528_230150_min_wam_robocasa365_pretrain_human300/checkpoint-80000}"
readonly CHECKPOINT_DESTINATION="robocasa_benchmark/new-robocasa/midtrained_checkpoints/n2_posttrain_0528_230150_min_wam_robocasa365_pretrain_human300/checkpoint-80000"
readonly DATASET_URI="s3://GrootDatasets/robocasa_benchmark/new-robocasa/robocasa-datasets/v1.0/archives/pretrain-112958637747-227001.tar"
readonly DATASET_SHA256="0de7d998712edee3f0503de858defefb25cf9877d19ecacbcf94122abf8a290e"
readonly RUNTIME_URI="s3://GrootDatasets/robocasa_benchmark/runtime-assets/20260731/robocasa365-runtime.tar"
readonly RUNTIME_SHA256="f03046021b5fff6d07888aaed17ef52e2dbe231de12703fb2f7003b04ed8e837"
readonly STATE_OUTPUT="${AUTO_RESEARCH_STATE_OUTPUT:-${PWD}/auto-research-colossus-cluster.json}"
readonly SETUP_CONCURRENCY="${AUTO_RESEARCH_SETUP_CONCURRENCY:-24}"
readonly USE_DEDICATED_ETHERNET="${AUTO_RESEARCH_USE_DEDICATED_ETHERNET:-0}"
readonly FORCE_MANAGEMENT_NETWORK="${AUTO_RESEARCH_FORCE_MANAGEMENT_NETWORK:-1}"
readonly ARTIFACT_MAX_MIB_PER_SECOND="${AUTO_RESEARCH_ARTIFACT_MAX_MIB_PER_SECOND:-32}"
readonly PREFERRED_HEAD_RESOURCE="${AUTO_RESEARCH_HEAD_RESOURCE:-pmgb300ws-0205}"

# These leases are shared with humans and remain excluded even if inventory
# status or justification changes later.
readonly -a EXCLUDED_RESOURCES=(
  pmgb300ws-0192
  pmgb300ws-0210
  pmgb300ws-0449
)

DRY_RUN="0"
WORK_DIR=""
LOG_DIR=""
HEAD_ADDRESS=""
HEAD_DOMAIN=""
TLS_CERT=""
TLS_KEY=""
declare -a REUSABLE_LEASE_IDS=()
declare -a REUSABLE_RESOURCE_NAMES=()
declare -a REUSABLE_FQDNS=()
declare -a REUSABLE_IPS=()
declare -a SUCCESS_STATE_FILES=()
declare -a PROVISIONING_PGIDS=()
declare -A SKIPPED_LEASE_IDS=()
INTERRUPT_PENDING="0"

log() {
  printf '[setup_auto_research_cluster] %s\n' "$*"
}

colossus_cli() {
  # Keep lease and inventory calls serialized across the parallel setup
  # subprocesses. The host bootstrap and artifact copies remain concurrent.
  local lock_file="${COLOSSUS_CLI_LOCK_FILE:-/tmp/groot-colossus-cli.lock}"
  (
    flock --exclusive 9
    command colossus "$@"
  ) 9>"${lock_file}"
}

die() {
  printf '[setup_auto_research_cluster] ERROR: %s\n' "$*" >&2
  exit 1
}

cleanup_sensitive_material() {
  if [[ -n "${TLS_KEY}" && -f "${TLS_KEY}" ]]; then
    rm -f -- "${TLS_KEY}"
  fi
}

defer_interrupt() {
  INTERRUPT_PENDING="1"
}

terminate_provisioning_group() {
  local pgid="$1"
  local attempt
  kill -TERM -- "-${pgid}" 2>/dev/null || return 0
  for attempt in {1..10}; do
    kill -0 -- "-${pgid}" 2>/dev/null || return 0
    sleep 0.1
  done
  # Re-address the process group rather than a stale list of bare PIDs. Any
  # descendant forked while handling TERM remains in this isolated group.
  kill -KILL -- "-${pgid}" 2>/dev/null || true
}

remove_provisioning_pgid() {
  local completed_pgid="$1"
  local tracked_pgid
  local -a remaining_pgids=()
  for tracked_pgid in "${PROVISIONING_PGIDS[@]}"; do
    if [[ "${tracked_pgid}" != "${completed_pgid}" ]]; then
      remaining_pgids+=("${tracked_pgid}")
    fi
  done
  PROVISIONING_PGIDS=("${remaining_pgids[@]}")
}

provisioning_pgid_is_running() {
  local expected_pgid="$1"
  local running_pgid
  for running_pgid in $(jobs -p); do
    if [[ "${running_pgid}" == "${expected_pgid}" ]]; then
      return 0
    fi
  done
  return 1
}

prune_finished_provisioning_pgids() {
  local tracked_pgid
  local -a running_pgids=()
  for tracked_pgid in "${PROVISIONING_PGIDS[@]}"; do
    if provisioning_pgid_is_running "${tracked_pgid}"; then
      running_pgids+=("${tracked_pgid}")
    fi
  done
  PROVISIONING_PGIDS=("${running_pgids[@]}")
}

run_provisioning_group() {
  local pgid status
  INTERRUPT_PENDING="0"
  trap defer_interrupt INT TERM
  set -m
  (set +m; exec "$@") &
  pgid="$!"
  PROVISIONING_PGIDS+=("${pgid}")
  if [[ "${INTERRUPT_PENDING}" == "1" ]]; then
    trap handle_interrupt INT TERM
    handle_interrupt
  fi
  # The group retains its PGID after monitor mode is disabled.
  set +m

  if wait "${pgid}"; then
    status=0
  else
    status="$?"
  fi
  # Keep interrupts deferred until the wait result and Bash's owned-job table
  # have been reconciled. This prevents signaling a PGID after wait reaped it.
  if ! provisioning_pgid_is_running "${pgid}"; then
    remove_provisioning_pgid "${pgid}"
  fi
  trap handle_interrupt INT TERM
  if [[ "${INTERRUPT_PENDING}" == "1" ]]; then
    handle_interrupt
  fi
  return "${status}"
}

wait_for_provisioning_groups() {
  local completed_pgid
  while (( ${#PROVISIONING_PGIDS[@]} > 0 )); do
    INTERRUPT_PENDING="0"
    trap defer_interrupt INT TERM
    completed_pgid=""
    wait -n -p completed_pgid "${PROVISIONING_PGIDS[@]}" || true
    if [[ -n "${completed_pgid:-}" ]]; then
      # Do not leave a reaped numeric PGID armed while another setup runs: the
      # kernel can reuse it for an unrelated process group.
      remove_provisioning_pgid "${completed_pgid}"
    else
      # An interrupt makes wait return before assigning completed_pgid. Consult
      # Bash's owned-job table instead of clearing live groups in that case.
      prune_finished_provisioning_pgids
    fi
    trap handle_interrupt INT TERM
    if [[ "${INTERRUPT_PENDING}" == "1" ]]; then
      handle_interrupt
    fi
  done
}

handle_interrupt() {
  trap '' INT TERM
  log "interrupted; terminating background provisioning"
  local pgid
  for pgid in "${PROVISIONING_PGIDS[@]}"; do
    terminate_provisioning_group "${pgid}"
  done
  for pgid in "${PROVISIONING_PGIDS[@]}"; do
    wait "${pgid}" 2>/dev/null || true
  done
  PROVISIONING_PGIDS=()
  set +m
  cleanup_sensitive_material
  exit 130
}

trap handle_interrupt INT TERM
trap cleanup_sensitive_material EXIT

usage() {
  cat <<EOF
Usage: ci/scripts/colossus/setup_auto_research_cluster.sh [--dry-run]

Required environment for a real setup:
  COLOSSUS_RAY_PASSWORD       Basic-auth password for the Ray Jobs endpoint.
  NVCR_TOKEN                  NVCR pull token used on every station.
  BUCKET_ACCESS_KEY_ID        PDX S3 access key.
  BUCKET_SECRET_ACCESS_KEY    PDX S3 secret key.
  BUCKET_ENDPOINT_URL         PDX endpoint, normally https://pdx.s8k.io.
  BUCKET_REGION               Optional; defaults to us-east-1.

Configuration environment:
  AUTO_RESEARCH_STATE_OUTPUT       Aggregate non-secret JSON output.
  AUTO_RESEARCH_SETUP_CONCURRENCY  Parallel reused-lease setup count (default: 24).
  AUTO_RESEARCH_USE_DEDICATED_ETHERNET
                                   Set to 1 to require eno3 on every host instead of
                                   the default per-host prefer-with-fallback policy.
  AUTO_RESEARCH_FORCE_MANAGEMENT_NETWORK
                                   Set to 1 to use routed management TCP on the
                                   connected ConnectX-6 interface (default: 1).
  AUTO_RESEARCH_ARTIFACT_MAX_MIB_PER_SECOND
                                   Per-host S3 staging cap (default: 32 MiB/s).
  AUTO_RESEARCH_HEAD_RESOURCE     Prefer this active reusable host as the Ray head.
                                   Default: ${PREFERRED_HEAD_RESOURCE}.
  COLOSSUS_LEASE_DURATION          Duration for new leases (default: 180d,
                                   the current Colossus maximum).

The script must run under the Colossus service account. It reuses active
service-owned two-GPU leases, but never touches CI leases, the explicit
human-shared exclusions, or one-GPU resources. Unreachable hosts are skipped.
EOF
}

require_command() {
  local command
  for command in "$@"; do
    command -v "${command}" >/dev/null 2>&1 || die "missing required command: ${command}"
  done
}

validate_inputs() {
  [[ -x "${RAY_SETUP}" ]] || die "missing executable Ray setup script: ${RAY_SETUP}"
  case "${SETUP_CONCURRENCY}" in
    ''|*[!0-9]*) die "AUTO_RESEARCH_SETUP_CONCURRENCY must be a positive integer" ;;
  esac
  (( SETUP_CONCURRENCY > 0 )) || die "AUTO_RESEARCH_SETUP_CONCURRENCY must be greater than 0"
  case "${USE_DEDICATED_ETHERNET}" in
    0|1) ;;
    *) die "AUTO_RESEARCH_USE_DEDICATED_ETHERNET must be 0 or 1" ;;
  esac
  case "${FORCE_MANAGEMENT_NETWORK}" in
    0|1) ;;
    *) die "AUTO_RESEARCH_FORCE_MANAGEMENT_NETWORK must be 0 or 1" ;;
  esac
  [[ "${USE_DEDICATED_ETHERNET}" != "1" || "${FORCE_MANAGEMENT_NETWORK}" != "1" ]] \
    || die "dedicated Ethernet and forced management network are mutually exclusive"
  case "${ARTIFACT_MAX_MIB_PER_SECOND}" in
    ''|*[!0-9]*) die "AUTO_RESEARCH_ARTIFACT_MAX_MIB_PER_SECOND must be positive" ;;
  esac
  (( ARTIFACT_MAX_MIB_PER_SECOND > 0 )) \
    || die "AUTO_RESEARCH_ARTIFACT_MAX_MIB_PER_SECOND must be greater than 0"
  [[ "${PREFERRED_HEAD_RESOURCE}" =~ ^[A-Za-z0-9._-]+$ ]] \
    || die "AUTO_RESEARCH_HEAD_RESOURCE must be a Colossus resource name"

  local identity
  identity="$(colossus_cli whoami 2>/dev/null | sed -n 's/^Current User: //p')"
  [[ "${identity}" == nvssa-* ]] \
    || die "Colossus must be authenticated as the service user; current identity is ${identity:-unknown}"

  if [[ "${DRY_RUN}" != "1" ]]; then
    local secret_name
    for secret_name in \
      COLOSSUS_RAY_PASSWORD \
      NVCR_TOKEN \
      BUCKET_ACCESS_KEY_ID \
      BUCKET_SECRET_ACCESS_KEY \
      BUCKET_ENDPOINT_URL
    do
      [[ -n "${!secret_name:-}" ]] || die "${secret_name} is required"
    done
  fi
}

load_inventory() {
  local assignments
  local inventory_file="${WORK_DIR}/leases.json"
  colossus_cli bm lease list --json > "${inventory_file}" \
    || die "could not list Colossus leases"
  assignments="$(python3 - \
    "${inventory_file}" \
    "${PREFERRED_HEAD_RESOURCE}" \
    "$(IFS=,; printf '%s' "${EXCLUDED_RESOURCES[*]}")" <<'PY'
import json
import re
import shlex
import sys

inventory_file, preferred_head, excluded_text = sys.argv[1:]
excluded = set(filter(None, excluded_text.split(",")))
with open(inventory_file) as source:
    records = json.load(source)
candidates = []
for record in records:
    entity = record.get("entity_details") or {}
    lease = record.get("lease_details") or {}
    name = entity.get("name", "")
    justification = (
        entity.get("leaseJustification")
        or lease.get("leasingJustification")
        or lease.get("leaseJustification")
        or ""
    )
    lease_id = lease.get("id") or entity.get("leaseId") or ""
    owner = (lease.get("owner") or {}).get("userName") or entity.get("leasee") or ""
    if (
        str(lease.get("status", "")).upper() != "ACTIVE"
        or str(entity.get("status", "")).upper() != "RESERVED"
    ):
        continue
    is_service_owned = isinstance(owner, str) and owner.startswith("nvssa-")
    is_ci = justification.startswith("Dedicated GR00T GitLab CI")
    if entity.get("gpuCount") != 2 or not is_service_owned or is_ci:
        continue
    if name in excluded:
        continue
    values = (name, lease_id, entity.get("fqdn", ""), entity.get("ipAddress", ""))
    if not all(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9._:-]+", value) for value in values):
        continue
    candidates.append(values)

print("REUSABLE_LEASE_IDS=()")
print("REUSABLE_RESOURCE_NAMES=()")
print("REUSABLE_FQDNS=()")
print("REUSABLE_IPS=()")
for name, lease_id, fqdn, ip_address in sorted(
    candidates, key=lambda values: (values[0] != preferred_head, values[0])
):
    print(f"REUSABLE_LEASE_IDS+=({shlex.quote(lease_id)})")
    print(f"REUSABLE_RESOURCE_NAMES+=({shlex.quote(name)})")
    print(f"REUSABLE_FQDNS+=({shlex.quote(fqdn)})")
    print(f"REUSABLE_IPS+=({shlex.quote(ip_address)})")
PY
)" || die "could not discover reusable service-user leases"
  eval "${assignments}"
}

available_two_gpu_count() {
  local inventory_file="${WORK_DIR}/resources.json"
  colossus_cli bm resource list \
    --filters "poolId=${POOL_ID}" status=AVAILABLE gpuTag=GB300 \
    --authorized-to-reserve \
    --page-size 500 \
    --json > "${inventory_file}" \
    || die "could not list available Colossus resources"
  python3 - \
    "${inventory_file}" \
    "$(IFS=,; printf '%s' "${EXCLUDED_RESOURCES[*]}")" <<'PY'
import json
import sys

inventory_file, excluded_text = sys.argv[1:]
excluded = set(filter(None, excluded_text.split(",")))
with open(inventory_file) as source:
    resources = json.load(source)
print(sum(
    1
    for resource in resources
    if resource.get("status") == "AVAILABLE"
    and resource.get("gpuCount") == 2
    and resource.get("name") not in excluded
))
PY
}

common_setup_args() {
  COMMON_SETUP_ARGS=(
    --pool-id "${POOL_ID}"
    --gpu-profile gb300-rtx-pro-2000
    --resource-gpu-count 2
    --allow-socket-transport
    --lease-duration "${LEASE_DURATION}"
    --lease-justification "${LEASE_JUSTIFICATION}"
    --ray-auth-password-env COLOSSUS_RAY_PASSWORD
    --nvcr-token-env NVCR_TOKEN
    --ssl-cert-local "${TLS_CERT}"
    --ssl-key-local "${TLS_KEY}"
    --post-training-checkpoint-uri "${CHECKPOINT_URI}"
    --post-training-checkpoint-destination "${CHECKPOINT_DESTINATION}"
    --dataset-archive-uri "${DATASET_URI}"
    --dataset-archive-sha256 "${DATASET_SHA256}"
    --robocasa-runtime-archive-uri "${RUNTIME_URI}"
    --robocasa-runtime-archive-sha256 "${RUNTIME_SHA256}"
    --artifact-max-mib-per-second "${ARTIFACT_MAX_MIB_PER_SECOND}"
  )
  if [[ "${USE_DEDICATED_ETHERNET}" == "1" ]]; then
    COMMON_SETUP_ARGS+=(
      --ethernet-only
      --data-interface eno3
      --static-data-ip-prefix 192.168.240
    )
  elif [[ "${FORCE_MANAGEMENT_NETWORK}" != "1" ]]; then
    COMMON_SETUP_ARGS+=(
      --prefer-ethernet-interface eno3
      --static-data-ip-prefix 192.168.240
    )
  fi
  local excluded
  for excluded in "${EXCLUDED_RESOURCES[@]}"; do
    COMMON_SETUP_ARGS+=(--exclude-resource "${excluded}")
  done
}

generate_tls_material() {
  TLS_CERT="${WORK_DIR}/auto-research-ray-ca.pem"
  TLS_KEY="${WORK_DIR}/auto-research-ray-key.pem"
  openssl req \
    -x509 \
    -newkey rsa:3072 \
    -sha256 \
    -nodes \
    -days 30 \
    -subj '/CN=GR00T Colossus auto-research Ray' \
    -addext 'basicConstraints=critical,CA:TRUE' \
    -addext 'keyUsage=critical,digitalSignature,keyEncipherment,keyCertSign' \
    -addext 'subjectAltName=DNS:*.bldgc1.colossus.nvidia.com,DNS:*.ipp9c1.colossus.nvidia.com' \
    -keyout "${TLS_KEY}" \
    -out "${TLS_CERT}" \
    >/dev/null 2>&1
  chmod 0600 "${TLS_CERT}" "${TLS_KEY}"
}

state_value() {
  local state_file="$1"
  local field="$2"
  python3 - "${state_file}" "${field}" <<'PY'
import json
import sys

with open(sys.argv[1]) as source:
    value = json.load(source)[sys.argv[2]]
if not isinstance(value, str):
    raise SystemExit(f"{sys.argv[2]} is not a string")
print(value)
PY
}

try_reused_head() {
  local idx lease_id name state_file log_file
  for idx in "${!REUSABLE_LEASE_IDS[@]}"; do
    lease_id="${REUSABLE_LEASE_IDS[$idx]}"
    name="${REUSABLE_RESOURCE_NAMES[$idx]}"
    SKIPPED_LEASE_IDS["${lease_id}"]=1
    state_file="${WORK_DIR}/head-${name}.json"
    log_file="${LOG_DIR}/head-${name}.log"
    log "${name}: trying reusable lease as Ray head"
    if run_provisioning_group "${RAY_SETUP}" \
      "${COMMON_SETUP_ARGS[@]}" \
      --nodes 1 \
      --existing-lease-id "${lease_id}" \
      --state-output "${state_file}" \
      > >(tee "${log_file}") 2>&1; then
      HEAD_ADDRESS="$(state_value "${state_file}" ray_address)"
      HEAD_DOMAIN="$(state_value "${state_file}" head_domain)"
      SUCCESS_STATE_FILES+=("${state_file}")
      log "${name}: selected as head at ${HEAD_ADDRESS}"
      return 0
    fi
    log "${name}: not usable as head; skipping it"
  done
  return 1
}

try_new_head() {
  local available state_file log_file
  available="$(available_two_gpu_count)"
  (( available > 0 )) || return 1
  state_file="${WORK_DIR}/head-new.json"
  log_file="${LOG_DIR}/head-new.log"
  log "requesting one newly leased head because no reusable lease was ready"
  if run_provisioning_group "${RAY_SETUP}" \
    "${COMMON_SETUP_ARGS[@]}" \
    --nodes 1 \
    --best-effort-hosts \
    --state-output "${state_file}" \
    > >(tee "${log_file}") 2>&1; then
    HEAD_ADDRESS="$(state_value "${state_file}" ray_address)"
    HEAD_DOMAIN="$(state_value "${state_file}" head_domain)"
    SUCCESS_STATE_FILES+=("${state_file}")
    return 0
  fi
  return 1
}

setup_reused_worker() {
  local idx="$1"
  local lease_id="${REUSABLE_LEASE_IDS[$idx]}"
  local name="${REUSABLE_RESOURCE_NAMES[$idx]}"
  local state_file="${WORK_DIR}/worker-${name}.json"
  local log_file="${LOG_DIR}/worker-${name}.log"
  if [[ -n "${SKIPPED_LEASE_IDS[${lease_id}]:-}" ]]; then
    return 0
  fi
  log "${name}: preparing reusable lease as a Ray worker"
  if "${RAY_SETUP}" \
    "${COMMON_SETUP_ARGS[@]}" \
    --nodes 1 \
    --existing-lease-id "${lease_id}" \
    --join-ray-address "${HEAD_ADDRESS}" \
    --state-output "${state_file}" \
    > >(tee "${log_file}") 2>&1; then
    printf '%s\n' "${state_file}" > "${state_file}.success"
  else
    log "${name}: setup or SSH failed; lease was ignored"
  fi
}

setup_reused_workers() {
  local idx pid
  local -a active_pids=()
  for idx in "${!REUSABLE_LEASE_IDS[@]}"; do
    setup_reused_worker "${idx}" &
    active_pids+=("$!")
    if (( ${#active_pids[@]} >= SETUP_CONCURRENCY )); then
      for pid in "${active_pids[@]}"; do
        wait "${pid}" || true
      done
      active_pids=()
    fi
  done
  for pid in "${active_pids[@]}"; do
    wait "${pid}" || true
  done
}

reserve_available_workers() {
  local available state_file log_file
  available="$(available_two_gpu_count)"
  if (( available == 0 )); then
    log "no additional available two-GPU station remains"
    return 0
  fi
  state_file="${WORK_DIR}/workers-new.json"
  log_file="${LOG_DIR}/workers-new.log"
  log "requesting all ${available} currently available two-GPU stations best-effort"
  if "${RAY_SETUP}" \
    "${COMMON_SETUP_ARGS[@]}" \
    --nodes "${available}" \
    --best-effort-hosts \
    --join-ray-address "${HEAD_ADDRESS}" \
    --state-output "${state_file}" \
    > >(tee "${log_file}") 2>&1; then
    printf '%s\n' "${state_file}" > "${state_file}.success"
  else
    log "new reservations yielded no usable worker (lease limit/race/SSH); keeping reused capacity"
  fi
}

setup_all_workers() {
  local reused_pid new_pid marker
  # Monitor mode gives each outer supervisor an isolated process group. Turn
  # it back off inside each supervisor so all of its descendants stay in that
  # group and can be terminated atomically on interruption.
  INTERRUPT_PENDING="0"
  trap defer_interrupt INT TERM
  set -m
  (set +m; setup_reused_workers) &
  reused_pid="$!"
  PROVISIONING_PGIDS+=("${reused_pid}")
  if [[ "${INTERRUPT_PENDING}" == "1" ]]; then
    handle_interrupt
  fi
  (set +m; reserve_available_workers) &
  new_pid="$!"
  PROVISIONING_PGIDS+=("${new_pid}")
  trap handle_interrupt INT TERM
  # The groups retain their PGIDs after monitor mode is disabled; turning it
  # off here avoids asynchronous job-control notifications in setup logs.
  set +m
  if [[ "${INTERRUPT_PENDING}" == "1" ]]; then
    handle_interrupt
  fi
  wait_for_provisioning_groups

  shopt -s nullglob
  for marker in "${WORK_DIR}"/*.success; do
    SUCCESS_STATE_FILES+=("$(tr -d '\r\n' < "${marker}")")
  done
  shopt -u nullglob
}

write_aggregate_state() {
  local ca_output="${STATE_OUTPUT}.ca.pem"
  install -m 0644 "${TLS_CERT}" "${ca_output}"
  python3 - \
    "${STATE_OUTPUT}" \
    "${HEAD_ADDRESS}" \
    "${HEAD_DOMAIN}" \
    "${LEASE_JUSTIFICATION}" \
    "${LOG_DIR}" \
    "${ca_output}" \
    "${SUCCESS_STATE_FILES[@]}" <<'PY'
import json
from pathlib import Path
import sys

output, head_address, head_domain, justification, log_dir, ca_cert, *state_files = sys.argv[1:]
hosts = []
node_ips = []
created_lease_ids = []
created_lease_names = []
reused_lease_ids = []
for state_file in state_files:
    with open(state_file) as source:
        state = json.load(source)
    hosts.extend(state.get("hosts", []))
    node_ips.extend(state.get("node_ips", []))
    created_lease_ids.extend(state.get("created_lease_ids", []))
    created_lease_names.extend(state.get("created_lease_names", []))
    if state.get("existing_lease_id"):
        reused_lease_ids.append(state["existing_lease_id"])


def unique(values):
    return list(dict.fromkeys(values))


payload = {
    "head_address": head_address,
    "head_domain": head_domain,
    "ray_jobs_url": f"https://{head_domain}/ray/",
    "gpu_profile": "gb300-rtx-pro-2000",
    "cuda_device_order": {"0": "NVIDIA GB300", "1": "NVIDIA RTX PRO 2000"},
    "transport": "preferred_ethernet_with_management_fallback",
    "lease_justification": justification,
    "hosts": unique(hosts),
    "node_ips": unique(node_ips),
    "reused_lease_ids": unique(reused_lease_ids),
    "created_lease_ids": unique(created_lease_ids),
    "created_lease_names": unique(created_lease_names),
    "logs": log_dir,
    "ca_cert": ca_cert,
}
path = Path(output)
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_name(f".{path.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
  chmod 0600 "${STATE_OUTPUT}"
}

main() {
  case "${1:-}" in
    "") ;;
    --dry-run) DRY_RUN="1" ;;
    -h|--help) usage; return 0 ;;
    *) usage >&2; die "unknown option: $1" ;;
  esac

  require_command colossus flock openssl python3 sshpass ssh scp tee
  validate_inputs
  WORK_DIR="$(mktemp -d /tmp/groot-auto-research-cluster.XXXXXX)"
  load_inventory
  local available
  available="$(available_two_gpu_count)"
  log "eligible reusable two-GPU leases: ${#REUSABLE_LEASE_IDS[@]}"
  log "currently available two-GPU resources: ${available}"
  log "preferred Ray head: ${PREFERRED_HEAD_RESOURCE}"
  log "explicitly excluded resources: ${EXCLUDED_RESOURCES[*]}"
  log "new lease justification: ${LEASE_JUSTIFICATION}"
  if [[ "${USE_DEDICATED_ETHERNET}" == "1" ]]; then
    log "network policy: dedicated eno3 Ethernet; ConnectX/mlx5 rejected"
  elif [[ "${FORCE_MANAGEMENT_NETWORK}" == "1" ]]; then
    log "network policy: routed management TCP over the connected ConnectX-6 interface; no RDMA"
  else
    log "network policy: prefer connected non-mlx5 eno3 per host; use routed management TCP only when unavailable; no RDMA"
  fi
  log "artifact staging cap: ${ARTIFACT_MAX_MIB_PER_SECOND} MiB/s per host"
  if [[ "${DRY_RUN}" == "1" ]]; then
    local idx
    for idx in "${!REUSABLE_LEASE_IDS[@]}"; do
      log "reusable: ${REUSABLE_RESOURCE_NAMES[$idx]} (${REUSABLE_LEASE_IDS[$idx]})"
    done
    return 0
  fi

  mkdir -p "${STATE_OUTPUT}.logs"
  LOG_DIR="$(mktemp -d "${STATE_OUTPUT}.logs/run.XXXXXX")"
  generate_tls_material
  common_setup_args

  try_reused_head || try_new_head \
    || die "no reusable or newly reservable host could become the Ray head"
  setup_all_workers
  if [[ "${USE_DEDICATED_ETHERNET}" != "1" \
        && "${FORCE_MANAGEMENT_NETWORK}" != "1" ]] \
      && grep -Rqs "cannot reach existing Ray head" "${LOG_DIR}"; then
    log "eno3 has carrier but no cross-host path to the Ray head; retrying the staged fleet on routed management TCP"
    export AUTO_RESEARCH_FORCE_MANAGEMENT_NETWORK=1
    cleanup_sensitive_material
    exec "${SCRIPT_DIR}/setup_auto_research_cluster.sh"
  fi
  write_aggregate_state
  cleanup_sensitive_material

  log "auto-research Ray cluster is ready with ${#SUCCESS_STATE_FILES[@]} setup group(s)"
  log "Ray address: ${HEAD_ADDRESS}"
  log "Ray Jobs endpoint: https://${HEAD_DOMAIN}/ray/"
  log "state: ${STATE_OUTPUT}"
  log "logs: ${LOG_DIR}"
  log "One Launcher: uv run ol tev -e n2_turbo_robocasa_stack_cans_colossus -ct colossus"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
