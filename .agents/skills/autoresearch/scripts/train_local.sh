#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-${HOST_REPO_ROOT}}"
CONFIG_PATH="${1:-${REPO_ROOT}/skills/autoresearch/scripts/train_config.yaml}"
if (( $# > 0 )); then
  shift
fi

cd "${REPO_ROOT}"

if [[ -z "${PYTHON:-}" ]]; then
  PYTHON="${HOST_REPO_ROOT}/.venv/bin/python"
  if [[ ! -x "${PYTHON}" && -x "${HOST_REPO_ROOT}/../.venv/bin/python" ]]; then
    PYTHON="${HOST_REPO_ROOT}/../.venv/bin/python"
  fi
fi
if [[ ! -x "${PYTHON}" ]]; then
  printf 'Environment not found. Run %s first.\n' "${SCRIPT_DIR}/setup_env.sh" >&2
  exit 1
fi

printf '[train_local] config: %s\n' "${CONFIG_PATH}"
printf '[train_local] loading training dependencies; this may take a few minutes...\n'

exec "${PYTHON}" \
  -m groot.core.training.experiment.experiment_ray \
  --load-config-path "${CONFIG_PATH}" \
  "$@"
