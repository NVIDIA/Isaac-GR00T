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
# Submit a Docker image build workflow directly to OSMO and stream logs in GitLab CI.
#
# Runs inside groot-ci-base-minimal (amd64, has osmo CLI/crane) and submits a kaniko OSMO
# workflow to a GB200 GPU pool that builds and pushes the image.
# Streams workflow logs live to stdout for visibility in GitLab CI.
#
# Usage:
#   osmo_docker_build.sh \
#     --registry nvcr.io/nvidian \
#     --image-name groot-ci-train \
#     --dockerfile ci/images/train.Dockerfile \
#     --tag nvcr.io/nvidian/groot-ci-train:abc1234 \
#     --platform linux/arm64 \
#     --pool groot-gb200-ci-02 \
#     [--cpu 15] [--gpu 4] [--memory 64Gi] \
#     [--exec-timeout 180m] \
#     [--build-arg KEY=VAL ...]
#
# Required environment variables (inherited from GitLab CI):
#   DOCKER_AUTH_CONFIG   - NGC registry credentials JSON
#   GITLAB_TOKEN or CI_JOB_TOKEN - GitLab token for raw-file API access
#   OSMO_TOKEN, OSMO_URL, OSMO_USERNAME
#   CI_PROJECT_DIR, CI_COMMIT_BRANCH or CI_MERGE_REQUEST_SOURCE_BRANCH_NAME

set -euo pipefail

REGISTRY=""
IMAGE_NAME=""
DOCKERFILE=""
TAG=""
PLATFORM="linux/arm64"
OSMO_PLATFORM=""
POOL="groot-gb200-ci-02"
CPU_COUNT="8"
GPU_COUNT="4"
MEMORY="64Gi"
EXEC_TIMEOUT="180m"
BUILD_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry)     REGISTRY="$2"; shift 2 ;;
    --image-name)   IMAGE_NAME="$2"; shift 2 ;;
    --dockerfile)   DOCKERFILE="$2"; shift 2 ;;
    --tag)          TAG="$2"; shift 2 ;;
    --platform)     PLATFORM="$2"; shift 2 ;;
    --osmo-platform) OSMO_PLATFORM="$2"; shift 2 ;;
    --pool)         POOL="$2"; shift 2 ;;
    --cpu)          CPU_COUNT="$2"; shift 2 ;;
    --gpu)          GPU_COUNT="$2"; shift 2 ;;
    --memory)       MEMORY="$2"; shift 2 ;;
    --exec-timeout) EXEC_TIMEOUT="$2"; shift 2 ;;
    --build-arg)    BUILD_ARGS="${BUILD_ARGS} --build-arg $2"; shift 2 ;;
    *)              echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${REGISTRY}" ]] || [[ -z "${IMAGE_NAME}" ]] || [[ -z "${DOCKERFILE}" ]] \
   || [[ -z "${TAG}" ]]; then
  echo "Missing required arguments (--registry, --image-name, --dockerfile, --tag)" >&2
  exit 1
fi

if [[ -n "${GITLAB_TOKEN:-}" ]]; then
  GITLAB_AUTH_HEADER="PRIVATE-TOKEN: ${GITLAB_TOKEN}"
elif [[ -n "${CI_JOB_TOKEN:-}" ]]; then
  GITLAB_TOKEN="${CI_JOB_TOKEN}"
  GITLAB_AUTH_HEADER="JOB-TOKEN: ${CI_JOB_TOKEN}"
else
  GITLAB_TOKEN=""
  GITLAB_AUTH_HEADER=""
fi
GITLAB_PROJECT_PATH="${GITLAB_PROJECT_PATH:-${CI_PROJECT_PATH:-}}"
GROOT_COMMIT_BRANCH="${GROOT_COMMIT_BRANCH:-${CI_MERGE_REQUEST_SOURCE_BRANCH_NAME:-${CI_COMMIT_BRANCH:-${CI_COMMIT_REF_NAME:-main}}}}"
export GITLAB_AUTH_HEADER
export GITLAB_PROJECT_PATH
export GROOT_COMMIT_BRANCH

# Extract just the tag portion (after the last colon) for kaniko --destination
TAG_SHORT="${TAG##*:}"

if [[ -z "${OSMO_PLATFORM}" ]]; then
  case "${PLATFORM}" in
    linux/amd64) OSMO_PLATFORM="" ;;
    *) OSMO_PLATFORM="gb200" ;;
  esac
fi

# ── Content-addressed skip ─────────────────────────────────────────────────────
# The commit SHA tag changes on every push (including rebases) even when the
# Dockerfile and its base image are byte-for-byte identical.  Instead we derive
# a stable "content-<hash>" tag from:
#   sha256( Dockerfile text + base-image platform manifest digest + non-FROM build args + platform )[:16]
# FROM-only tag args such as BASE_MINIMAL_TAG=${CI_COMMIT_SHORT_SHA} are intentionally
# excluded after resolving the base manifest digest. Otherwise every commit SHA
# misses the GitLab-runner fast path even when the base image content is identical.
# If that tag already exists we retag it as :<sha> via the registry API (no
# data copy, ~2 sec) and skip the OSMO workflow entirely.  When we do build we
# push the content tag as a second destination so future runs can find it.
# ──────────────────────────────────────────────────────────────────────────────

# Compute the content hash; prints "content-<16hexchars>" or exits non-zero.
_compute_content_hash() {
  python3 - "${DOCKERFILE}" "${BUILD_ARGS:-}" "${PLATFORM}" "${OSMO_PLATFORM:-}" <<'PYEOF'
import json, os, sys, hashlib, re, subprocess, tempfile
from pathlib import Path
import urllib.request, urllib.parse

dockerfile_path = sys.argv[1]
build_args_str  = sys.argv[2] if len(sys.argv) > 2 else ""
image_platform  = sys.argv[3] if len(sys.argv) > 3 else ""
osmo_platform   = sys.argv[4] if len(sys.argv) > 4 else ""

gitlab_token = os.environ.get('GITLAB_TOKEN', '')
gitlab_auth_header = os.environ.get('GITLAB_AUTH_HEADER', '')
gitlab_project_path = os.environ.get('GITLAB_PROJECT_PATH', '')
branch = (os.environ.get('CI_MERGE_REQUEST_SOURCE_BRANCH_NAME')
          or os.environ.get('CI_COMMIT_BRANCH')
          or os.environ.get('CI_COMMIT_REF_NAME', 'main'))

def _crane_env():
    docker_auth = os.environ.get('DOCKER_AUTH_CONFIG', '')
    env = os.environ.copy()
    tmpdir = tempfile.TemporaryDirectory()
    if docker_auth:
        Path(tmpdir.name, 'config.json').write_text(docker_auth)
        env['DOCKER_CONFIG'] = tmpdir.name
    return env, tmpdir

def manifest_digest(image_ref, platform):
    """Resolve an image tag to the target platform manifest digest when possible."""
    def top_level_digest():
        env, tmpdir = _crane_env()
        try:
            return subprocess.check_output(
                ['crane', 'digest', image_ref],
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
                env=env,
            ).strip()
        finally:
            tmpdir.cleanup()

    try:
        if not platform or ',' in platform or '/' not in platform:
            return top_level_digest()

        target_os, target_arch, *_ = platform.split('/')
        env, tmpdir = _crane_env()
        raw_manifest = subprocess.check_output(
            ['crane', 'manifest', image_ref],
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            env=env,
        )
        manifest = json.loads(raw_manifest)
        for child in manifest.get('manifests', []):
            child_platform = child.get('platform') or {}
            if (child_platform.get('os') == target_os
                    and child_platform.get('architecture') == target_arch):
                digest = child.get('digest', '')
                if digest:
                    return digest
        return top_level_digest()
    except Exception as e:
        sys.stderr.write(f"Cannot resolve base image digest for {image_ref}: {e}\n")
        return ''
    finally:
        if 'tmpdir' in locals():
            tmpdir.cleanup()

# Fetch Dockerfile text from GitLab raw-file API
branch_enc = branch.replace('/', '%2F')
path_enc   = urllib.parse.quote(dockerfile_path, safe='')
project_enc = urllib.parse.quote(gitlab_project_path, safe='')
url = (f"https://gitlab-master.nvidia.com/api/v4/projects/"
       f"{project_enc}/repository/files/{path_enc}/raw"
       f"?ref={branch_enc}")
try:
    headers = {}
    if gitlab_auth_header and ':' in gitlab_auth_header:
        name, value = gitlab_auth_header.split(':', 1)
        headers[name.strip()] = value.strip()
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        dockerfile_text = r.read().decode()
except Exception as e:
    sys.stderr.write(f"Cannot fetch Dockerfile: {e}\n")
    sys.exit(1)

# Resolve --build-arg KEY=VAL pairs
build_args = {}
for m in re.finditer(r'--build-arg\s+(\w+)=(\S+)', build_args_str):
    build_args[m.group(1)] = m.group(2)

# Find the first real FROM line, substitute ARG values, then resolve digest.
base_digest = ''
base_ref = ''
from_arg_names = set()
for line in dockerfile_text.splitlines():
    s = line.strip()
    if not s.upper().startswith('FROM'):
        continue
    parts = s.split()
    if len(parts) < 2 or parts[1].lower() == 'scratch':
        continue
    base_ref = parts[1]
    from_arg_names.update(
        match.group(1) or match.group(2)
        for match in re.finditer(r'\$\{(\w+)\}|\$(\w+)', base_ref)
    )
    for k, v in build_args.items():
        base_ref = base_ref.replace(f'${{{k}}}', v).replace(f'${k}', v)
    if '/' in base_ref:
        base_digest = manifest_digest(base_ref, image_platform)
    break

# Build args used only to choose the FROM tag are represented by base_digest.
# Keeping the raw commit tag in the hash would defeat the runner-side retag skip.
if from_arg_names and not base_digest:
    sys.stderr.write(f"Cannot compute stable content hash without digest for FROM {base_ref}\n")
    sys.exit(1)
if base_digest:
    hash_build_args = {
        key: value for key, value in build_args.items() if key not in from_arg_names
    }
else:
    hash_build_args = dict(build_args)
normalized_build_args_str = ' '.join(
    f'--build-arg {key}={hash_build_args[key]}' for key in sorted(hash_build_args)
)

# Fetch content of every COPY source so changes to context files (pyproject.toml,
# uv.lock, etc.) invalidate the hash even when the Dockerfile text is unchanged.
# COPY syntax: COPY [--flag ...] <src>... <dest>  — all args except last are sources.
copy_content = b''
for line in dockerfile_text.splitlines():
    s = line.strip()
    if not s.upper().startswith('COPY') or s.startswith('#'):
        continue
    args = [p for p in s.split()[1:] if not p.startswith('--')]
    sources = args[:-1]  # last arg is destination
    for src in sources:
        src_enc = urllib.parse.quote(src, safe='')
        url = (f"https://gitlab-master.nvidia.com/api/v4/projects/"
               f"{project_enc}/repository/files/{src_enc}/raw"
               f"?ref={branch_enc}")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as r:
                copy_content += f"file:{src}\n".encode() + r.read() + b'\n'
        except Exception as e:
            sys.stderr.write(f"Warning: cannot fetch COPY source {src}: {e}\n")
            copy_content += f"file:{src}:MISSING\n".encode()

content_hash = hashlib.sha256(
    dockerfile_text.encode()
    + b'\n'
    + base_digest.encode()
    + b'\n'
    + normalized_build_args_str.encode()
    + b'\n'
    + image_platform.encode()
    + b'\n'
    + osmo_platform.encode()
    + b'\n'
    + copy_content
).hexdigest()[:16]
print(f"content-{content_hash}")
PYEOF
}

# crane is pre-installed in groot-ci-base-minimal (Layer 2).
# Both functions write DOCKER_AUTH_CONFIG to /tmp/crane_auth/config.json so
# crane can authenticate against NGC without a running Docker daemon.

# Check if a full image ref exists — exit 0 if yes, non-zero otherwise.
_image_tag_exists() {
  local ref="$1"
  mkdir -p /tmp/crane_auth
  printf '%s' "${DOCKER_AUTH_CONFIG}" > /tmp/crane_auth/config.json
  DOCKER_CONFIG=/tmp/crane_auth crane manifest "${ref}" >/dev/null 2>&1
}

# Retag src_ref (full image ref) as new_tag in the same repo.
# No layer data is transferred — just a manifest PUT.
_crane_tag() {
  local src_ref="$1"
  local new_tag="$2"
  mkdir -p /tmp/crane_auth
  printf '%s' "${DOCKER_AUTH_CONFIG}" > /tmp/crane_auth/config.json
  DOCKER_CONFIG=/tmp/crane_auth crane tag "${src_ref}" "${new_tag}"
}

# Derive the content tag and decide whether to build or skip
CONTENT_TAG=""  # empty = hash failed; content tag push skipped after build
echo "--- Computing content hash for ${REGISTRY}/${IMAGE_NAME} ---"
_CONTENT_HASH_OUT=""
_CONTENT_HASH_ERR="$(mktemp)"
_CONTENT_HASH_RC=0
_CONTENT_HASH_OUT=$(_compute_content_hash 2>"${_CONTENT_HASH_ERR}") || _CONTENT_HASH_RC=$?

if [[ "${_CONTENT_HASH_RC}" -eq 0 && -n "${_CONTENT_HASH_OUT}" ]]; then
  CONTENT_TAG="${_CONTENT_HASH_OUT}"
  CONTENT_REF="${REGISTRY}/${IMAGE_NAME}:${CONTENT_TAG}"
  echo "--- Content tag: ${CONTENT_REF} ---"

  _EXISTS_RC=0
  _image_tag_exists "${CONTENT_REF}" || _EXISTS_RC=$?

  if [[ "${_EXISTS_RC}" -eq 0 ]]; then
    echo "--- Content-identical image exists; retagging as :${TAG_SHORT} ---"
    _crane_tag "${CONTENT_REF}" "${TAG_SHORT}"
    echo "--- Skipping OSMO build ---"
    exit 0
  elif [[ "${_EXISTS_RC}" -eq 1 ]]; then
    echo "--- Content tag not found; proceeding with OSMO build ---"
  else
    echo "--- Content tag check inconclusive; proceeding with OSMO build ---"
  fi
else
  echo "--- Content hash computation failed (rc=${_CONTENT_HASH_RC}); proceeding with OSMO build ---"
  cat "${_CONTENT_HASH_ERR}" >&2 || true
fi
rm -f "${_CONTENT_HASH_ERR}"

# Install osmo CLI and login
bash ci/scripts/osmo_runner_install.sh

# Base64-encode the NGC auth JSON to avoid quoting/escaping issues in YAML template
DOCKER_AUTH_CONFIG_B64="$(echo "${DOCKER_AUTH_CONFIG}" | base64 -w0)"

# Branch name: MR pipelines set CI_MERGE_REQUEST_SOURCE_BRANCH_NAME; push pipelines set CI_COMMIT_BRANCH
GROOT_COMMIT_BRANCH="${CI_MERGE_REQUEST_SOURCE_BRANCH_NAME:-${CI_COMMIT_BRANCH:-main}}"

# Priority: LOW on main branch (preserve CI quota), HIGH for MR pipelines
if [[ "${CI_COMMIT_BRANCH:-}" = "${CI_DEFAULT_BRANCH:-main}" ]]; then
  OSMO_PRIORITY=LOW
else
  OSMO_PRIORITY=HIGH
fi

# Cache isolation follows the same slug pattern as docker_buildx_push.sh:
#   write to  <image>:cache-<CI_COMMIT_REF_SLUG>   (per-branch, never poisons main)
#   read from <image>:cache-<CI_COMMIT_REF_SLUG>   (warm layers from prior runs)
#         AND <image>:cache-main                    (warm layers from main branch)
#
# In kaniko, --cache-repo stores layers as <repo>:<hash>.  We use separate repos
# (one per slug) to match the isolation that docker_buildx_push.sh achieves with
# per-slug tags.  CACHE_FROM is set to the main-slug repo as a read-only fallback.
CACHE_SLUG="${CI_COMMIT_REF_SLUG:-main}"
CACHE_REPO="${REGISTRY}/${IMAGE_NAME}-build-cache-${CACHE_SLUG}"

# Submit workflow directly via osmo CLI (no osmo_runner — logs stream live below)
echo "--- Submitting OSMO kaniko build workflow ---"
echo "  Registry: ${REGISTRY}/${IMAGE_NAME}:${TAG_SHORT}"
echo "  Pool:     ${POOL}"
echo "  Platform: ${OSMO_PLATFORM} (${PLATFORM})"
echo "  Priority: ${OSMO_PRIORITY}"

SUBMIT_RC=0
# Pass all --set / --set-string values in a single invocation each.
# With argparse nargs='+', multiple separate --set-string flags don't accumulate
# (last-one-wins); all values must follow a single flag so they're all collected.
SUBMIT_OUTPUT=$(osmo workflow submit ci/docker_build_workflows/docker_build.yaml \
  --pool "${POOL}" \
  --priority "${OSMO_PRIORITY}" \
  --set \
    "CPU_COUNT=${CPU_COUNT}" \
    "GPU_COUNT=${GPU_COUNT}" \
  --set-string \
    "REGISTRY=${REGISTRY}" \
    "IMAGE_NAME=${IMAGE_NAME}" \
    "DOCKERFILE=${DOCKERFILE}" \
    "TAG=${TAG_SHORT}" \
    "PLATFORM=${PLATFORM}" \
    "OSMO_PLATFORM=${OSMO_PLATFORM}" \
    "BUILD_ARGS=${BUILD_ARGS# }" \
    "MEMORY=${MEMORY}" \
    "EXEC_TIMEOUT=${EXEC_TIMEOUT}" \
    "DOCKER_AUTH_CONFIG_B64=${DOCKER_AUTH_CONFIG_B64}" \
    "GITLAB_TOKEN=${GITLAB_TOKEN}" \
    "GITLAB_AUTH_HEADER=${GITLAB_AUTH_HEADER}" \
    "GITLAB_PROJECT_PATH=${GITLAB_PROJECT_PATH}" \
    "GROOT_COMMIT_BRANCH=${GROOT_COMMIT_BRANCH}" \
    "CACHE_REPO=${CACHE_REPO}" \
  2>&1) || SUBMIT_RC=$?

# Always print submit output (error messages appear here on validation failures)
echo "${SUBMIT_OUTPUT}"

if [[ "${SUBMIT_RC}" -ne 0 ]]; then
  echo "ERROR: osmo workflow submit failed (exit ${SUBMIT_RC})" >&2
  exit "${SUBMIT_RC}"
fi

# Extract workflow ID from "Workflow ID - <id>" line in submit output
WORKFLOW_ID=$(echo "${SUBMIT_OUTPUT}" | grep -oP 'Workflow ID\s+-\s+\K\S+' || true)
if [[ -z "${WORKFLOW_ID}" ]]; then
  echo "ERROR: Could not extract workflow ID from submit output" >&2
  exit 1
fi

echo "--- OSMO workflow submitted: ${WORKFLOW_ID} ---"

# Stream logs live by polling osmo workflow logs
PREV_LINE_COUNT=0
while true; do
  # Fetch all current logs and print only lines since the last poll
  CURRENT_LOGS=$(osmo workflow logs "${WORKFLOW_ID}" 2>/dev/null || true)
  if [[ -n "${CURRENT_LOGS}" ]]; then
    CURRENT_LINE_COUNT=$(echo "${CURRENT_LOGS}" | wc -l)
    if [[ "${CURRENT_LINE_COUNT}" -gt "${PREV_LINE_COUNT}" ]]; then
      echo "${CURRENT_LOGS}" | tail -n +"$((PREV_LINE_COUNT + 1))"
      PREV_LINE_COUNT="${CURRENT_LINE_COUNT}"
    fi
  fi

  # Check workflow status (text format: "Status   : <STATUS>")
  STATUS=$(osmo workflow query "${WORKFLOW_ID}" 2>/dev/null \
    | grep -E '^\s*Status\s*:' \
    | sed 's/.*:\s*//' \
    | tr -d '[:space:]' \
    || true)

  case "${STATUS}" in
    COMPLETED)
      echo "--- Docker image pushed: ${REGISTRY}/${IMAGE_NAME}:${TAG_SHORT} ---"
      # Push the stable content tag from here (GitLab machine) — no need to
      # thread it through OSMO; it's just a manifest PUT, no layer data.
      if [[ -n "${CONTENT_TAG}" ]]; then
        echo "--- Pushing content tag :${CONTENT_TAG} ---"
        _crane_tag "${REGISTRY}/${IMAGE_NAME}:${TAG_SHORT}" "${CONTENT_TAG}" \
          || echo "WARNING: content tag push failed (non-fatal)"
      fi
      exit 0
      ;;
    FAILED|FAILED_CANCELED|CANCELLED)
      echo "ERROR: Workflow ${WORKFLOW_ID} failed with status: ${STATUS}" >&2
      exit 1
      ;;
  esac

  sleep 30
done
