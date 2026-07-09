#!/usr/bin/env sh
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
# Build and push a Docker image using buildx with registry caching.
#
# Single `buildx build --push --cache-to` replaces the old 3-step flow
# (build+load / docker push / build+push for cache), avoiding the ~90s
# --load export overhead on cache hits.
#
# A stable content tag is also used to skip the build entirely when the
# Dockerfile, copied context files, base image digest, build args, and platform
# are unchanged. This keeps per-commit tags available without forcing downstream
# images to see a new base digest on every commit.
#
# Secrets are read from environment variables:
#   GITLAB_TOKEN, BUCKET_ACCESS_KEY_ID, BUCKET_SECRET_ACCESS_KEY
#
# Usage:
#   docker_buildx_push.sh \
#     --registry nvcr.io/nvidian \
#     --image-name groot-ci-minimal \
#     --dockerfile ci/images/minimal.Dockerfile \
#     --cache-slug main \
#     [--tag IMAGE:TAG ...] \
#     [--build-arg KEY=VAL ...]

set -eu

REGISTRY=""
IMAGE_NAME=""
DOCKERFILE=""
CACHE_SLUG=""
TAGS=""
TAG_REFS=""
BUILD_ARGS=""
PLATFORM=""

while [ $# -gt 0 ]; do
  case "$1" in
    --registry)
      REGISTRY="$2"
      shift 2
      ;;
    --image-name)
      IMAGE_NAME="$2"
      shift 2
      ;;
    --dockerfile)
      DOCKERFILE="$2"
      shift 2
      ;;
    --cache-slug)
      CACHE_SLUG="$2"
      shift 2
      ;;
    --tag)
      TAGS="${TAGS} -t $2"
      TAG_REFS="${TAG_REFS} $2"
      shift 2
      ;;
    --build-arg)
      BUILD_ARGS="${BUILD_ARGS} --build-arg $2"
      shift 2
      ;;
    --platform)
      PLATFORM="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [ -z "${REGISTRY}" ] || [ -z "${IMAGE_NAME}" ] || [ -z "${DOCKERFILE}" ] || [ -z "${CACHE_SLUG}" ]; then
  echo "Missing required arguments" >&2
  exit 1
fi

CACHE_REF="${REGISTRY}/${IMAGE_NAME}"
CONTENT_TAG=""
CONTENT_REF=""

_from_uses_build_arg() {
  case "$1" in
    *'${'"$2"'}'* | *'$'"$2"*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

_resolve_base_digest() {
  IMAGE_REF="$1"
  TARGET_PLATFORM="$2"

  # Prefer the runnable platform manifest over the top-level index. BuildKit
  # attestations live in unknown/unknown child manifests and can change the
  # index digest even when the actual image layers are unchanged.
  case "${TARGET_PLATFORM}" in
    "" | *,*)
      ;;
    */*)
      TARGET_OS="${TARGET_PLATFORM%%/*}"
      TARGET_ARCH_VARIANT="${TARGET_PLATFORM#*/}"
      TARGET_ARCH="${TARGET_ARCH_VARIANT%%/*}"

      PLATFORM_DIGEST="$(
        docker buildx imagetools inspect \
          --format "{{range .Manifest.Manifests}}{{if and (eq .Platform.OS \"${TARGET_OS}\") (eq .Platform.Architecture \"${TARGET_ARCH}\")}}{{.Digest}}{{end}}{{end}}" \
          "${IMAGE_REF}" 2>/dev/null || true
      )"
      if [ -n "${PLATFORM_DIGEST}" ]; then
        printf '%s\n' "${PLATFORM_DIGEST}"
        return 0
      fi
      ;;
  esac

  docker buildx imagetools inspect "${IMAGE_REF}" 2>/dev/null \
    | awk '/^Digest:/ { print $2; exit }'
}

_apply_build_args() {
  VALUE="$1"
  set -- ${BUILD_ARGS:-}
  while [ $# -gt 0 ]; do
    if [ "$1" = "--build-arg" ] && [ $# -ge 2 ]; then
      ARG_KEY="${2%%=*}"
      ARG_VALUE="${2#*=}"
      VALUE="$(
        printf '%s' "${VALUE}" \
          | sed "s|\\\${${ARG_KEY}}|${ARG_VALUE}|g; s|\\\$${ARG_KEY}|${ARG_VALUE}|g"
      )"
      shift 2
    else
      shift
    fi
  done
  printf '%s\n' "${VALUE}"
}

_compute_content_hash() {
  HASH_INPUT="$(mktemp)"
  BASE_REF_TEMPLATE="$(
    awk 'toupper($1) == "FROM" && tolower($2) != "scratch" { print $2; exit }' "${DOCKERFILE}"
  )"
  BASE_REF="$(_apply_build_args "${BASE_REF_TEMPLATE}")"

  BASE_DIGEST=""
  if [ -n "${BASE_REF}" ]; then
    BASE_DIGEST="$(_resolve_base_digest "${BASE_REF}" "${PLATFORM:-}")"
  fi

  NORMALIZED_BUILD_ARGS=""
  set -- ${BUILD_ARGS:-}
  while [ $# -gt 0 ]; do
    if [ "$1" = "--build-arg" ] && [ $# -ge 2 ]; then
      ARG_KEY="${2%%=*}"
      if [ -z "${BASE_DIGEST}" ] \
        || ! _from_uses_build_arg "${BASE_REF_TEMPLATE}" "${ARG_KEY}"; then
        NORMALIZED_BUILD_ARGS="${NORMALIZED_BUILD_ARGS}
--build-arg $2"
      fi
      shift 2
    else
      shift
    fi
  done
  NORMALIZED_BUILD_ARGS="$(printf '%s\n' "${NORMALIZED_BUILD_ARGS}" | sort | sed '/^$/d')"

  {
    cat "${DOCKERFILE}"
    printf '\n%s\n%s\n%s\n' "${BASE_DIGEST}" "${NORMALIZED_BUILD_ARGS}" "${PLATFORM:-}"
    awk 'toupper($1) == "COPY" {
      for (i = 2; i < NF; i++) {
        if ($i !~ /^--/) {
          print $i
        }
      }
    }' "${DOCKERFILE}" | while IFS= read -r COPY_SOURCE_TEMPLATE; do
      COPY_SOURCE="$(_apply_build_args "${COPY_SOURCE_TEMPLATE}")"
      printf 'file:%s\n' "${COPY_SOURCE}"
      if [ -f "${COPY_SOURCE}" ]; then
        cat "${COPY_SOURCE}"
        printf '\n'
      else
        printf 'MISSING\n'
      fi
    done
  } >"${HASH_INPUT}"

  printf 'content-%s\n' "$(sha256sum "${HASH_INPUT}" | awk '{ print substr($1, 1, 16) }')"
  rm -f "${HASH_INPUT}"
}

_image_exists() {
  docker buildx imagetools inspect "$1" >/dev/null 2>&1
}

if CONTENT_TAG="$(_compute_content_hash 2>/tmp/docker_buildx_content_hash.err)"; then
  CONTENT_REF="${REGISTRY}/${IMAGE_NAME}:${CONTENT_TAG}"
  echo "--- Content tag: ${CONTENT_REF} ---"
  if _image_exists "${CONTENT_REF}"; then
    echo "--- Content tag found; retagging requested tags and skipping buildx build ---"
    for TAG_REF in ${TAG_REFS}; do
      docker buildx imagetools create -t "${TAG_REF}" "${CONTENT_REF}"
    done
    exit 0
  fi
  echo "--- Content tag not found; proceeding with buildx build ---"
  TAGS="${TAGS} -t ${CONTENT_REF}"
else
  echo "WARNING: failed to compute content tag; proceeding without build skip" >&2
  cat /tmp/docker_buildx_content_hash.err >&2 || true
fi

# shellcheck disable=SC2086
docker buildx build \
  --cache-from "type=registry,ref=${CACHE_REF}:cache-${CACHE_SLUG}" \
  --cache-from "type=registry,ref=${CACHE_REF}:cache-main" \
  --cache-to "type=registry,ref=${CACHE_REF}:cache-${CACHE_SLUG},mode=max" \
  --secret id=gitlab_token,env=GITLAB_TOKEN \
  --secret id=bucket_access_key,env=BUCKET_ACCESS_KEY_ID \
  --secret id=bucket_secret_key,env=BUCKET_SECRET_ACCESS_KEY \
  -f "${DOCKERFILE}" \
  ${TAGS} \
  ${BUILD_ARGS} \
  ${PLATFORM:+--platform "${PLATFORM}"} \
  --push \
  .

echo "Image published: ${TAGS}"
