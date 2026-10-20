#!/usr/bin/env sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Build and push a Docker image using buildx with registry caching.
#
# Single `buildx build --push --cache-to` replaces the old 3-step flow
# (build+load / docker push / build+push for cache), avoiding the ~90s
# --load export overhead on cache hits.
#
# A stable content tag is also used to skip the build entirely when the
# Dockerfile, copied context files, the digests of every FROM and COPY --from
# image, build args, and platform are unchanged. This keeps per-commit tags available without forcing downstream
# images to see a new base digest on every commit.
#
# Secrets are read from environment variables:
#   GITLAB_TOKEN, BUCKET_ACCESS_KEY_ID, BUCKET_SECRET_ACCESS_KEY
#
# Usage:
#   docker_buildx_push.sh \
#     --registry nvcr.io/nvidian \
#     --image-name groot-ci-oss \
#     --dockerfile docker/Dockerfile \
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
  set -- ${RESOLVE_ARGS:-${BUILD_ARGS:-}}
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

# Global ARG defaults (declared before the first FROM). Values with spaces are
# skipped because the resolved argument list is word-split.
_global_arg_defaults() {
  awk 'toupper($1) == "FROM" { exit }
    toupper($1) == "ARG" && NF == 2 && index($2, "=") { print "--build-arg " $2 }' "${DOCKERFILE}"
}

# Prints "ref <template>" for every FROM image and COPY --from source and
# "stage <name>" for every named stage, in Dockerfile order.
_image_references() {
  awk 'toupper($1) == "FROM" {
      i = 2
      while ($i ~ /^--/) i++
      print "ref " $i
      if (toupper($(i + 1)) == "AS") print "stage " $(i + 2)
    }
    toupper($1) == "COPY" {
      for (i = 2; i <= NF; i++) if ($i ~ /^--from=/) print "ref " substr($i, 8)
    }' "${DOCKERFILE}"
}

# Prints "base:<ref>@<digest>" for every external image the build reads, and
# fails when one cannot be resolved so that the build is never skipped on an
# unknown base.
_base_image_digests() {
  STAGES=" "
  _image_references | while read -r KIND NAME; do
    if [ "${KIND}" = "stage" ]; then
      STAGES="${STAGES}${NAME} "
      continue
    fi
    REF="$(_apply_build_args "${NAME}")"
    # Skip scratch, numeric stage indexes (COPY --from=0) and named stages.
    case "${REF}" in
      scratch) continue ;;
      *[!0-9]*) ;;
      *) continue ;;
    esac
    case "${STAGES}" in
      *" ${REF} "*) continue ;;
    esac
    case "${REF}" in
      *'$'*)
        echo "unresolved image reference: ${NAME}" >&2
        exit 1
        ;;
    esac
    DIGEST="$(_resolve_base_digest "${REF}" "${PLATFORM:-}")"
    if [ -z "${DIGEST}" ]; then
      echo "cannot resolve the digest of ${REF}" >&2
      exit 1
    fi
    printf 'base:%s@%s\n' "${REF}" "${DIGEST}"
  done
}

_compute_content_hash() {
  HASH_INPUT="$(mktemp)"
  PLATFORM_ARGS=""
  case "${PLATFORM:-}" in
    "" | *,*) ;;
    */*)
      TARGET_ARCH_VARIANT="${PLATFORM#*/}"
      PLATFORM_ARGS="--build-arg TARGETOS=${PLATFORM%%/*} --build-arg TARGETARCH=${TARGET_ARCH_VARIANT%%/*}"
      ;;
  esac
  # Explicit build args come first so they win over the Dockerfile defaults.
  RESOLVE_ARGS="${BUILD_ARGS:-} $(_global_arg_defaults | tr '\n' ' ') ${PLATFORM_ARGS}"
  BASE_DIGESTS="$(_base_image_digests)" || return 1
  REF_TEMPLATES="$(_image_references | awk '$1 == "ref" { print $2 }')"

  NORMALIZED_BUILD_ARGS=""
  set -- ${BUILD_ARGS:-}
  while [ $# -gt 0 ]; do
    if [ "$1" = "--build-arg" ] && [ $# -ge 2 ]; then
      ARG_KEY="${2%%=*}"
      if ! _from_uses_build_arg "${REF_TEMPLATES}" "${ARG_KEY}"; then
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
    printf '\n%s\n%s\n%s\n' "${BASE_DIGESTS}" "${NORMALIZED_BUILD_ARGS}" "${PLATFORM:-}"
    awk 'toupper($1) == "COPY" && $0 !~ /--from=/ {
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

docker buildx create --use --driver docker-container --name ci-builder

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
