#!/bin/bash
set -euo pipefail
set -x

image_name="gr00t-dev"

export DOCKER_BUILDKIT=1
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Copy gr00t directory to src/gr00t
mkdir -p "$DIR/src"
rm -rf /tmp/gr00t

cp -r "$DIR/../" /tmp/gr00t
cp -r /tmp/gr00t "$DIR/src/"

# Defaults
profile="server"
dockerfile="Dockerfile"

# Filter out --fix flag and other script-specific flags before passing to docker
docker_args=()
for arg in "$@"; do
  case "$arg" in
    --fix)
      # script-only flag, ignore
      ;;
    profile=orin)
      profile="orin"
      dockerfile="orin.Dockerfile"
      ;;
    profile=thor)
      profile="thor"
      dockerfile="thor.Dockerfile"
      ;;
    profile=server)
      profile="server"
      dockerfile="Dockerfile"
      ;;
    *)
      docker_args+=("$arg")
      ;;
  esac
done

# Build
docker build "${docker_args[@]}" \
  --file "$DIR/$dockerfile" \
  -t "$image_name" \
  "$DIR" \
  && echo "Image $image_name BUILT SUCCESSFULLY (profile=$profile)"

# Cleanup
rm -rf "$DIR/src/"