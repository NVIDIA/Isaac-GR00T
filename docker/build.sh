#!/bin/bash

set -x

export DOCKER_BUILDKIT=1
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO_ROOT="$DIR/.."

# Parse --profile and filter script-specific flags before passing to docker
profile="default"
docker_args=()
for arg in "$@"; do
    case $arg in
        --profile=*)
            profile="${arg#--profile=}"
            ;;
        --fix)
            # Skip --fix flag as it's not a valid docker build flag
            ;;
        *)
            docker_args+=("$arg")
            ;;
    esac
done

if [ "$profile" = "thor" ]; then
    image_name="gr00t-thor"
    docker build "${docker_args[@]}" \
        --network host \
        -f "$REPO_ROOT/scripts/deployment/thor/Dockerfile" \
        -t "$image_name" "$REPO_ROOT" \
        && echo "Image $image_name BUILT SUCCESSFULLY"
elif [ "$profile" = "spark" ]; then
    image_name="gr00t-spark"
    docker build "${docker_args[@]}" \
        --network host \
        -f "$REPO_ROOT/scripts/deployment/spark/Dockerfile" \
        -t "$image_name" "$REPO_ROOT" \
        && echo "Image $image_name BUILT SUCCESSFULLY"
elif [ "$profile" = "orin" ]; then
    image_name="gr00t-orin"
    docker build "${docker_args[@]}" \
        --network host \
        -f "$REPO_ROOT/scripts/deployment/orin/Dockerfile" \
        -t "$image_name" "$REPO_ROOT" \
        && echo "Image $image_name BUILT SUCCESSFULLY"
else
    image_name="gr00t"
    docker build "${docker_args[@]}" \
        --network host \
        -f "$DIR/Dockerfile" \
        -t "$image_name" "$REPO_ROOT" \
        && echo "Image $image_name BUILT SUCCESSFULLY"
fi
