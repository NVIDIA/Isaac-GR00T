#!/bin/bash

set -x

image_name="gr00t-dev"

export DOCKER_BUILDKIT=1
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Copy gr00t directory to src/gr00t
mkdir -p $DIR/src
rm -rf /tmp/gr00t

echo $DIR

cp -r $DIR/../ /tmp/gr00t
cp -r /tmp/gr00t $DIR/src/

export DOCKER_BUILDKIT=1

# Filter out --fix flag and other script-specific flags before passing to docker
docker_args=()
for arg in "$@"; do
    case $arg in
        --fix)
            # Skip --fix flag as it's not a valid docker build flag
            ;;
        *)
            docker_args+=("$arg")
            ;;
    esac
done

docker build "${docker_args[@]}" \
    --platform linux/amd64 \
    --network host \
    -t $image_name $DIR \
    && echo Image $image_name BUILT SUCCESSFULLY

rm -rf $DIR/src/