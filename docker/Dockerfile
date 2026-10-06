# syntax=docker/dockerfile:1
# GR00T Docker image (x86_64 + aarch64)
#
# Single Dockerfile for both architectures. flash-attn is installed from
# official cu12torch2.9 cp312 release wheels (x86_64 + aarch64); on aarch64
# (GB200, Grace Hopper) torchcodec is installed from the prebuilt wheel under
# scripts/deployment/dgpu/wheels/ via a pyproject.toml path source. If that
# wheel is missing, the Docker build bootstraps it from source first.
#
# Build:
#   docker build -f docker/Dockerfile -t gr00t .
#
# Run:
#   docker run -it --rm --gpus all --ipc=host gr00t

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute \
    GR00T_GLOBAL_VENV=/opt/gr00t-venv \
    UV_PROJECT_ENVIRONMENT=/opt/gr00t-venv \
    PYTHON=/usr/bin/python \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# System dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked <<INNEREOF
set -euo pipefail
APT_RETRY_OPTS=(
    -o Acquire::Retries=5
    -o Acquire::http::Timeout=30
    -o Acquire::https::Timeout=30
)
# The CI build network can reach NVIDIA's HTTPS apt repo, but repeatedly times
# out on Ubuntu's default HTTP mirror URLs. Use HTTPS for Ubuntu indexes so apt
# does not depend on port 80 egress from the runner.
find /etc/apt -type f \( -name '*.list' -o -name '*.sources' \) -exec sed -i \
    -e 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' \
    -e 's|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' \
    {} +

apt_update() {
    for attempt in 1 2 3 4 5; do
        if apt-get "${APT_RETRY_OPTS[@]}" update \
            -o APT::Update::Error-Mode=any; then
            return 0
        fi
        rm -rf /var/lib/apt/lists/*
        sleep $((attempt * 10))
    done
    return 1
}

apt_update

apt_install_retry() {
    for attempt in 1 2 3 4 5; do
        if apt-get "${APT_RETRY_OPTS[@]}" install -y --no-install-recommends \
            build-essential \
            ca-certificates \
            git \
            git-lfs \
            curl \
            wget \
            python3.12 \
            python3.12-venv \
            python3.12-dev \
            python3-pip \
            python-is-python3 \
            software-properties-common \
            cmake \
            ninja-build \
            pkg-config \
            pybind11-dev \
            ffmpeg \
            libavdevice-dev \
            libavfilter-dev \
            libavformat-dev \
            libavcodec-dev \
            libavutil-dev \
            libswresample-dev \
            libswscale-dev \
            libegl1; then
            return 0
        fi
        if [ "$attempt" != "5" ]; then
            rm -rf /var/lib/apt/lists/*
            apt_update || true
            sleep $((attempt * 10))
        fi
    done
    return 1
}

apt_install_retry

# The sim-eval setup scripts (LIBERO, SimplerEnv, RoboCasa*) build isolated
# `uv venv --python 3.10` islands whose deps pin an older torch (e.g. 2.5.1),
# so the image must still ship a 3.10 interpreter even though the GR00T root env
# is now 3.12. ubuntu24.04 has no python3.10 in its default repos -> deadsnakes.
add_python310_retry() {
    for attempt in 1 2 3 4 5; do
        if add-apt-repository -y ppa:deadsnakes/ppa \
            && apt_update \
            && apt-get "${APT_RETRY_OPTS[@]}" install -y --no-install-recommends \
                python3.10 \
                python3.10-venv \
                python3.10-dev; then
            return 0
        fi
        if [ "$attempt" != "5" ]; then
            rm -rf /var/lib/apt/lists/*
            apt_update || true
            sleep $((attempt * 10))
        fi
    done
    return 1
}

add_python310_retry

pip_install_retry() {
    for attempt in 1 2 3 4 5; do
        # Ubuntu 24.04's system Python is PEP 668 "externally managed"; this only
        # bootstraps pip/setuptools/wheel before uv builds its own venv, so opting
        # out is safe inside the image. --ignore-installed avoids trying to
        # uninstall debian's pip 24.0 (no RECORD file -> "Cannot uninstall").
        if python -m pip install --upgrade --retries 5 --break-system-packages --ignore-installed pip setuptools wheel; then
            return 0
        fi
        sleep $((attempt * 10))
    done
    return 1
}

pip_install_retry

curl --retry 5 --retry-delay 10 --retry-max-time 300 -LsSf https://astral.sh/uv/0.8.14/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
INNEREOF

WORKDIR /tmp/gr00t-venv-build

# Install Python dependencies from lockfile
# The wheels/ dir must be present before `uv sync` because pyproject.toml
# references the aarch64 torchcodec wheel there as a path source.
COPY pyproject.toml uv.lock ./
COPY scripts/deployment/dgpu/bootstrap_wheels.sh ./scripts/deployment/dgpu/bootstrap_wheels.sh
COPY scripts/deployment/dgpu/wheels/ ./scripts/deployment/dgpu/wheels/

RUN <<INNEREOF
set -euo pipefail

uv_lock_retry() {
    for attempt in 1 2 3 4 5; do
        if UV_PREVIEW=1 UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4 uv lock; then
            return 0
        fi
        sleep $((attempt * 10))
    done
    return 1
}

uv_sync_retry() {
    for attempt in 1 2 3 4 5; do
        if UV_PREVIEW=1 UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4 uv sync --frozen --no-install-project --extra dev --no-cache; then
            return 0
        fi
        sleep $((attempt * 10))
    done
    return 1
}

if [ "$(uname -m)" = "aarch64" ]; then
    bash scripts/deployment/dgpu/bootstrap_wheels.sh
    uv_lock_retry
fi

uv_sync_retry
sha256sum uv.lock | awk '{print $1}' > "${GR00T_GLOBAL_VENV}/.gr00t-uv-lock.sha256"
mkdir -p /opt/gr00t-image-metadata
cp pyproject.toml uv.lock /opt/gr00t-image-metadata/
cd /
rm -rf /tmp/gr00t-venv-build
INNEREOF

WORKDIR /
RUN mkdir -p /workspace && ln -sfn "${GR00T_GLOBAL_VENV}" /workspace/.venv
WORKDIR /workspace

ENV PATH="${GR00T_GLOBAL_VENV}/bin:${PATH}" \
    VIRTUAL_ENV="${GR00T_GLOBAL_VENV}"

# EGL/Vulkan setup for headless rendering (MuJoCo, PyOpenGL)
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    cat >/usr/share/glvnd/egl_vendor.d/10_nvidia.json <<'JSONEOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
JSONEOF

RUN mkdir -p /usr/share/vulkan/icd.d && \
    cat >/usr/share/vulkan/icd.d/nvidia_icd.json <<'JSONEOF'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version": "1.2.140"
    }
}
JSONEOF

ENV MUJOCO_GL="egl" \
    PYOPENGL_PLATFORM="egl" \
    __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"

# aarch64 torchcodec is installed from the repo-local wheel by `uv sync` above
# via a pyproject.toml path source; the wheel is built from source by
# bootstrap_wheels.sh before sync (then committed back) if missing. flash-attn
# comes from official release URLs — no source build needed.

CMD ["/bin/bash"]
