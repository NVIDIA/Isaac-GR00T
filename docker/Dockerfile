# syntax=docker/dockerfile:1
# GR00T Docker image (x86_64 + aarch64)
#
# Single Dockerfile for both architectures. On aarch64 (GB200, Grace Hopper)
# flash-attn and torchcodec are installed from prebuilt wheels under
# scripts/deployment/dgpu/wheels/ via pyproject.toml path sources.
#
# Build:
#   docker build -f docker/Dockerfile -t gr00t .
#
# Run:
#   docker run -it --rm --gpus all --ipc=host gr00t

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute \
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
            python3.10 \
            python3.10-venv \
            python3.10-dev \
            python3-pip \
            python-is-python3 \
            software-properties-common \
            ffmpeg \
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

pip_install_retry() {
    for attempt in 1 2 3 4 5; do
        if python -m pip install --upgrade --retries 5 pip setuptools wheel; then
            return 0
        fi
        sleep $((attempt * 10))
    done
    return 1
}

pip_install_retry

curl --retry 5 --retry-delay 10 --retry-max-time 300 -LsSf https://astral.sh/uv/0.8.14/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
INNEREOF

WORKDIR /workspace

# Install Python dependencies from lockfile
# The wheels/ dir must be present before `uv sync` because pyproject.toml
# references them as path sources for aarch64 flash-attn and torchcodec.
COPY pyproject.toml uv.lock ./
COPY scripts/deployment/dgpu/wheels/ ./scripts/deployment/dgpu/wheels/

RUN <<INNEREOF
set -euo pipefail

uv_sync_retry() {
    for attempt in 1 2 3 4 5; do
        if UV_PREVIEW=1 UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4 uv sync --frozen --no-install-project --extra dev --no-cache; then
            return 0
        fi
        sleep $((attempt * 10))
    done
    return 1
}

uv_sync_retry
INNEREOF

# Install the gr00t project itself
COPY . .
RUN <<INNEREOF
set -euo pipefail

uv_pip_install_retry() {
    for attempt in 1 2 3 4 5; do
        if uv pip install --python /workspace/.venv/bin/python -e . --no-deps; then
            return 0
        fi
        sleep $((attempt * 10))
    done
    return 1
}

uv_pip_install_retry
INNEREOF

ENV PATH="/workspace/.venv/bin:${PATH}" \
    VIRTUAL_ENV="/workspace/.venv"

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

# torchcodec and flash-attn for aarch64 are installed from prebuilt wheels
# by `uv sync` above via pyproject.toml path sources — no source build needed.

CMD ["/bin/bash"]
