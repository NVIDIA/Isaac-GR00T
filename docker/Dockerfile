# syntax=docker/dockerfile:1
# GR00T Docker image (x86_64 + aarch64)
#
# Single Dockerfile for both architectures. On aarch64 (GB200, Grace Hopper)
# torchcodec is built from source since no prebuilt wheel is available.
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
    --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOF
set -e
apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
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
    ffmpeg

python -m pip install --upgrade pip setuptools wheel
curl -LsSf https://astral.sh/uv/0.8.14/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
EOF

WORKDIR /workspace

# Install Python dependencies from lockfile
COPY pyproject.toml uv.lock ./
COPY groot_infra/pyproject.toml groot_infra/pyproject.toml

RUN <<EOF
set -e
UV_PREVIEW=1 UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4 uv sync --frozen --no-install-workspace --all-packages --extra dev --no-cache
EOF

# Install workspace packages
COPY . .
RUN uv pip install --python /workspace/.venv/bin/python -e . --no-deps && \
    uv pip install --python /workspace/.venv/bin/python -e groot_infra --no-deps

ENV PATH="/workspace/.venv/bin:${PATH}" \
    VIRTUAL_ENV="/workspace/.venv"

# EGL/Vulkan setup for headless rendering (MuJoCo, PyOpenGL)
RUN apt-get update && apt-get install -y --no-install-recommends libegl1
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    cat >/usr/share/glvnd/egl_vendor.d/10_nvidia.json <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF

RUN mkdir -p /usr/share/vulkan/icd.d && \
    cat >/usr/share/vulkan/icd.d/nvidia_icd.json <<'EOF'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version": "1.2.140"
    }
}
EOF

ENV MUJOCO_GL="egl" \
    PYOPENGL_PLATFORM="egl" \
    __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"

# torchcodec: no aarch64 wheel, build from source
ARG TORCHCODEC_GIT_URL=https://github.com/pytorch/torchcodec.git
ARG TORCHCODEC_GIT_REF=v0.10.0
RUN ARCH="$(uname -m)" && \
    if [ "${ARCH}" = "aarch64" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            libavdevice-dev libavfilter-dev libavformat-dev libavcodec-dev \
            libavutil-dev libswresample-dev libswscale-dev \
            pkg-config cmake build-essential pybind11-dev \
        && rm -rf /var/lib/apt/lists/* \
        && uv pip install --python /workspace/.venv/bin/python setuptools \
        && git clone --depth 1 --branch "${TORCHCODEC_GIT_REF}" "${TORCHCODEC_GIT_URL}" /tmp/torchcodec \
        && cd /tmp/torchcodec \
        && I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 uv pip install --python /workspace/.venv/bin/python . --no-build-isolation \
        && rm -rf /tmp/torchcodec; \
    else \
        echo "Skipping torchcodec source build (ARCH='${ARCH}', wheel available via uv sync)."; \
    fi

CMD ["/bin/bash"]
