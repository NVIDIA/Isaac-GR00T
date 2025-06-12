FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace:${PYTHONPATH}

# System dependencies
RUN apt update && \
    apt install -y tzdata && \
    ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime && \
    apt install -y netcat dnsutils && \
    apt-get update && \
    apt-get install -y libgl1-mesa-glx git libvulkan-dev \
    zip unzip wget curl git git-lfs build-essential cmake \
    vim less sudo htop ca-certificates man tmux ffmpeg tensorrt \
    # Add OpenCV system dependencies
    libglib2.0-0 libsm6 libxext6 libxrender-dev

RUN pip install --upgrade pip setuptools
RUN pip install gpustat wandb==0.19.0
# Create and set working directory
WORKDIR /workspace
# Copy pyproject.toml for dependencies
COPY pyproject.toml .
# Install dependencies from pyproject.toml
RUN pip install -e .[base]
# There's a conflict in the native python, so we have to resolve it by
RUN pip uninstall -y transformer-engine
RUN pip install flash_attn==2.7.1.post4 -U --force-reinstall
# Clean any existing OpenCV installations
RUN pip uninstall -y opencv-python opencv-python-headless || true
RUN rm -rf /usr/local/lib/python3.10/dist-packages/cv2 || true
RUN pip install opencv-python==4.8.0.74
RUN pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 numpy==1.26.4
COPY getting_started /workspace/getting_started
COPY scripts /workspace/scripts
COPY demo_data /workspace/demo_data
RUN pip install -e . --no-deps
# need to install accelerate explicitly to avoid version conflicts
RUN pip install accelerate>=0.26.0
