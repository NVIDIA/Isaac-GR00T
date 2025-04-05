# Creating a Dockerfile to run the Isaac-GROOT by NVIDIA
# Importing nvidia/cuda image with CUDA12.4 support 
# Making sure we have Ubuntu image
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Setting the frontend to be non-interactive
# This is to avoid any user input required during the installation of packages
ENV DEBIAN_FRONTEND=noninteractive

# Updating repositories and installing necessary packages for installing Conda and Python Libraries
RUN apt-get update && apt-get install wget -y\
        ffmpeg \
        libopencv-dev \
        git \
	python3.10 \
	python3-pip 

RUN git clone https://github.com/nvidia/Isaac-GR00T.git /GR00T

WORKDIR /GR00T

RUN <<EOT bash
pip install --upgrade setuptools
pip install -e .
pip install --no-build-isolation flash-attn==2.7.1.post4
pip install notebook
EOT

ENTRYPOINT [ "/bin/bash" ]
CMD ["/usr/local/bin/jupyter-notebook", "--allow-root", "--no-browser", "--ip=0.0.0.0", "--port=8888"]

