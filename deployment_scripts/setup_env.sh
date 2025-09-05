#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-dev \
      python3-antlr4 \
      libsm6 \
      libxext6 \
      ffmpeg \
      libhdf5-serial-dev \
      libtesseract-dev \
      libgtk-3-0 \
      libtbb12 \
      libtbb2 \
      libatlas-base-dev \
      libopenblas-dev \
      build-essential \
      python3-setuptools \
      make \
      cmake \
      nasm

# Install cuDSS (CUDA Deep Neural Network library)
echo "Installing cuDSS..."
cd /tmp
wget https://developer.download.nvidia.com/compute/cudss/0.6.0/local_installers/cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb
sudo dpkg -i cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb
sudo cp /var/cudss-local-tegra-repo-ubuntu2204-0.6.0/cudss-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudss

# Clean up downloaded package
rm -f cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb

# Set to get precompiled jetson wheels
export PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126
export PIP_TRUSTED_HOST=pypi.jetson-ai-lab.io

pip3 install --upgrade pip setuptools
pip3 install -e .[orin]

# Build and install decord
echo "Building and installing decord..."
# Change to /tmp for builds
cd /tmp
# Clone the FFmpeg repository
git clone https://git.ffmpeg.org/ffmpeg.git
cd ffmpeg
git checkout n4.4.2
./configure \
  --enable-shared \
  --enable-pic \
  --prefix=/usr
make -j$(nproc)
sudo make install
cd /tmp
# Build and install decord
git clone --recursive https://github.com/dmlc/decord
cd decord
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
cd ../python
python3 setup.py install --user
cd /tmp
rm -rf /tmp/ffmpeg /tmp/decord
