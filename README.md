<div align="center">


  <img src="media/header_compress.png" width="800" alt="NVIDIA Isaac GR00T N1 Header">
  
  <!-- --- -->
  
  <p style="font-size: 1.2em;">
    <a href="https://developer.nvidia.com/isaac/gr00t"><strong>Website</strong></a> | 
    <a href="https://huggingface.co/nvidia/GR00T-N1-2B"><strong>Model</strong></a> |
    <a href="https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"><strong>Dataset</strong></a> |
    <a href="https://arxiv.org/abs/2503.14734"><strong>Paper</strong></a>
  </p>
</div>

[![CI](https://github.com/NVIDIA/Isaac-GR00T/actions/workflows/main.yml/badge.svg)](https://github.com/NVIDIA/Isaac-GR00T/actions/workflows/main.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![GitHub star chart](https://img.shields.io/github/stars/NVIDIA/Isaac-GR00T?style=flat-square)](https://star-history.com/#NVIDIA/Isaac-GR00T)
[![Open Issues](https://img.shields.io/github/issues-raw/NVIDIA/Isaac-GR00T?style=flat-square)](https://github.com/NVIDIA/Isaac-GR00T/issues)

## NVIDIA Isaac GR00T N1

<div align="center">
<img src="media/robot-demo.gif" width="800" alt="NVIDIA Isaac GR00T N1 Header">
</div>


NVIDIA Isaac GR00T N1 is the world's first open foundation model for generalized humanoid robot reasoning and skills. This cross-embodiment model takes multimodal input, including language and images, to perform manipulation tasks in diverse environments.

GR00T N1 is trained on an expansive humanoid dataset, consisting of real captured data, synthetic data generated using the components of NVIDIA Isaac GR00T Blueprint ([examples of neural-generated trajectories](./media/videos)), and internet-scale video data. It is adaptable through post-training for specific embodiments, tasks and environments.

<div align="center">
<img src="media/real-data.gif" height="150" alt="real-robot-data">
<img src="media/sim-data.gif" height="150" alt="sim-robot-data">
</div>

The neural network architecture of GR00T N1 is a combination of vision-language foundation model and diffusion transformer head that denoises continuous actions. Here is a schematic diagram of the architecture:

<div align="center">
<img src="media/model-architecture.png" width="800" alt="model-architecture">
</div>

Here is the general procedure to use GR00T N1:

1. Assuming the user has already collected a dataset of robot demonstrations in the form of (video, state, action) triplets. 
2. User will first convert the demonstration data into the LeRobot compatible data schema (more info in [`getting_started/LeRobot_compatible_data_schema.md`](getting_started/LeRobot_compatible_data_schema.md)), which is compatible with the upstream [Huggingface LeRobot](https://github.com/huggingface/lerobot).
3. Our repo provides examples to configure different configurations for training with different robot embodiments.
4. Our repo provides convenient scripts to finetune the pre-trained GR00T N1 model on user's data, and run inference.
5. User will connect the `Gr00tPolicy` to the robot controller to execute actions on their target hardware.

## Target Audience

GR00T N1 is intended for researchers and professionals in humanoid robotics. This repository provides tools to:

- Leverage a pre-trained foundation model for robot control
- Fine-tune on small, custom datasets
- Adapt the model to specific robotics tasks with minimal data
- Deploy the model for inference

The focus is on enabling customization of robot behaviors through finetuning.

## Prerequisites
- We have tested the code on Ubuntu 20.04 and 22.04, GPU: H100, L40, RTX 4090 and A6000 for finetuning and Python==3.10, CUDA version 12.4.
- For inference, we have tested on Ubuntu 20.04 and 22.04, GPU: RTX 4090 and A6000
- If you haven't installed CUDA 12.4, please follow the instructions [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) to install it.
- Please make sure you have the following dependencies installed in your system: `ffmpeg`, `libsm6`, `libxext6`

## Installation Guide

Clone the repo:

```sh
git clone https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
```

Create a new conda environment and install the dependencies. We recommend Python 3.10:

> Note that, please make sure your CUDA version is 12.4. Otherwise, you may have a hard time with properly configuring flash-attn module.

```sh
conda create -n gr00t python=3.10
conda activate gr00t
pip install --upgrade setuptools
pip install -e .
pip install --no-build-isolation flash-attn==2.7.1.post4 
```


## Getting started with this repo

We provide accessible Jupyter notebooks and detailed documentations in the [`./getting_started`](./getting_started) folder. Utility scripts can be found in the [`./scripts`](./scripts) folder.

## 1. Data Format & Loading

- To load and process the data, we use [Huggingface LeRobot data](https://github.com/huggingface/lerobot), but with a more detailed modality and annotation schema (we call it "LeRobot compatible data schema").
- An example of LeRobot dataset is stored here: `./demo_data/robot_sim.PickNPlace`. (with additional [`modality.json`](./demo_data/robot_sim.PickNPlace/meta/modality.json) file)
- Detailed explanation of the dataset format is available in [`getting_started/LeRobot_compatible_data_schema.md`](getting_started/LeRobot_compatible_data_schema.md)
- Once your data is organized in this format, you can load the data using `LeRobotSingleDataset` class.

```python
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset import ModalityConfig
from gr00t.experiment.data_config import DATA_CONFIG_MAP

# get the data config
data_config = DATA_CONFIG_MAP["gr1_arms_only"]

# get the modality configs and transforms
modality_config = data_config.modality_config()
transforms = data_config.transform()

# This is a LeRobotSingleDataset object that loads the data from the given dataset path.
dataset = LeRobotSingleDataset(
    dataset_path="demo_data/robot_sim.PickNPlace",
    modality_configs=modality_config,
    transforms=None,  # we can choose to not apply any transforms
    embodiment_tag=EmbodimentTag.GR1, # the embodiment to use
)

# This is an example of how to access the data.
dataset[5]
```

- [`getting_started/0_load_dataset.ipynb`](getting_started/0_load_dataset.ipynb) is an interactive tutorial on how to load the data and process it to interface with the GR00T N1 model.
- [`scripts/load_dataset.py`](scripts/load_dataset.py) is an executable script with the same content as the notebook.


## 2. Inference

* The GR00T N1 model is hosted on [Huggingface](https://huggingface.co/nvidia/GR00T-N1-2B)
* Example cross embodiment dataset is available at [demo_data/robot_sim.PickNPlace](./demo_data/robot_sim.PickNPlace)

```python
from gr00t.model.policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag

# 1. Load the modality config and transforms, or use above
modality_config = ComposedModalityConfig(...)
transforms = ComposedModalityTransform(...)

# 2. Load the dataset
dataset = LeRobotSingleDataset(.....<Same as above>....)

# 3. Load pre-trained model
policy = Gr00tPolicy(
    model_path="nvidia/GR00T-N1-2B",
    modality_config=modality_config,
    modality_transform=transforms,
    embodiment_tag=EmbodimentTag.GR1,
    device="cuda"
)

# 4. Run inference
action_chunk = policy.get_action(dataset[0])
```

- [`getting_started/1_gr00t_inference.ipynb`](getting_started/1_gr00t_inference.ipynb) is an interactive Jupyter notebook tutorial to build an inference pipeline.

User can also run the inference service using the provided script. The inference service can run in either server mode or client mode.

```bash
python scripts/inference_service.py --model_path nvidia/GR00T-N1-2B --server
```

On a different terminal, run the client mode to send requests to the server.
```bash
python scripts/inference_service.py  --client
```

## 3. Fine-Tuning

User can run the finetuning script below to finetune the model with the example dataset. A tutorial is available in [`getting_started/2_finetuning.ipynb`](getting_started/2_finetuning.ipynb).

Then run the finetuning script:
```bash
# first run --help to see the available arguments
python scripts/gr00t_finetune.py --help

# then run the script
python scripts/gr00t_finetune.py --dataset-path ./demo_data/robot_sim.PickNPlace --num-gpus 1
```

You can also download a sample dataset from our huggingface sim data release [here](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim)

```
huggingface-cli download  nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
  --repo-type dataset \
  --include "gr1_arms_only.CanSort/**" \
  --local-dir $HOME/gr00t_dataset
```

The recommended finetuning configurations is to boost your batch size to the max, and train for 20k steps.

*Hardware Performance Considerations*
- **Finetuning Performance**: We used 1 H100 node or L40 node for optimal finetuning. Other hardware configurations (e.g. A6000, RTX 4090) will also work but may take longer to converge. The exact batch size is dependent on the hardware, and on which component of the model is being tuned.
- **Inference Performance**: For real-time inference, most modern GPUs perform similarly when processing a single sample. Our benchmarks show minimal difference between L40 and RTX 4090 for inference speed.

For new embodiment finetuning, checkout our notebook in [`getting_started/3_new_embodiment_finetuning.ipynb`](getting_started/3_new_embodiment_finetuning.ipynb).

## 4. Evaluation

To conduct an offline evaluation of the model, we provide a script that evaluates the model on a dataset, and plots it out.

Run the newly trained model
```bash
python scripts/inference_service.py --server \
    --model_path <MODEL_PATH> \
    --embodiment_tag new_embodiment
```

Run the offline evaluation script
```bash
python scripts/eval_policy.py --plot \
    --dataset_path <DATASET_PATH> \
    --embodiment_tag new_embodiment
```

You will then see a plot of Ground Truth vs Predicted actions, along with unnormed MSE of the actions. This would give you an indication if the policy is performing well on the dataset.


# FAQ

*I have my own data, what should I do next for finetuning?*
- This repo assumes that your data is already organized according to the LeRobot format. 


*What is Modality Config? Embodiment Tag? and Transform Config?*
- Embodiment Tag: Defines the robot embodiment used, non-pretrained embodiment tags are all considered as new embodiment tags.
- Modality Config: Defines the modalities used in the dataset (e.g. video, state, action)
- Transform Config: Defines the Data Transforms applied to the data during dataloading.
- For more details, see [`getting_started/4_deeper_understanding.md`](getting_started/4_deeper_understanding.md)

*What is the inference speed for Gr00tPolicy?*

Below are benchmark results based on a single L40 GPU. Performance is approximately the same on consumer GPUs like RTX 4090 for inference (single sample processing):

| Module | Inference Speed |
|----------|------------------|
| VLM Backbone | 22.92 ms |
| Action Head with 4 diffusion steps | 4 x 9.90ms = 39.61 ms |
| Full Model | 62.53 ms |

We noticed that 4 denoising steps are sufficient during inference.

# Contributing

For more details, see [CONTRIBUTING.md](CONTRIBUTING.md)


## License 

```
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
```
