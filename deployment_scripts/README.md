# Deployment Scripts

## Inference with Python TensorRT on X86

Export ONNX model
```bash
python deployment_scripts/export_onnx.py
```
Build TensorRT engine
```bash
bash deployment_scripts/build_engine.sh
```
Inference with TensorRT
```bash
python deployment_scripts/gr00t_inference.py --inference_mode=tensorrt
```

---

## Jetson Deployment

A detailed guide for deploying GR00T N1.5 on Orin is available in [`orin/README.md`](orin/README.md).

### Prerequisites

- Jetson Thor installed with Jetpack 7.0

### 1. Installation Guide

Clone the repo:

```sh
git clone https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
```

### Deploy Isaac-GR00T with Container

#### Build Container

To build a container for Isaac-GR00T:

Build container for Jetson Thor:
```sh
docker build -t isaac-gr00t-n1.5:l4t-jp7.0 -f thor.Dockerfile .
```

#### Run Container

To run the container:

Run container for Thor:
```sh
docker run --rm -it --runtime nvidia -v "$PWD":/workspace -w /workspace -p 5555:5555 isaac-gr00t-n1.5:l4t-jp7.0
```

### 2. Inference

* The GR00T N1.5 model is hosted on [Huggingface](https://huggingface.co/nvidia/GR00T-N1.5-3B)
* Example cross embodiment dataset is available at [demo_data/robot_sim.PickNPlace](./demo_data/robot_sim.PickNPlace)
* This project supports to run the inference with PyTorch or Python TensorRT as instructions below
* Add Isaac-GR00T to PYTHONPATH: `export PYTHONPATH=/path/to/Isaac-GR00T:$PYTHONPATH`

### 2.1 Inference with PyTorch

```bash
python deployment_scripts/gr00t_inference.py --inference-mode=pytorch
```

### 2.2 Inference with Python TensorRT

Export ONNX model
```bash
python deployment_scripts/export_onnx.py
```

Build TensorRT engine
```bash
bash deployment_scripts/build_engine.sh
```

Inference with TensorRT
```bash
python deployment_scripts/gr00t_inference.py --inference-mode=tensorrt
```

## 3. Performance
### 3.1 Pipline Performance
Here's comparison of E2E performance between PyTorch and TensorRT on Thor:

<div align="center">
<img src="../media/thor-perf.png" width="1200" alt="thor-perf">
</div>

### 3.2 Models Performance
Model latency measured by `trtexec` with batch_size=1.     
| Model Name                                     |Thor benchmark perf (ms) (FP16) |Thor benchmark perf (ms) (FP8+FP4) |
|:----------------------------------------------:|:------------------------------:|:---------------------------------:|
| Action_Head - process_backbone_output          | 2.35                           | /                                 |
| Action_Head - state_encoder                    | 0.04                           | /                                 |
| Action_Head - action_encoder                   | 0.10                           | /                                 |
| Action_Head - DiT                              | 5.46                           | 3.41                              |
| Action_Head - action_decoder                   | 0.03                           | /                                 |
| VLM - ViT                                      | 5.21                           | 4.10                              |
| VLM - LLM                                      | 8.53                           | 5.81                              |

**Note**: The module latency (e.g., DiT Block) in pipeline is slightly longer than the model latency in benchmark table above because the module (e.g., Action_Head - DiT) latency not only includes the model latency in table above but also accounts for the overhead of data transfer from PyTorch to TRT and returning from TRT to PyTorch.
