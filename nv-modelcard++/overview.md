# Model Overview

### Description:

NVIDIA Isaac GR00T 2 generates embodiment-specific robot-action sequences and optional future-video predictions from multiview RGB video, natural-language instructions, and robot-state inputs. NVIDIA Isaac GR00T 2 was developed by NVIDIA as part of the NVIDIA Isaac GR00T model family. <br>

This model is for research and development only. <br>

### License/Terms of Use:

**Software and model checkpoint:** [NVIDIA Software and Model Evaluation License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-and-model-evaluation-license/) <br>

### Deployment Geography:

North America and Asia-Pacific, including mainland China, and Germany. <br>

### Use Case: <br>

Twenty selected external robotics partners may privately evaluate and post-train the checkpoint for non-production robotics research and integration testing. Redistribution, monetization, and service-level commitments are excluded. <br>

### Release Date: <br>

GitHub 08/17/2026 via [NVIDIA/Isaac-GR00T-EA](https://github.com/NVIDIA/Isaac-GR00T-EA) <br>

Hugging Face 08/17/2026 via [nvidia/GR00T2Internal](https://huggingface.co/nvidia/GR00T2Internal) <br>

## Model Architecture:

**Architecture Type:** Transformer <br>

**Network Architecture:** World action model built on a Cosmos Predict 2.5 2B diffusion transformer with multiview video tokenization and per-embodiment state/action modules for flow-matched video and action prediction. <br>

**This model was developed based on:** Cosmos Predict 2.5 2B, Cosmos Reason 1.1 7B text encoder. <br>

**Number of model parameters:** `1.1175 x 10^10` <br>

## Input(s): <br>

**Input Type(s):** Video, Text, Other: robot state <br>

**Input Format(s):** <br>

- Video: per-camera RGB `uint8` NumPy arrays <br>
- Text: lists of natural-language instruction strings <br>
- Robot state: `float32` NumPy arrays <br>

**Input Parameters:** <br>

- Video: `(B, T, H, W, 3)`, up to four camera streams, 13 frames per stream, processed at `224 x 320` <br>
- Text: one-dimensional token sequence with a maximum length of 512 tokens <br>
- Robot state: `(B, T, D)` <br>

**Other Properties Related to Input:** Camera ordering, preprocessing, normalization, and embodiment configuration must match the checkpoint. <br>

## Output(s)

**Output Type(s):** Other: continuous robot action; Video <br>

**Output Format(s):** <br>

- Robot action: dictionaries containing `float32` NumPy arrays <br>
- Optional future video: video latents or decoded RGB `uint8` frames <br>

**Output Parameters:** <br>

- Robot action: `(B, T, D)`, with checkpoint maxima of `T = 48` and `D = 128` <br>
- Optional future video: `(B, T, H, W, 3)` per view <br>

**Other Properties Related to Output:** Action representation, dimensionality, and horizon are defined by the checkpoint's embodiment configuration. <br>

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware and software frameworks, the model achieves faster training and inference times compared to CPU-only solutions. <br>

## Software Integration:

**Runtime Engine(s):** PyTorch 2.7.1+cu128 on x86_64 and PyTorch 2.11.0 with CUDA 13 on Linux aarch64, with Hugging Face Transformers 4.57.6 <br>

**Supported Hardware Microarchitecture Compatibility:** NVIDIA Hopper (H100, 80 GB; documented integration smoke-test environment). Linux aarch64 NVIDIA GB300 (compute capability 10.3) is an additional supported host for the locked training and open-loop path. Closed-loop Isaac Sim 5.1.0 stays on x86_64. <br>

**Preferred/Supported Operating System(s):** x86_64 Linux and Linux aarch64 <br>

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment. <br>

## Model Version(s):

**Checkpoint:** [GR00T 2 EA checkpoint](https://huggingface.co/nvidia/GR00T2Internal/tree/4fe662abaf302ee46916881c447377da8c4bd15a/base_200k) <br>

## Training Dataset:

**Data Modality:** Video; Text; Other: robot proprioceptive state and action trajectories <br>

**Training Data Size:** <br>

- Video: 10,000 to 1 Million Hours <br>
- Text: [Less than a Billion Tokens] <br>
- Robot state and action trajectories: [Fewer than 10 Billion Aligned State/Action Timesteps] <br>

**Data Collection Method by dataset:** Hybrid: Automatic/Sensors, Human <br>

**Labeling Method by dataset:** Hybrid: Automatic/Sensors, Human <br>

**Properties:** The training data consists of multimodal robotics data sourced from NVIDIA-created, public open-source, and vendor-provided datasets. It contains multi-camera RGB video, natural-language task or subtask annotations, robot proprioceptive state, and action trajectories. No customer data is used. <br>

### Testing Dataset:

**Data Collection Method by dataset:** Not applicable <br>

**Labeling Method by dataset:** Not applicable <br>

**Properties:** No separate testing dataset was used for this checkpoint. <br>

# Evaluation:

**Benchmark Summary:** No checkpoint-specific evaluation was performed for this EA release. <br>

**Data Collection Method:** Not applicable. <br>

**Labeling Method:** Not applicable. <br>

**Properties:** Not applicable. <br>

# Inference:

**Acceleration Engine:** PyTorch on CUDA using bfloat16, with optional `torch.compile` <br>

**Test Hardware:** NVIDIA H100-class CUDA GPU with at least 80 GB of GPU memory in the documented integration smoke-test environment <br>

## Ethical Considerations:

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

Users are responsible for model inputs and outputs. Users are responsible for ensuring safe integration of this model, including implementing guardrails as well as other safety mechanisms, prior to deployment. <br>

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/). <br>

Please make sure you have proper rights and permissions for all input image and video content; if image or video includes people, personal health information, or intellectual property, the image or video generated will not blur or maintain proportions of image subjects included. <br>
