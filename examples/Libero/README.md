# GR00T Libero Benchmarks

This directory contains fine-tuning and evaluation scripts for **GR00T N1.5** on the Libero benchmark suite.



## 🎯 Model Evaluation

Evaluation is performed using [`run_libero_eval.py`](https://github.com/NVIDIA/Isaac-GR00T/examples/Libero/eval/run_libero_eval.py).

<!-- Spatial: /mnt/amlfs-02/shared/checkpoints/xiaoweij/0827/libero-checkpoints-20K/checkpoint-20000/ -->
<!-- Goal: /mnt/amlfs-02/shared/checkpoints/xiaoweij/0911/libero-goal-checkpoints-20K/ https://wandb.ai/nv-gear/huggingface/runs/wibov9ph?nw=nwuserxiaoweij -->
<!-- Object: /mnt/amlfs-02/shared/checkpoints/xiaoweij/0904/libero-object-checkpoints-20K/ https://wandb.ai/nv-gear/huggingface/runs/38tmzwcw?nw=nwuserxiaoweij -->
<!-- Libero-90: /mnt/amlfs-02/shared/checkpoints/xiaoweij/0905/libero-90-checkpoints-60K/  https://wandb.ai/nv-gear/huggingface/runs/3wpxrsri?nw=nwuserxiaoweij -->
<!-- Libero-Long: /mnt/amlfs-02/shared/checkpoints/xiaoweij/0908/libero-10-checkpoints-60K/ https://wandb.ai/nv-gear/huggingface/runs/cyh7mdtx?nw=nwuserxiaoweij  -->

### Eval Result and Training Config Table

| Task      | Success rate       | max_steps | grad_accum_steps | batch_size | Data config                                                     |                   Checkpoint                  |
|-----------|--------------------|-----------|------------------|------------|-----------------------------------------------------------------|-----------------------------------------------|
| Spatial   | 46/50 (92%)        | 20K       | 1                | 128        | examples.Libero.custom_data_config:LiberoDataConfig             |youliangtan/gr00t-n1.5-libero-spatial-posttrain|
| Goal      | 43/50 (86%)        | 20K       | 4                | 72         | examples.Libero.custom_data_config:LiberoDataConfigMeanStd      |youliangtan/gr00t-n1.5-libero-goal-posttrain|
| Object    | 46/50 (92%)        | 20K       | 1                | 128        | examples.Libero.custom_data_config:LiberoDataConfig             |youliangtan/gr00t-n1.5-libero-object-posttrain|
| Libero-90 | 402/450 (89.3%)    | 60K       | 1                | 128        | examples.Libero.custom_data_config:LiberoDataConfig             |youliangtan/gr00t-n1.5-libero-90-posttrain|
| Long      | 38/50 (76%)        | 60K       | 1                | 128        | examples.Libero.custom_data_config:LiberoDataConfig             |youliangtan/gr00t-n1.5-libero-long-posttrain|





> Note: The results reported above were obtained with minimal hyperparameter tuning and are intended primarily for demonstration purposes. More comprehensive studies have fine-tuned GR00T on LIBERO and achieved substantially higher performance. For example, see Table 3 in this [paper](https://arxiv.org/pdf/2508.21112).
----

To evaluate, first start the inference server with our provided checkpoint:

```bash
python scripts/inference_service.py \
    --model_path youliangtan/gr00t-n1.5-libero-spatial-posttrain \
    --server \
    --data_config examples.Libero.custom_data_config:LiberoDataConfig \
    --denoising-steps 8 \
    --port 5555 \
    --embodiment-tag new_embodiment
```

> Note, for **Libero-Goal**, the checkpoints are trained using data config: `examples.Libero.custom_data_config:LiberoDataConfigMeanStd`. So the corresponding checkpoints should be served using commands:
```bash
python scripts/inference_service.py \
    --model_path /mnt/amlfs-02/shared/checkpoints/xiaoweij/0913/libero-goal-checkpoints-20K/checkpoint-20000 \
    --server \
    --data_config examples.Libero.custom_data_config:LiberoDataConfigMeanStd \
    --denoising-steps 8 \
    --port 5555 \
    --embodiment-tag new_embodiment
```
----

### Installation

Follow the [official Libero installation guide](https://lifelong-robot-learning.github.io/LIBERO/html/getting_started/installation.html).

### Troubleshooting

If you see:
```
ModuleNotFoundError: No module named 'robosuite.environments.manipulation.single_arm_env'
```

Make sure you install:
```bash
pip install robosuite==1.4.0
```

Then run the evaluation:
```bash
cd examples/Libero/eval
python run_libero_eval.py --task_suite_name libero_spatial
```

----

## Reproduce Training Results

To reproduce the training results, you can use the following steps:
1. Download the datasets
2. Add the modality configuration files
3. Fine-tune the model
4. Evaluate the model (same as above)

## 📦 1. Dataset Preparation

### Dataset Downloads
Download LeRobot-compatible datasets directly from Hugging Face.

```bash
huggingface-cli download \
    --repo-type dataset IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
    --local-dir /tmp/libero_spatial/
```

> 🔄 Replace with the appropriate dataset name:
> - `IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot` (for **goal**)
> - `IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot` (for **object**)
> - `IPEC-COMMUNITY/libero_90_no_noops_lerobot` (for **libero-90**)
> - `IPEC-COMMUNITY/libero_10_no_noops_lerobot` (for **libero-10**)

### Modality Configuration

After downloading the datasets, you need to add the appropriate modality configuration files to make them compatible with GR00T N1.5. These configuration files define the observation and action space mappings.

```bash
cp examples/Libero/modality.json /tmp/libero_spatial/meta/modality.json
```

## 🚀 Model Fine-tuning

### Training Commands

The fine-tuning script supports multiple configurations.

```bash
python scripts/gr00t_finetune.py \
    --dataset-path /tmp/libero_spatial/ \
    --data_config examples.Libero.custom_data_config:LiberoDataConfig \
    --num-gpus 8 \
    --batch-size 128 \
    --output-dir /tmp/my_libero_spatial_checkpoint/ \
    --max-steps 60000 \
    --video-backend torchvision_av
```
> Note, replace with the corresponding data config class and training configs according to the [table](#training-config-table).
