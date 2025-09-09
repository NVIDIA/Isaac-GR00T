# GR00T Libero Benchmarks

This directory contains fine-tuning and evaluation scripts for **GR00T N1.5** on the Libero benchmark suite.



## 🎯 Model Evaluation

Evaluation is performed using [`run_libero_eval.py`](https://github.com/NVIDIA/Isaac-GR00T/examples/Libero/eval/run_libero_eval.py).

<!-- TODO: Upload the checkpoint to Youliang's HF repo. -->
<!-- TODO: Update with new number for Goal. -->

| Task                              | Success rate (300) |
| --------------------------------- | ------------------ |
| Spatial                           | 46/50 (92%)      |
| Goal                              | 38/50 (76%)      |
| Object                            | 46/50 (92%)      |
| Libero-90                         | 402/450 (89.3%)      |

To evaluate, first start the inference server with our provided checkpoint:

<!-- TODO: Replace with Youliang's repo. -->
```bash
python scripts/inference_service.py \
    --model_path /mnt/amlfs-02/shared/checkpoints/xiaoweij/0827/libero-checkpoints-20K/checkpoint-20000 \
    --server \
    --data_config examples.Libero.custom_data_config:LiberoDataConfig \
    --denoising-steps 8 \
    --port 5555 \
    --embodiment-tag new_embodiment
```

Then run the evaluation:
```bash
cd libero_eval
python run_libero_eval.py --task_suite_name spatial
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
