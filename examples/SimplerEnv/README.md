# GR00T SimplerEnv Benchmarks

This directory contains fine-tuning and evaluation scripts for GR00T N1.5 on simulation benchmarks, specifically targeting Bridge WidowX and Fractal Google Robot tasks.

## 📦 Dataset Preparation

### Dataset Downloads
Download LeRobot-compatible datasets directly from Hugging Face.

#### 1. Bridge Dataset
```bash
huggingface-cli download \
    --repo-type dataset IPEC-COMMUNITY/bridge_orig_lerobot \
    --local-dir /tmp/bridge_orig_lerobot/
```

#### 2. Fractal Dataset
```bash
huggingface-cli download \
    --repo-type dataset IPEC-COMMUNITY/fractal20220817_data_lerobot \
    --local-dir /tmp/fractal20220817_data_lerobot/
```

### Modality Configuration

After downloading the datasets, you need to add the appropriate modality configuration files to make them compatible with GR00T N1.5. These configuration files define the observation and action space mappings.

#### 1. Bridge Dataset
Copy the Bridge modality configuration to your dataset:
```bash
cp examples/SimplerEnv/bridge_modality.json /tmp/bridge_orig_lerobot/meta/modality.json
```

#### 2. Fractal Dataset
Copy the Fractal modality configuration to your dataset:
```bash
cp examples/SimplerEnv/fractal_modality.json /tmp/fractal20220817_data_lerobot/meta/modality.json
```


## 🚀 Model Fine-tuning

### Training Commands

The fine-tuning script supports multiple configurations. Below are examples for each simulation environment:

#### 1. Bridge Dataset
```bash
python scripts/gr00t_finetune.py \
    --dataset-path /tmp/bridge_orig_lerobot/ \
    --data_config examples.SimplerEnv.custom_data_config:BridgeDataConfig \
    --num-gpus 8 \
    --batch-size 64 \
    --output-dir /tmp/bridge-checkpoints \
    --max-steps 60000 \
    --video-backend torchvision_av
```

#### 2. Fractal Dataset
```bash
python scripts/gr00t_finetune.py \
    --dataset-path /tmp/fractal20220817_data_lerobot/ \
    --data_config examples.SimplerEnv.custom_data_config:FractalDataConfig \
    --num-gpus 8 \
    --batch-size 128 \
    --output-dir /tmp/fractal-checkpoints/ \
    --max-steps 60000 \
    --video-backend torchvision_av
```

## 🎯 Model Evaluation

Evaluation is performed using the [SimplerEnv repository](https://github.com/youliangtan/SimplerEnv/tree/main).

### 1. Bridge/WidowX

TODO: Add checkpoint link
| Task | Result |
|------|--------|
| widowx_spoon_on_towel  | 40/50 (80.0%) |
| widowx_carrot_on_plate | 33/50 (66.0%) |
| widowx_put_eggplant_in_basket | 29/50 (58.0%) |
| widowx_put_eggplant_in_sink | 5/50  (10.0%) |
| widowx_close_drawer | 11/50 (22.0%) |
| widowx_open_drawer | 21/50 (42.0%) |
| widowx_stack_cube | 28/50 (56.0%) |

To evaluate, first start the inference server:
```bash
python scripts/inference_service.py \
    --model-path /tmp/bridge-checkpoints/checkpoint-60000/ \
    --server \
    --data_config examples.SimplerEnv.custom_data_config:BridgeDataConfig \
    --denoising-steps 8 \
    --port 5555 \
    --embodiment-tag new_embodiment
```

Then run the evaluation:
```bash
python eval_simpler.py --env widowx_spoon_on_towel --groot_port 5555
```

### 2. Fractal/Google Robot

TODO: Add checkpoint link
| Task | Result |
|------|--------|
| google_robot_pick_coke_can | xx% |
| google_robot_pick_object | xx% |
| google_robot_move_near | xx% |
| google_robot_open_drawer | xx% |
| google_robot_close_drawer | xx% |
| google_robot_place_in_closed_drawer | xx% |

To evaluate, first start the inference server:
```bash
python scripts/inference_service.py \
    --model-path /tmp/fractal-checkpoints/checkpoint-60000/ \
    --server \
    --data_config examples.SimplerEnv.custom_data_config:FractalDataConfig \
    --denoising-steps 8 \
    --port 5555 \
    --embodiment-tag new_embodiment
```

Then run the evaluation:
```bash
python eval_simpler.py --env google_robot_pick_object --groot_port 5555
```
