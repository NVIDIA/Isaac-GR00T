# GR00T SimplerEnv Benchmarks

This directory contains fine-tuning and evaluation scripts for GR00T N1.5 on simulation benchmarks, specifically targeting Bridge WidowX and Fractal Google Robot tasks.


## 🎯 Model Evaluation

Evaluation is performed using the [SimplerEnv repository](https://github.com/youliangtan/SimplerEnv/tree/main).

> Note: The results reported below were obtained with minimal hyperparameter tuning and are intended primarily for demonstration purposes. 

### 1. Bridge/WidowX

Provided checkpoint: youliangtan/gr00t-n1.5-bridge-posttrain 

| Task                              | Success rate (300) |
| --------------------------------- | ------------------ |
| widowx\_spoon\_on\_towel          | 246/300 (82%)      |
| widowx\_carrot\_on\_plate         | 216/300 (72%)      |
| widowx\_put\_eggplant\_in\_basket | 189/300 (63%)      |
| widowx\_stack\_cube               | 162/300 (54%)      |
| widowx\_put\_eggplant\_in\_sink** | 62/300 (21%)        |
| widowx\_close\_drawer**           | 196/300 (65%)       |
| widowx\_open\_drawer**            | 251/300 (84%)      |
| **Average**                       | **63%**            |

**Denotes as "non-original" new simpler task [here](https://github.com/youliangtan/SimplerEnv)

To evaluate, first start the inference server with our provided checkpoint:
```bash
python scripts/inference_service.py \
    --model-path youliangtan/gr00t-n1.5-bridge-posttrain \
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

Provided checkpoint: youliangtan/gr00t-n1.5-fractal-posttrain

To evaluate, first start the inference server with our provided checkpoint:
```bash
python scripts/inference_service.py \
    --model-path youliangtan/gr00t-n1.5-fractal-posttrain \
    --server \
    --data_config examples.SimplerEnv.custom_data_config:FractalDataConfig \
    --denoising-steps 8 \
    --port 5555 \
    --embodiment-tag new_embodiment
```

#### Visual Matching
| Task                                     | Success Rate (300) |
| ---------------------------------------- | ------------------ |
| google\_robot\_pick\_coke\_can           | 208/300 (69%)      |
| google\_robot\_pick\_object              | 97/300 (32%)       |
| google\_robot\_move\_near                | 206/300 (69%)      |
| google\_robot\_open\_drawer              | 80/300 (27%)       |
| google\_robot\_close\_drawer             | 135/300 (45%)      |
| google\_robot\_place\_in\_closed\_drawer | 12/300 (4%)        |
| **Average**                              | **41%**            |
----

```bash
python eval_simpler.py --env google_robot_pick_object --groot_port 5555
```

#### Variant Aggregations
| Environment   | Runs | Total Successes | Total Trials | Average Success Rate |
|---------------|------|-----------------|--------------|-----------------------|
| Drawer        | 42   | 66              | 378          | 17.46%               |
| Pick Coke Can | 9    | 105             | 225          | 46.67%               |
| Move Near     | 8    | 302             | 480          | 62.92%               |

----

```bash
bash run_evaluations_variant_agg_drawer.sh
bash run_evaluations_variant_agg_move_near.sh
bash run_evaluations_variant_agg_pick_coke_can.sh
```

## Reproduce Training Results

To reproduce the training results, you can use the following steps:
1. Download the datasets
2. Add the modality configuration files
3. Fine-tune the model
4. Evaluate the model (same as above)

### 📦 1. Dataset Preparation

#### Dataset Downloads
Download LeRobot-compatible datasets directly from Hugging Face.

**1. Bridge Dataset**

```bash
huggingface-cli download \
    --repo-type dataset IPEC-COMMUNITY/bridge_orig_lerobot \
    --local-dir /tmp/bridge_orig_lerobot/
```

**2. Fractal Dataset**

```bash
huggingface-cli download \
    --repo-type dataset IPEC-COMMUNITY/fractal20220817_data_lerobot \
    --local-dir /tmp/fractal20220817_data_lerobot/
```

#### Modality Configuration

After downloading the datasets, you need to add the appropriate modality configuration files to make them compatible with GR00T N1.5. These configuration files define the observation and action space mappings.

**1. Bridge Dataset**

Copy the Bridge modality configuration to your dataset:
```bash
cp examples/SimplerEnv/bridge_modality.json /tmp/bridge_orig_lerobot/meta/modality.json
```

**2. Fractal Dataset**

Copy the Fractal modality configuration to your dataset:
```bash
cp examples/SimplerEnv/fractal_modality.json /tmp/fractal20220817_data_lerobot/meta/modality.json
```


### 🚀 Model Fine-tuning

#### Training Commands

The fine-tuning script supports multiple configurations. Below are examples for each simulation environment:

**1. Bridge Dataset**

```bash
python scripts/gr00t_finetune.py \
    --dataset-path /tmp/bridge_orig_lerobot/ \
    --data_config examples.SimplerEnv.custom_data_config:BridgeDataConfig \
    --num-gpus 8 \
    --batch-size 90 \
    --output-dir /tmp/bridge-checkpoints \
    --max-steps 60000
```

**2. Fractal Dataset**

```bash
python scripts/gr00t_finetune.py \
    --dataset-path /tmp/fractal20220817_data_lerobot/ \
    --data_config examples.SimplerEnv.custom_data_config:FractalDataConfig \
    --num-gpus 8 \
    --batch-size 128 \
    --output-dir /tmp/fractal-checkpoints/ \
    --max-steps 60000
```
