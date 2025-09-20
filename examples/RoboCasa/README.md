# GR00T RoboCasa Tabletop Task Benchmarks

This directory contains fine-tuning and evaluation scripts for **GR00T N1.5** on the RoboCasa Tabletop benchmark suite.



## 🎯 Model Evaluation

<!-- /mnt/amlfs-02/shared/checkpoints/xiaoweij/0910/robocasa-checkpoints-60K/  https://wandb.ai/nv-gear/huggingface/runs/zhvckr9n -->
The finetuned model is uploaded to youliangtan/gr00t-n1.5-robocasa-tabletop-posttrain.

| Environment                                                                 | Success Rate |
|-----------------------------------------------------------------------------|--------------|
| gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env             | 0.38         |
| gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env       | 0.32         |
| gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env         | 0.60         |
| gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env         | 0.54         |
| gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env           | 0.38         |
| gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env             | 0.50         |
| gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env | 0.38 |
| gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env | 0.46 |
| gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env | 0.58 |
| gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env | 0.62 |
| gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env | 0.28 |
| gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env | 0.30 |
| gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env | 0.60 |
| gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env | 0.56 |
| gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env | 0.36 |
| gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env | 0.58 |
| gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env | 0.44 |
| gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env | 0.60 |
| gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env | 0.64 |
| gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env | 0.52 |
| gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env | 0.48 |
| gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env | 0.60 |
| gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env | 0.52 |
| gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env | 0.32 |
| **Average**                              | **0.48**            |
----

All the above tasks are evaluated at 50 rollouts each.

To evaluate, first start the inference server with our provided checkpoint:

```bash
python3 scripts/inference_service.py --server \
    --model_path youliangtan/gr00t-n1.5-robocasa-tabletop-posttrain \
    --data_config fourier_gr1_arms_waist
```

### Installation

Follow the [official RoboCasa installation guide](https://github.com/robocasa/robocasa-gr1-tabletop-tasks?tab=readme-ov-file#getting-started).

Then run the evaluation:
```bash
python3 scripts/simulation_service.py --client \
    --env_name <TASK_NAME> \
    --video_dir ./videos \
    --max_episode_steps 720 \
    --n_episodes 50
```

> Note, using `n_envs` larger than 1 may lower success rate due to [this issue](https://github.com/NVIDIA/Isaac-GR00T/pull/292).

----

## Reproduce Training Results

To reproduce the training results, you can use the following steps:
1. Download the datasets
2. Fine-tune the model
3. Evaluate the model (same as above)

### 📦 1. Dataset Preparation

Download LeRobot-compatible datasets directly from [Hugging Face](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim).
We are using "Humanoid robot tabletop manipulation - downsampled: 24k trajectories" for finetuning.

To download only the relevant folders, follow the instructions below. This will only download the _1000 dataset.

```bash
# Clone the repo without downloading files
git clone --filter=blob:none --no-checkout https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
cd PhysicalAI-Robotics-GR00T-X-Embodiment-Sim

# Configure sparse-checkout for just your folder
git sparse-checkout init --cone
git sparse-checkout set "**/*_1000/"
```

Modality config is already pre-configured for you in the dataset.

### 🚀 Model Fine-tuning

The fine-tuning script supports multiple configurations.

```bash
data_root=/tmp/robocasa_finetune_data

ALL_DATASET_PATHS=(
  "${data_root}/gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000"
)


python scripts/gr00t_finetune.py \
  --dataset-path "${ALL_DATASET_PATHS[@]}" \
  --num-gpus 8 --batch-size 48 --learning_rate 3e-5 \
  --output-dir /mnt/amlfs-02/shared/checkpoints/xiaoweij/0910/robocasa-checkpoints-60K/  \
  --data-config fourier_gr1_arms_waist --embodiment_tag gr1 --tune-visual \
  --max-steps 60000 --save-steps 5000  --gradient-accumulation-steps 4
```
