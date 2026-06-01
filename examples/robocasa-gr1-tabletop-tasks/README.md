# RoboCasa GR1 Tabletop Tasks

Simulation benchmarks for GR-1 Tabletop Tasks developed for NVIDIA's GR00T N1 foundation model. Includes 24 tabletop manipulation tasks with 1,000 demonstrations each, enabling evaluation of generalist robotic policies in diverse household scenarios.

For more information, see the [official repository](https://github.com/robocasa/robocasa-gr1-tabletop-tasks) and [research paper](https://arxiv.org/abs/2503.14734).

---

# RoboCasa GR1 Tabletop Tasks evaluation benchmark result

Checkpoint: finetuned N1.7 `ROBOCASA_GR1_TABLETOP` checkpoint

| Task | Success rate | Trials |
| ---- | ------------ | ------ |
| `gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env` | 70.0% | 20 |
| `gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env` | 70.0% | 20 |
| `gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env` | 35.0% | 20 |
| `gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env` | 45.0% | 20 |
| `gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env` | 40.0% | 20 |
| `gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env` | 65.0% | 20 |
| `gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env` | 10.0% | 20 |
| `gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env` | 30.0% | 20 |
| `gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env` | 40.0% | 20 |
| `gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env` | 45.0% | 20 |
| `gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env` | 25.0% | 20 |
| `gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env` | 40.0% | 20 |
| `gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env` | 40.0% | 20 |
| `gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env` | 40.0% | 20 |
| `gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env` | 18.2% | 22 |
| `gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env` | 50.0% | 20 |
| `gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env` | 35.0% | 20 |
| `gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env` | 40.0% | 20 |
| `gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env` | 75.0% | 20 |
| `gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env` | 60.0% | 20 |
| `gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env` | 50.0% | 20 |
| `gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env` | 45.0% | 20 |
| `gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env` | 55.0% | 20 |
| `gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env` | 45.0% | 20 |
| **Average** | 44.5% | - |

The average is the mean of per-task success rates. Trials are the per-task closed-loop episode counts.

# Finetune

To finetune from the N1.7 base model on the RoboCasa GR1 Tabletop Tasks dataset, convert the demonstrations to the GR00T LeRobot format and launch the shared finetune script:

```bash
NUM_GPUS=8 MAX_STEPS=60000 GLOBAL_BATCH_SIZE=512 SAVE_STEPS=2000 uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path <path-to-gr1-tabletop-lerobot-dataset> \
  --embodiment-tag ROBOCASA_GR1_TABLETOP \
  --output-dir /tmp/gr1_tabletop_finetune
```

The original N1.6 `gr1_unified` embodiment tag was retired with the N1.7 release. Use the N1.7 `ROBOCASA_GR1_TABLETOP` finetuning tag for this dataset.

`ROBOCASA_GR1_TABLETOP` uses an 8-step action target to match the closed-loop RoboCasa GR1 evaluation setting.

# Evaluate checkpoint

First, set up the evaluation simulation environment. This only needs to run once for each simulation benchmark. After it's done, we only need to launch server and client.

```bash
sudo apt update
sudo apt install libegl1-mesa-dev libglu1-mesa
bash gr00t/eval/sim/robocasa-gr1-tabletop-tasks/setup_RoboCasaGR1TabletopTasks.sh
```

Then, run client server evaluation under the project root directory in separate terminals:

**Terminal 1 - Server:**
```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path <path-to-finetuned-gr1-tabletop-checkpoint> \
    --embodiment-tag ROBOCASA_GR1_TABLETOP \
    --use-sim-policy-wrapper
```

**Terminal 2 - Client:**
```bash
gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env \
    --n-action-steps 8 \
    --n-envs 5
```
