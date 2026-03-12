# RoboCasa GR1 Tabletop Tasks

Simulation benchmarks for GR-1 Tabletop Tasks developed for NVIDIA's GR00T N1 foundation model. Includes 24 tabletop manipulation tasks with 1,000 demonstrations each, enabling evaluation of generalist robotic policies in diverse household scenarios.

For more information, see the [official repository](https://github.com/robocasa/robocasa-gr1-tabletop-tasks) and [research paper](https://arxiv.org/abs/2503.14734).

---

# RoboCasa GR1 Tabletop Tasks evaluation benchmark result

| Task | Success rate |
| ---- | ------------ |
| `gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env` | 51.5% |
| `gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env` | 13.0% |
| `gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env` | 8.5% |
| `gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env` | 14.0% |
| `gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env` | 41.5% |
| `gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env` | 16.5% |
| `gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env` | 58.0% |
| `gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env` | 46.5% |
| `gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env` | 68.5% |
| `gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env` | 65.0% |
| `gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env` | 46.5% |
| `gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env` | 58.5% |
| `gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env` | 57.5% |
| `gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env` | 63.0% |
| `gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env` | 28.5% |
| `gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env` | 57.0% |
| `gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env` | 43.5% |
| `gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env` | 51.0% |
| `gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env` | 78.7% |
| `gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env` | 51.5% |
| `gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env` | 71.0% |
| `gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env` | 64.5% |
| `gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env` | 57.0% |
| `gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env` | 31.5% |
| **Average** | 47.6% |

# Evaluate checkpoint

First, setup the evaluation simulation environment. This only needs to run once for each simulation benchmark. After it's done, we only need to launch server and client.

```bash
sudo apt update
sudo apt install libegl1-mesa-dev libglu1-mesa
bash gr00t/eval/sim/robocasa-gr1-tabletop-tasks/setup_RoboCasaGR1TabletopTasks.sh
```

Then, run client server evaluation under the project root directory in separate terminals:

**Terminal 1 - Server:**
```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --embodiment-tag GR1 \
    --use-sim-policy-wrapper
```

**Terminal 2 - Client:**
```bash
gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --max_episode_steps=720 \
    --env_name gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env \
    --n_action_steps 8 \
    --n_envs 5
```
