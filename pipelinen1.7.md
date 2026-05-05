# GR00T N1.7 Finetuning Pipeline — RoboCasa GR1 Tabletop

End-to-end记录：在单卡 RTX 4090 (48GB) 上用 RoboCasa GR1 数据集
(`gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000`)
微调 `nvidia/GR00T-N1.7-3B`。

---

## 0. 仓库与环境

| 项 | 值 |
|---|---|
| 仓库 | `/home/d024/Isaac-GR00T` |
| 分支/Tag | `n1.7-release` (commit `23ace64`) |
| Python | 3.10 |
| 包管理 | **uv** (项目本地 `.venv`，**不**用 conda) |
| 关键依赖 | torch 2.7.1+cu128, transformers 4.57.3, wandb 0.23.x |

激活环境：

```bash
cd /home/d024/Isaac-GR00T
source .venv/bin/activate            # 进入 N1.7 venv
# 退出: deactivate
```

> 同一终端里如果还显示 `(base)`，那是 conda 的 base 环境名残留，**实际 Python 会优先解析 venv** —— 用 `which python` 验证应指向 `.venv/bin/python`。

---

## 1. 资产清单

| 资产 | 路径 | 用途 |
|---|---|---|
| **GR00T-N1.7-3B 主模型** | `/home/d024/models/GR00T-N1.7-3B` | 微调起点 (`--base-model-path`) |
| **Cosmos-Reason2-2B VLM** | `/home/d024/models/Cosmos-Reason2-2B` | N1.7 backbone (gated 仓库) |
| **Cosmos 符号链接** | `/home/d024/models/nvidia/Cosmos-Reason2-2B` → 上一项 | 路径必须包含 `nvidia/Cosmos-Reason2`（代码中字符串校验） |
| **数据集** | `/home/d024/DiT4DiT/playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000` | LeRobot v2.0 parquet 格式，44-DoF GR1 |
| **微调输出** | `/home/d024/models/gr00t_n17_gr1_finetune/<exp_name>` | checkpoint / wandb_config / experiment_cfg |
| **RoboCasa/DiT4DiT 仿真仓库** | `/home/d024/DiT4DiT` | 复用 RoboCasa GR1 tabletop 评估 harness |
| **RoboCasa conda 环境** | `/home/d024/miniconda3/envs/robocasa` | 运行仿真评估 |

下载命令（已完成）：

```bash
huggingface-cli download nvidia/GR00T-N1.7-3B   --local-dir /home/d024/models/GR00T-N1.7-3B
huggingface-cli download nvidia/Cosmos-Reason2-2B --local-dir /home/d024/models/Cosmos-Reason2-2B
ln -sfn /home/d024/models/Cosmos-Reason2-2B /home/d024/models/nvidia/Cosmos-Reason2-2B
```

> 两个仓库都是 **gated**，必须先在 HuggingFace 网页上 accept license，再 `huggingface-cli login` 输入 token。

---

## 2. 关键代码改动

### 2.1 自定义 modality config

N1.7 不再原生支持 `EmbodimentTag.GR1`，所以注册 GR1 为 `NEW_EMBODIMENT`。

文件：[examples/robocasa_gr1/gr1_config.py](examples/robocasa_gr1/gr1_config.py)

- DiT4DiT-style 5 个关节组 (state + action)：`left_arm, right_arm, left_hand, right_hand, waist` (=29D)
- state 使用 sin/cos encoding：29D raw joint state -> 58D encoded state
- video key: `ego_view`
- language key: `annotation.human.coarse_action`
- action 预测 horizon: 16 步，所有组用 `ABSOLUTE / NON_EEF` 表示

最终配置核心：

```python
GR1_STATE_KEYS = [
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "waist",
]

GR1_ACTION_KEYS = GR1_STATE_KEYS

"state": ModalityConfig(
    delta_indices=[0],
    modality_keys=GR1_STATE_KEYS,
    sin_cos_embedding_keys=GR1_STATE_KEYS,
)

"action": ModalityConfig(
    delta_indices=list(range(0, 16)),
    modality_keys=GR1_ACTION_KEYS,
    action_configs=[ActionConfig(
        rep=ActionRepresentation.ABSOLUTE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
    )] * len(GR1_ACTION_KEYS),
)
```

这套配置对齐 DiT4DiT 的 `FourierGr1ArmsWaistDataConfig`：

```text
state/action order:
left_arm, right_arm, left_hand, right_hand, waist
```

### 2.2 让 backbone 走本地路径

N1.7 checkpoint 的 `config.json` 里硬编码了 `"model_name": "nvidia/Cosmos-Reason2-2B"`，导致每次启动还会去 HF 拉 gated 仓库。两个修复：

**a. patch checkpoint 的 model_name**：
```bash
# 改成本地符号链接路径（路径里仍含 "nvidia/Cosmos-Reason2"）
python -c "import json; p='/home/d024/models/GR00T-N1.7-3B/config.json'; \
  c=json.load(open(p)); c['model_name']='/home/d024/models/nvidia/Cosmos-Reason2-2B'; \
  json.dump(c, open(p,'w'), indent=2)"
```

**b. patch `launch_finetune.py`**（同目的，覆盖 default config）：

```python
# gr00t/experiment/launch_finetune.py
config.model.model_name = "/home/d024/models/nvidia/Cosmos-Reason2-2B"
config.model.apply_sincos_state_encoding = True
config.model.use_relative_action = False
```

> 检查点已备份在 `config.json.bak`。

### 2.3 为什么需要 `nvidia/Cosmos-Reason2` 字串

在 [gr00t/model/gr00t_n1d7/gr00t_n1d7.py:477](gr00t/model/gr00t_n1d7/gr00t_n1d7.py)：

```python
if "nvidia/Cosmos-Reason2" in config.model_name or "Qwen/Qwen3-VL" in config.model_name:
    return Qwen3Backbone
raise ValueError(f"Unsupported model name: {config.model_name}")
```

所以本地路径必须包含子串 `nvidia/Cosmos-Reason2`（用符号链接最简单）。

---

## 3. 启动微调

### 烟雾测试（推荐先跑，~几分钟）

```bash
cd /home/d024/Isaac-GR00T && source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
    --base-model-path /home/d024/models/GR00T-N1.7-3B \
    --dataset-path /home/d024/DiT4DiT/playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/robocasa_gr1/gr1_config.py \
    --num-gpus 1 \
    --output-dir /home/d024/models/gr00t_n17_gr1_finetune \
    --experiment-name gr1_pnp_cup_to_drawer_dit4dit_state_smoke \
    --max-steps 20 \
    --global-batch-size 8 \
    --dataloader-num-workers 4 \
    --save-steps 20 \
    --use-wandb \
    --wandb-project finetune-gr00t-n1d7-gr1
```

### 正式训练（已跑通）

```bash
cd /home/d024/Isaac-GR00T
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
    --base-model-path /home/d024/models/GR00T-N1.7-3B \
    --dataset-path /home/d024/DiT4DiT/playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/robocasa_gr1/gr1_config.py \
    --num-gpus 1 \
    --output-dir /home/d024/models/gr00t_n17_gr1_finetune \
    --experiment-name gr1_pnp_cup_to_drawer_dit4dit_state \
    --max-steps 30000 \
    --global-batch-size 8 \
    --dataloader-num-workers 4 \
    --save-steps 5000 \
    --save-total-limit 3 \
    --use-wandb \
    --wandb-project finetune-gr00t-n1d7-gr1
```

OOM 时降 `--global-batch-size` 到 4–8 并把 `--gradient-accumulation-steps` 设为 2/4 维持等效 batch。

本次训练结果：

```text
experiment: /home/d024/models/gr00t_n17_gr1_finetune/gr1_pnp_cup_to_drawer_dit4dit_state
max_steps: 30000
runtime: 2:24:57
train_loss: 0.09864946630746126
last logged loss at step 30000: 0.0309
```

训练完成后，最终模型会保存到实验目录顶层；同时保留最近的 checkpoint：

```text
/home/d024/models/gr00t_n17_gr1_finetune/gr1_pnp_cup_to_drawer_dit4dit_state/
├── config.json
├── model-00001-of-00003.safetensors
├── model-00002-of-00003.safetensors
├── model-00003-of-00003.safetensors
├── model.safetensors.index.json
├── processor/
├── experiment_cfg/
├── checkpoint-20000/
├── checkpoint-25000/
└── checkpoint-30000/
```

推理时优先用顶层目录；如果要指定某个中间版本，则把 `--model-path` 指到 `checkpoint-xxxxx`。

### Co-training 全部 24 个 GR1 任务

DiT4DiT 用 `data_mix: fourier_gr1_unified_1000` 训练 GR1 RoboCasa，这个 mixture 在 `/home/d024/DiT4DiT/DiT4DiT/dataloader/gr00t_lerobot/mixtures.py` 中定义，包含 24 个数据集，每个 sampling weight 都是 `1.0`，robot config 都是 `fourier_gr1_arms_waist`。

DiT4DiT 对应训练配置核心：

```text
data_mix: fourier_gr1_unified_1000
max_state_dim: 64
max_action_dim: 32
state/action keys: left_arm, right_arm, left_hand, right_hand, waist
max_train_steps: 200000
```

GR00T N1.7 的 `launch_finetune.py` 已扩展支持 `--dataset-paths`，可以一次传入多个 LeRobot 数据集路径。下面命令会自动收集本地 24 个 `gr1_unified.*GR1ArmsAndWaistFourierHands_1000` 数据集并 co-train：

```bash
cd /home/d024/Isaac-GR00T
source .venv/bin/activate

DATA_ROOT=/home/d024/DiT4DiT/playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
mapfile -t DATASET_PATHS < <(
  find "$DATA_ROOT" -maxdepth 1 -type d \
    -name 'gr1_unified.*GR1ArmsAndWaistFourierHands_1000' | sort
)

printf "Using %d datasets\n" "${#DATASET_PATHS[@]}"
printf '%s\n' "${DATASET_PATHS[@]}"

CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
    --base-model-path /home/d024/models/GR00T-N1.7-3B \
    --dataset-paths "${DATASET_PATHS[@]}" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/robocasa_gr1/gr1_config.py \
    --num-gpus 1 \
    --output-dir /home/d024/models/gr00t_n17_gr1_finetune \
    --experiment-name gr1_unified_24tasks_dit4dit_state \
    --max-steps 200000 \
    --global-batch-size 8 \
    --dataloader-num-workers 4 \
    --save-steps 25000 \
    --save-total-limit 4 \
    --use-wandb \
    --wandb-project finetune-gr00t-n1d7-gr1
```

说明：

- `--dataset-paths` 覆盖单任务的 `--dataset-path`。
- 当前 24 个数据集都是 1000 demos 级别，长度接近，因此混合采样近似等权，和 DiT4DiT 的 `1.0` weights 对齐。
- 单卡 4090 上 30k steps 约 2.4 小时；200k steps 约十几小时量级，建议先跑 50k/100k 做趋势检查，再完整拉到 200k。
- 推理时把 `--model-path` 指向 `/home/d024/models/gr00t_n17_gr1_finetune/gr1_unified_24tasks_dit4dit_state` 或具体 `checkpoint-xxxxx`。

---

## 4. 训练哪些模块（默认）

| 模块 | 训练? | 控制开关 (默认) |
|---|---|---|
| Cosmos-Reason2 LLM 主干 | ❌ | `--tune-llm False` |
| Cosmos-Reason2 视觉编码器 (ViT) | ❌ | `--tune-visual False` |
| 多模态 projector / VLLN | ✅ | `--tune-projector True` |
| Action head (DiT diffusion) | ✅ | `--tune-diffusion-model True` |

策略：**冻结 VLM，只训 action head + projector**。这是 GR00T 官方推荐的轻量后训练方案；既保留预训练语义/视觉能力，也让 4090 单卡能跑得动。

---

## 5. wandb

- 已 `wandb login`，账号 `ju-dong6276 (technical-university-of-munich)`
- project: `finetune-gr00t-n1d7-gr1`
- run name: `--experiment-name` 的值
- 关闭：去掉 `--use-wandb`
- 离线模式：`wandb offline` 或环境变量 `WANDB_MODE=offline`

---

## 6. 输出目录结构

```
/home/d024/models/gr00t_n17_gr1_finetune/<experiment-name>/
├── experiment_cfg/          # YAML config + 数据 statistics + modality
├── checkpoint-1000/         # 每 --save-steps 一份
├── checkpoint-2000/
├── ...
└── wandb_config.json
```

---

## 7. RoboCasa/DiT4DiT 评估环境

训练在 Isaac-GR00T 的 `.venv` 中完成；RoboCasa 仿真评估在 DiT4DiT 的 `robocasa` conda 环境中完成。

### 7.1 激活 RoboCasa 环境

```bash
cd /home/d024/DiT4DiT
deactivate 2>/dev/null; true
source /home/d024/miniconda3/etc/profile.d/conda.sh
conda activate robocasa

export PYTHONPATH=/home/d024/DiT4DiT:/home/d024/Isaac-GR00T:$PYTHONPATH
```

### 7.2 已修复的环境依赖

RoboCasa fork 要求 NumPy 版本只能是 `1.23.2/1.23.3/1.23.5/1.26.4`。当前环境已调整为：

```bash
/home/d024/miniconda3/envs/robocasa/bin/python -m pip install "numpy==1.26.4"
```

DiT4DiT 的评估脚本顶层 import 需要 `websockets`：

```bash
/home/d024/miniconda3/envs/robocasa/bin/python -m pip install "websockets>=13,<16"
```

验证：

```bash
PYTHONPATH=/home/d024/DiT4DiT:/home/d024/Isaac-GR00T:$PYTHONPATH \
/home/d024/miniconda3/envs/robocasa/bin/python - <<'PY'
import numpy
print("numpy", numpy.__version__)
from examples.Robocasa_tabletop.eval_files.simulation_env import run_evaluation
print("simulation_env import ok")
PY
```

### 7.3 RoboCasa 怎么被调用

评估脚本：[scripts/eval/eval_robocasa_gr1_gr00t.py](scripts/eval/eval_robocasa_gr1_gr00t.py)

它复用 DiT4DiT 的 RoboCasa rollout harness：

```python
from examples.Robocasa_tabletop.eval_files.simulation_env import run_evaluation
```

数据流：

```text
RoboCasa env.reset()
  -> obs: video/state/language
  -> Gr00tRoboCasaPolicy.step(obs)
  -> GR00T PolicyClient 通过 ZeroMQ 请求 GR00T server
  -> 返回 action chunk
  -> RoboCasa env.step(actions)
  -> VideoRecordingWrapper 保存视频
```

RoboCasa `GR1ArmsAndWaist` env 提供的 state/action 只有 5 组：

```text
left_arm, right_arm, left_hand, right_hand, waist
```

language 由 env observation 中的 `annotation.human.coarse_action` 提供；adapter 不修改 language，只转发给 GR00T sim policy wrapper。

---

## 8. 推理评估

### 8.1 终端 A：启动 GR00T policy server

```bash
cd /home/d024/Isaac-GR00T
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python gr00t/eval/run_gr00t_server.py \
    --model-path /home/d024/models/gr00t_n17_gr1_finetune/gr1_pnp_cup_to_drawer_dit4dit_state \
    --embodiment-tag NEW_EMBODIMENT \
    --device cuda:0 \
    --host 0.0.0.0 \
    --port 6398 \
    --no-strict \
    --use-sim-policy-wrapper
```

等待出现 server ready 后再开终端 B。

参数说明：

- `--use-sim-policy-wrapper`：让 server 接受仿真环境常用的 flat key，例如 `video.ego_view`、`state.left_arm`、`annotation.human.coarse_action`，并转换成 GR00T 内部 nested observation。
- `--no-strict`：关闭严格 shape/key 校验，外接仿真环境时更稳；接口完全对齐后可以尝试去掉。

### 8.2 终端 B：启动 RoboCasa 评估

```bash
cd /home/d024/DiT4DiT
deactivate 2>/dev/null; true
source /home/d024/miniconda3/etc/profile.d/conda.sh
conda activate robocasa

export PYTHONPATH=/home/d024/DiT4DiT:/home/d024/Isaac-GR00T:$PYTHONPATH

/home/d024/miniconda3/envs/robocasa/bin/python \
    /home/d024/Isaac-GR00T/scripts/eval/eval_robocasa_gr1_gr00t.py \
    --args.host 127.0.0.1 \
    --args.port 6398 \
    --args.env-name "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env" \
    --args.n-episodes 5 \
    --args.n-envs 1 \
    --args.max-episode-steps 720 \
    --args.n-action-steps 12
```

默认视频输出：

```text
/home/d024/models/gr00t_n17_gr1_finetune/gr1_pnp_cup_to_drawer_dit4dit_state/robocasa_eval_videos
```

### 8.3 检查评估状态和视频

检查进程/端口：

```bash
ps -ef | rg "run_gr00t_server|eval_robocasa_gr1_gr00t|simulation_env|6398"
ss -ltnp | rg ":6398"
```

列出最新视频：

```bash
find /home/d024/models/gr00t_n17_gr1_finetune/gr1_pnp_cup_to_drawer_dit4dit_state \
    -type f -name "*.mp4" \
    -printf "%TY-%Tm-%Td %TH:%TM:%TS %s %p\n" | sort | tail -20
```

检查视频元数据：

```bash
ffprobe -v error \
    -show_entries format=duration,size \
    -show_entries stream=width,height,nb_frames,r_frame_rate \
    -of default=noprint_wrappers=1 \
    /path/to/video.mp4
```

文件名中的 `success1` / `success0` 是 episode 成功标记。

本次 5 episode 评估结果：

```text
success: 1 / 5 = 20%
successful video:
/home/d024/models/gr00t_n17_gr1_finetune/gr1_pnp_cup_to_drawer_dit4dit_state/robocasa_eval_videos/48cf6960-b24a-463f-8a9c-78a3b8c56558_success1.mp4
video format: 1280x800, 10 fps, 360 frames, 36s
```

---

## 9. 常见坑

| 错误 | 根因 | 解决 |
|---|---|---|
| `GatedRepoError ... Cosmos-Reason2-2B` | 模型名仍是 HF id，去拉 gated 仓库 | 走本地路径（见 §2.2），并 `huggingface-cli login` |
| `Unsupported model name: /xxx/Cosmos-Reason2-2B` | `gr00t_n1d7.py` 字符串检查要求 `nvidia/Cosmos-Reason2` 子串 | 用 `…/nvidia/Cosmos-Reason2-2B` 符号链接 |
| `ModuleNotFoundError: lerobot` | N1.7 venv 不含 lerobot | 在 robocasa conda env 里跑可视化；或 `pip install "lerobot==0.1.0"`（v2.0 兼容版） |
| `AssertionError: numpy version must be...` | RoboCasa fork 不支持 NumPy 2.x | 在 robocasa env 安装 `numpy==1.26.4` |
| `ModuleNotFoundError: websockets` | DiT4DiT 评估脚本顶层 import websocket client | 在 robocasa env 安装 `websockets>=13,<16` |
| `image_crop_size will be deprecated` | 警告不影响运行 | 忽略 |
| OOM | batch 过大或 backbone 全调 | 降 `--global-batch-size`，加 `--gradient-accumulation-steps` |

---

## 10. 数据可视化（可选）

不在 N1.7 venv 里，建议在 robocasa env 用 lerobot CLI（v2.0 数据需要 `lerobot==0.1.0`）。
或者直接复用项目里现成的 [replay_lerobot_episode.py](../robocasa-gr1-tabletop-tasks/replay_lerobot_episode.py) —— 在 robosuite 里物理回放 episode。



---
cd /home/d024/Isaac-GR00T
source .venv/bin/activate

DATA_ROOT=/home/d024/DiT4DiT/playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim

mapfile -t DATASET_PATHS < <(
  find "$DATA_ROOT" -maxdepth 1 -type d \
    -name 'gr1_unified.*GR1ArmsAndWaistFourierHands_1000' | sort
)

printf "Using %d datasets\n" "${#DATASET_PATHS[@]}"
printf '%s\n' "${DATASET_PATHS[@]}"

CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
  --base-model-path /home/d024/models/GR00T-N1.7-3B \
  --dataset-paths "${DATASET_PATHS[@]}" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/robocasa_gr1/gr1_config.py \
  --num-gpus 1 \
  --output-dir /home/d024/models/gr00t_n17_gr1_finetune \
  --experiment-name gr1_unified_24tasks_dit4dit_state \
  --max-steps 200000 \
  --global-batch-size 8 \
  --dataloader-num-workers 4 \
  --save-steps 25000 \
  --save-total-limit 4 \
  --use-wandb \
  --wandb-project finetune-gr00t-n1d7-gr1

---


---
cd /home/d024/DiT4DiT
source /home/d024/miniconda3/etc/profile.d/conda.sh
conda activate robocasa

export PYTHONPATH=/home/d024/DiT4DiT:/home/d024/Isaac-GR00T:$PYTHONPATH

/home/d024/miniconda3/envs/robocasa/bin/python \
  /home/d024/Isaac-GR00T/scripts/eval/eval_robocasa_gr1_gr00t.py \
  --args.host 127.0.0.1 \
  --args.port 6398 \
  --args.env-name "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env" \
  --args.n-episodes 5 \
  --args.n-envs 1 \
  --args.max-episode-steps 720 \
  --args.n-action-steps 12

---