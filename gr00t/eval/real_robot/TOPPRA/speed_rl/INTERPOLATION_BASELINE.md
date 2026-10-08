# SpeedTuning 30 Hz 插值 baseline

该 backend 用来和 TOPPRA-SpeedRL 对比。两者共享 VLA `last_block_pre_norm` feature、Rainbow
实现、人工 episode 标签和训练状态机，但速度动作和执行语义彼此独立，checkpoint、replay、标定
目录也不能共用。

## Receding-horizon 语义

默认参数：

```text
policy chunk horizon H = 40
k_skip = 10
action frequency = 30 Hz
v = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
```

每个 speed decision 都请求一个新的 40 点 GR00T chunk，只执行 10 个 target，然后无条件丢弃
当前 chunk 的其余部分并使用最新 observation 发起下一次同步推理。第 `j` 个 target 使用源 action
index `j * v`，其中 `j=0..9`。例如 `v=4` 使用：

```text
0, 4, 8, 12, 16, 20, 24, 28, 32, 36
```

启动时验证：

```text
(k_skip - 1) * v_max <= H - 1
```

因此不会读取第 40 点之外的 action。执行窗口恒为 `10/30 = 0.333 s`，而不是执行完整的
`ceil(H/v)` 个重采样 target。严格同步语义下，窗口结束后如果新推理尚未返回，机器人保持最后
target；代码会报告 inference hold，而不会偷偷继续执行旧 chunk。

## 增量 TCP action 的 SE(3) 插值

GR00T action 是：

```text
[dx, dy, dz, droll, dpitch, dyaw]
```

欧拉角不能直接线性插值。实现先按 rollout 的左乘约定累积增量：

```python
R_next = R.from_euler("xyz", delta_euler) * R_previous
```

然后：

1. 将 40 个增量累计成相对 SE(3) pose knots；
2. 将 SpeedTuning 的 action index `0,v,2v,...` 映射成相对路径 phase `1,1+v,1+2v,...`；
3. 平移线性插值，旋转沿相邻 SO(3) 最短弧插值；
4. 得到 10 个绝对双臂 TCP target；
5. pinch 等辅助 action 使用相同 phase 插值。

绝对 target 只在 30 Hz 切换，唯一 Servo owner 仍以 250 Hz 重复发布当前 target，不改变厂商
watchdog 和 tracking 架构。

## Rainbow 与奖励

baseline action 数为 7，和 TOPPRA 默认的 4 档 action head 不兼容。默认 hidden dimension 为
256，可通过 `RAINBOW_HIDDEN_DIM` 修改。每个 transition 的奖励为：

```python
reward = 0.01 * executed_30hz_steps * v**2
if terminal and safe_success:
    reward += 100.0
```

完整 decision 的 `executed_30hz_steps=10`。failure、abort 或速度违例没有 terminal bonus，但保留
已经执行运动对应的速度奖励；控制故障 episode 不进入 replay。

## 启动

在 `rollout.local.env` 中设置：

```text
ROLLOUT_MODE=speed-rl-baseline
INFERENCE_MODE=sync
TTS_SAMPLES=1
SPEED_RL_PHASE=calibration
POLICY_CHUNK_HORIZON=40
BASELINE_K_SKIP=10
BASELINE_ACTION_FREQUENCY=30
BASELINE_SPEED_MIN=1.0
BASELINE_SPEED_MAX=4.0
BASELINE_SPEED_STEP=0.5
ONLINE_EPISODES=100
DRY_RUN=1
LEFT_TWIST_THRESHOLDS=<六个硬件认证值>
RIGHT_TWIST_THRESHOLDS=<六个硬件认证值>
```

运行：

```bash
bash gr00t/eval/real_robot/TOPPRA/rollout/launch.sh rollout.local.env
```

默认持久化目录：

```text
logs/kion_speed_rl_baseline/state/
logs/kion_speed_rl_baseline/episodes/
```

7 档速度分别需要两个有效标定 episode 和逐档人工批准，随后独立执行默认 100 个 online episode
及 5 个 greedy 验收 episode。

## 限制与安全

- baseline 仅支持同步、单候选推理；异步预取会改变严格的 decision-boundary observation 语义。
- 0.333 s 执行窗口远短于秒级云端推理时，会出现明显 hold；这属于系统延迟限制。
- baseline 不运行 TOPPRA，不提供连续时间速度/加速度约束。
- `v` 是 nominal action-index 增量，不是 TOPPRA Cartesian constraint scale。
- TOPPRA 四档标定不能替代 baseline 七档标定。
- 实测 twist 100 帧 violation monitor 只标注 episode，不替代急停、watchdog 或碰撞保护。

## 自动测试

```bash
.venv/bin/python -m pytest tests/gr00t/eval/test_speed_rl.py -v
```

测试覆盖 `k_skip` 边界、`v=4` 的最大 action index、SE(3) 等价性、旋转 wrap、30/250 Hz hold、
terminal `+100`、backend checkpoint 隔离和控制循环预算。
