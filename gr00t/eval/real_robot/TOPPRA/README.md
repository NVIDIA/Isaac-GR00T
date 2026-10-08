# 双臂 GR00T TOPPRA Agent

`eval_toppra_bimanual.py` 在原 GR00T rollout 的 `act(obs, task)` 接口内完成：

1. 异步调用 GR00T server，获得双臂 action chunk。
2. 将 30 Hz delta TCP waypoints 积分为双臂绝对 TCP 路径。
3. 使用三次样条和 TOPPRA，在双臂笛卡尔速度、加速度约束下进行同步时间参数化。
4. 按 `control_frequency`（默认 250 Hz）预采样轨迹并写入 buffer。
5. 外部每调用一次 `act()`，从 buffer 中取出一个绝对 TCP 目标。

脚本只生成动作，不发布 ROS 消息，也不直接读取机器人 SDK。

Kion双臂ROS订阅、250 Hz `DualPose`发布及目标/实测TCP跟踪日志见
[`kion_client/README.md`](kion_client/README.md)。

新的统一实机入口、episode 状态机、夹爪执行以及 ObservationGUILite 人工标注见
[`rollout/README.md`](rollout/README.md)。同步、异步和 Speed-RL 均通过该入口启动。

仍需实机确认的速度阈值、硬件约束和验收门槛记录在
[`HARDWARE_TBD.md`](HARDWARE_TBD.md)；其中待确认数值不得作为实机安全上限。

## 创建 Agent

```python
from gr00t.eval.real_robot.TOPPRA.eval_toppra_bimanual import (
    CartesianLimits,
    GR00TAgent,
)

limits = CartesianLimits(
    max_linear_velocity=(1.0, 1.0, 1.0),       # m/s
    max_angular_velocity=(3.0, 3.0, 3.0),      # rad/s
    max_linear_acceleration=(5.0, 5.0, 5.0),   # m/s^2，按实机修改
    max_angular_acceleration=(15.0, 15.0, 15.0),  # rad/s^2，按实机修改
    safety_margin=0.9,
)

agent = GR00TAgent(
    host="127.0.0.1",
    port=5555,
    fps=30.0,                    # GR00T waypoint 频率
    control_frequency=250.0,     # act buffer 采样频率
    inference_mode="async",      # "sync" 或 "async"
    left_limits=limits,
    right_limits=limits,
    tts_samples=1,               # 多候选 TTS 时改为 server 返回的候选数
)
```

`GR00TAgent` 保留原实机适配器的构造参数和顺序；新增参数均为 keyword-only。未传
`left_limits/right_limits` 时，默认使用 1 m/s、3 rad/s、5 m/s²、15 rad/s²（再乘
`safety_margin=0.9`），默认 `inference_mode="async"`。实机应按机器人能力显式传入约束。

使用结束后调用：

```python
agent.teardown()
```

依赖为 `toppra==0.6.3`，已写入项目 `pyproject.toml` 和 `uv.lock`。

## Observation 输入

每次 `act()` 至少需要当前双臂绝对 TCP pose：

```python
observation = {
    "observation.state.left_tcp": left_pose,    # (7,)
    "observation.state.right_tcp": right_pose,  # (7,)
    # 相机和其他训练时使用的 observation 字段
}
```

pose 格式为基坐标系下：

```text
[x, y, z, qw, qx, qy, qz]
```

以后加入实测 TCP twist 时，配置：

```python
agent = GR00TAgent(
    ...,
    left_velocity_state_key="observation.state.left_tcp_twist",
    right_velocity_state_key="observation.state.right_tcp_twist",
    velocity_filter=1.0,
)
```

twist 必须是基坐标系下的 spatial twist：

```text
[vx, vy, vz, wx, wy, wz]
```

默认 `include_velocity_in_policy_observation=False`，twist 只供本地 TOPPRA 使用，不会发送
给未使用速度模态训练的 GR00T server。当前没有实测 twist 时，异步模式暂时使用 active
trajectory 在 observation 采集时对应的命令 twist；同步模式使用零起始速度。如果该近似
命令 twist 超出新路径的起点可控集，脚本会把它缩放到可控上界后继续 TOPPRA，避免因不可靠
的速度估计直接解算失败。实测 twist 不会被静默缩放。

## `act()` 输出

```python
action = agent.act(observation, task)

{
    "action.left_tcp": np.ndarray,     # (7,)
    "action.right_tcp": np.ndarray,    # (7,)
    "action.left_pinch": np.ndarray,   # (1,)
    "action.right_pinch": np.ndarray,  # (1,)
}
```

`action.left_tcp` 和 `action.right_tcp` 的语义为：

```text
基坐标系绝对目标位姿 [x, y, z, qw, qx, qy, qz]
```

它不是 delta pose、Euler angle 或 rotation vector，底层不能再次累加。ROS
`geometry_msgs/Pose.orientation` 使用 `xyzw`，发布前需要把本脚本的 `wxyz` 调整为 `xyzw`。
pinch 保持模型 action 的连续数值，本脚本不负责二值化。

GR00T server 的输入 action chunk 仍为：

```text
action.left_delta_tcp   [dx, dy, dz, dEulerX, dEulerY, dEulerZ]
action.right_delta_tcp  [dx, dy, dz, dEulerX, dEulerY, dEulerZ]
```

Euler 使用弧度和 `xyz` 顺序。`parcel_4f_v5.7` 数据验证的积分规则为
`p_next = p + dp`、`R_next = dR * R`。

## 底层 250 Hz 调用

`act()` 不使用 wall-clock 时间选择轨迹点。每调用一次固定消费一个 buffer 样本，因此 250 Hz
必须由外部 ROS 控制循环保证：

```python
import time

period = 1.0 / 250.0
next_tick = time.monotonic()

while running:
    observation = read_current_robot_observation()
    action = agent.act(observation, task)
    publish_absolute_tcp_targets(action)

    next_tick += period
    remaining = next_tick - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)
    else:
        next_tick = time.monotonic()  # 超时后不连续调用 act() 追赶旧周期
```

active trajectory 执行期间，连续传入相同 observation 也会依次得到相邻轨迹样本。调用快于
250 Hz 会让轨迹执行过快；停止或慢于 250 Hz 调用会让轨迹暂停或变慢。实机建议由 ROS
timer 或实时控制线程保证周期，Python `sleep` 示例只用于说明接口。

## 同步与异步模式

- `sync`：首次规划从 observation 的实测 pose 开始；已有轨迹结束后，推理期间持续输出
  上一轨迹的最后命令 pose，并以同一 pose 作为下一轨迹的零速起点，避免命令在 chunk 边界
  跳回存在跟踪滞后的实测 pose。规划完成后完整执行新轨迹，首末速度为零。该模式不检查
  实测 pose 是否已经收敛，假设底层控制器会在推理期间持续跟踪并到达最后命令 pose。
- `async`：旧轨迹执行期间提前请求下一个 chunk。策略返回后，在旧轨迹中预约未来的命令
  pose/twist 作为 handoff，按照 observation 到 handoff 之间完成的 policy waypoints 裁掉
  过时 delta，再从该 handoff 状态运行 TTS 和 TOPPRA。新轨迹提前完成后等待预约控制周期，
  通过命令 pose/twist 连续性检查后原子切换，不再按照规划期间消费的高频样本数裁剪新轨迹。

异步触发默认使用固定预算：

```text
策略延迟预算       120 ms
调度余量            50 ms
handoff余量         50 ms
合计提前量          220 ms
open-loop horizon     8 policy waypoints
```

当前轨迹剩余时间不大于 220 ms，或者预计切换时 observation 已经开环推进 8 个 policy
waypoints 时，启动下一次请求。同一时刻最多一个请求。`open_loop_horizon` 不表示 8 个
250 Hz 控制样本。

异步模式的 `async_terminal_path_velocity=1.0` 是终点路径速度上限。TOPPRA 在 `[0, 1]` 中
选择可行且最快的终点速度；末段曲率较大时允许自动降低，不再强制以路径速度 1 结束。

可通过诊断确认起始速度是否被缩放：

```python
status = agent.diagnostics()
print(status["initial_twist_scale"])  # 1.0 表示未缩放，小于 1.0 表示使用可控集上界
print(status["terminal_path_speed"])  # TOPPRA 实际选择的终点路径速度
print(status["latest_error"])
```

默认 `allow_timing_fallback=False`。TOPPRA 仍失败时不会退回原始 30 Hz chunk：异步模式继续
消费旧 buffer，旧 buffer 耗尽后输出 hold pose，并在后续周期重新请求规划。

## 运行日志

脚本使用 Python `logging`。WARNING/ERROR 默认可见；要查看推理、TOPPRA 和 handoff 的完整
状态，在程序入口配置 INFO 级别：

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
)
```

关键标签包括：`inference_request`、`policy_response`、`handoff_reserved`、`tts_selection`、
`plan_success`、`trajectory_ready`、`handoff_activated`、`handoff_missed`、
`handoff_discontinuous`、`candidate_failed`、`planning_failed` 和 `buffer_underflow`。日志只在
状态变化时输出，不会在每个 250 Hz `act()` 周期打印。

## 当前缺少的能力

- 没有 ROS/SDK 状态订阅、状态时间戳和 pose/twist 同步检查，实时性取决于调用方提供的 obs。
- 新轨迹激活前只检查新旧命令轨迹的 pose/twist 连续性；实测状态与命令状态之间的跟踪
  误差不作为正常 handoff 的硬拒绝条件。
- 没有 IK 可达性、关节限位、关节速度/力矩、奇异点、自碰撞和双臂碰撞检查。
- 旋转路径使用 continuous rotation vector；物理角速度/角加速度约束是小角度近似。实机应
  保留 `safety_margin`，并对采样轨迹进行后验越界检查。
- `act()` 只返回 pose 和 pinch；轨迹内部虽计算速度、加速度，但当前公共 action 接口不输出。
- TOPPRA 在速度和加速度约束下求时间最优时间律，不约束 jerk。位姿和速度连续，但加速度可
  在约束上下限之间快速切换；三次插值也会原样穿过模型生成的抖动 waypoints，而不是做平滑
  拟合。
- Python 线程和 `act()` 不是硬实时控制器；底层仍需负责 watchdog、通信超时和安全停机。
