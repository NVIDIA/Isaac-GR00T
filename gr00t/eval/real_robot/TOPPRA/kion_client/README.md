# Kion双臂ROS Client

该包在机器人本地运行，连接GR00T server，复用
`eval_toppra_bimanual.py`完成同步或异步TOPPRA规划，并按固定频率向以下topic发布双臂绝对
TCP目标：

```text
/zj_humanoid/upperlimb/servol/dual_arm
upperlimb/DualPose
```

每个控制周期只发布一条`DualPose`，左右臂目标来自同一次`agent.act()`。
`agent.act()`输出的是TOPPRA轨迹上的绝对目标`action.left_tcp/action.right_tcp`，格式均为
`[x,y,z,qw,qx,qy,qz]`，不是模型原始的delta action。该client不发布速度或加速度。

## 输入

默认订阅：

```text
/zj_humanoid/sensor/left_wrist/image_raw/compressed
/zj_humanoid/sensor/right_wrist/image_raw/compressed
/zj_humanoid/sensor/realsense_head/color/image_raw/compressed
/zj_humanoid/upperlimb/tcp_pose/left_arm
/zj_humanoid/upperlimb/tcp_pose/right_arm
/zj_humanoid/upperlimb/tcp_speed/dual_arm
/zj_humanoid/hand/finger_pressures/left
/zj_humanoid/hand/finger_pressures/right
/wrist_force_control/left_arm_compensated_force
/wrist_force_control/right_arm_compensated_force
```

不订阅chest相机。TCP pose为：

```text
[x, y, z, qw, qx, qy, qz]
```

TCP speed为：

```text
[vx, vy, vz, wx, wy, wz]
```

twist默认不发送给GR00T server。当前异步handoff仍使用旧命令轨迹的预约pose/twist来保证
target连续，实测twist用于记录和后续跟踪误差分析。

同步模式和异步模式都由Agent内部producer线程调用server和运行TOPPRA，所以250 Hz ROS循环
不会等待网络推理。同步模式在新轨迹准备期间重复发布hold target，准备完成后从头逐点执行；
异步模式在推理期间继续消费旧轨迹，到预约handoff点后切换。每次控制循环只调用一次
`agent.act()`并消费一个预采样点，不按墙钟时间跳过buffer中的轨迹点。

## 云端到实机Rollout教程

以下步骤假设云端已经有可用的GR00T模型和GPU，本地笔记本连接机器人ROS网络。云端只运行
模型server；相机和机器人状态采集、TOPPRA规划、250 Hz轨迹消费及ROS发布均在本地运行。

### 1. 确认云端Server

云端server的标准启动命令为：

```bash
cd /path/to/Isaac-GR00T
source .venv/bin/activate
python gr00t/eval/run_gr00t_server.py \
  --model-path /path/to/checkpoint \
  --embodiment-tag <训练时使用的embodiment-tag> \
  --host 127.0.0.1 \
  --port 5555
```

终端出现`Server ready`后保持该进程运行。`model-path`和`embodiment-tag`必须与微调模型匹配。

推荐通过SSH隧道访问server，避免直接向公网暴露未加密的ZMQ端口。在机器人本地笔记本另开
终端：

```bash
ssh -N -L 5555:127.0.0.1:5555 <user>@<cloud-host>
```

后续client使用`--server-host 127.0.0.1 --server-port 5555`。如果本地5555端口已被占用，
可改为`-L 15555:127.0.0.1:5555`，同时向client传入`--server-port 15555`。

如果必须直接连接，则云端server需要使用`--host 0.0.0.0`，云防火墙只向机器人本地出口IP
开放TCP 5555，本地client使用`--server-host <cloud-host>`。可先检查端口：

```bash
nc -vz <cloud-host> 5555
```

### 2. 准备本地ROS和Python环境

在机器人本地笔记本进入本仓库，并加载ROS及机器人SDK环境：

```bash
cd /home/xukainan/Isaac-GR00T
source /opt/ros/noetic/setup.bash
# 如果机器人消息包来自catkin工作空间，再执行：
source /path/to/robot_ws/devel/setup.bash
source .venv/bin/activate
```

本机GR00T的`.venv`已经安装client依赖，并通过`.pth`加载用户级Kion ROS消息overlay。
同一个`python`必须能导入`rospy`、机器人消息包、`numpy`、`scipy`、`toppra`、`pyzmq`、
`msgpack`和`cv2`。脚本兼容
`zj_humanoid.upperlimb.*`和`upperlimb.*`两种Python命名空间。检查命令：

```bash
python -c \
  "from gr00t.eval.real_robot.TOPPRA.kion_client.client import load_ros_types; print(load_ros_types())"
```

该命令成功不代表已经连接机器人，只表示Python依赖和ROS消息类型可用。

### 3. 检查机器人ROS接口

确认本地能够访问机器人ROS master：

```bash
echo "$ROS_MASTER_URI"
rostopic list | head
```

然后检查本client使用的关键接口：

```bash
rostopic type /zj_humanoid/upperlimb/servol/dual_arm
rostopic type /zj_humanoid/upperlimb/tcp_pose/left_arm
rostopic type /zj_humanoid/upperlimb/tcp_speed/dual_arm
rosservice type /zj_humanoid/upperlimb/set_servo_params
rosservice type /zj_humanoid/upperlimb/clear_servo_params
```

当前代码期望：

```text
upperlimb/DualPose
upperlimb/Pose
upperlimb/TcpSpeed
upperlimb/Servo
upperlimb/Servo
```

再检查反馈和图像是否持续更新：

```bash
rostopic hz /zj_humanoid/upperlimb/tcp_pose/left_arm
rostopic hz /zj_humanoid/upperlimb/tcp_pose/right_arm
rostopic hz /zj_humanoid/upperlimb/tcp_speed/dual_arm
rostopic hz /zj_humanoid/sensor/left_wrist/image_raw/compressed
rostopic hz /zj_humanoid/sensor/right_wrist/image_raw/compressed
rostopic hz /zj_humanoid/sensor/realsense_head/color/image_raw/compressed
```

缺少任意必需观测时，client不会开始执行。

### 4. 先运行Dry-run

保持机器人急停可用，先运行不发布动作的完整链路测试：

```bash
python -m gr00t.eval.real_robot.TOPPRA.kion_client \
  --server-host 127.0.0.1 \
  --server-port 5555 \
  --inference-mode sync \
  --control-frequency 250 \
  --task "move parcel onto conveyor belt one by one" \
  --dry-run
```

直接连接云端时，把`127.0.0.1`替换为云端地址。dry-run仍会采集真实observation、请求云端
action chunk并执行TOPPRA，但不会调用Servo参数service，也不会发布`DualPose`。确认日志中
出现`policy_response`、`plan_success`和`trajectory_ready`，且没有缺少观测、超时或TOPPRA
失败后，按Ctrl-C退出。

### 5. 先进行同步实机测试

第一次运动建议降低速度和加速度约束，并先使用100 Hz确认坐标系、四元数顺序及跟踪方向：

```bash
python -m gr00t.eval.real_robot.TOPPRA.kion_client \
  --server-host 127.0.0.1 \
  --server-port 5555 \
  --inference-mode sync \
  --control-frequency 100 \
  --max-linear-velocity 0.10 \
  --max-angular-velocity 0.50 \
  --max-linear-acceleration 0.50 \
  --max-angular-acceleration 2.0 \
  --task "move parcel onto conveyor belt one by one"
```

同步模式下，云端推理和TOPPRA在Agent producer线程运行。等待新轨迹时，ROS循环持续发布
当前hold target；轨迹准备完成后再从第一个预采样点开始逐点执行。确认行为正确后，再逐步
提高`--control-frequency`和运动约束。

### 6. 进行异步实机Rollout

同步模式验证通过后再启用异步模式：

```bash
python -m gr00t.eval.real_robot.TOPPRA.kion_client \
  --server-host 127.0.0.1 \
  --server-port 5555 \
  --inference-mode async \
  --control-frequency 250 \
  --max-latency-s 0.30 \
  --scheduling-margin-s 0.05 \
  --handoff-margin-s 0.05 \
  --open-loop-horizon 8 \
  --task "move parcel onto conveyor belt one by one"
```

这里的`max-latency-s`应覆盖本地到云端的网络往返和模型推理时间。示例`0.30`仅表示
300 ms；应根据`policy_response`日志中的`latency_s`调整为稳定运行时的高分位延迟，不能
继续沿用局域网环境下不适用的120 ms。`handoff-margin-s`需要大于本地TTS/TOPPRA规划时间和
控制抖动，`pipeline_latency_s`则用于观察推理与规划的总时间。异步模式在推理期间继续执行
旧buffer，并仅在预约handoff点切换到新轨迹。

250 Hz时，脚本调用`set_servo_params`并设置`time=0.004`。可以在另一个终端检查实际发布
频率：

```bash
rostopic hz /zj_humanoid/upperlimb/servol/dual_arm
```

模型返回的pinch保留在Agent内部，但该client只下发双臂target pose，不控制夹爪。

### 7. 停止和异常处理

正常停止使用Ctrl-C。client会停止发布并调用`clear_servo_params`。运行期间应持续关注：

```text
planning_failed
buffer_underflow
handoff_missed
handoff_discontinuous
Control deadline missed
Holding target because ROS observations are stale
```

出现目标位姿跳变、持续buffer underflow、ROS反馈中断或机器人跟踪异常时，应立即停止
rollout并使用实机急停或底层安全接口。Python/ROS发布循环不是硬实时安全控制器，底层仍必须
提供通信超时、watchdog、速度限制和急停。

## 跟踪日志

每次运行生成：

```text
logs/kion_toppra/YYYYMMDD_HHMMSS_xxxxxx/
  metadata.json
  tcp_tracking.csv
```

CSV包含目标/实测双臂pose、实测twist、ROS stamp、本地接收时间、控制发布时间、位置/旋转
误差、轨迹序号、buffer大小和控制周期超时。写盘使用后台线程，不阻塞控制循环；丢行数写入
`metadata.json`。

估计左右臂跟踪延迟：

```bash
python -m gr00t.eval.real_robot.TOPPRA.kion_client.analyze_tracking \
  logs/kion_toppra/<run>/tcp_tracking.csv \
  --max-lag-ms 500
```

结果打印到终端，并保存为同目录的`tracking_report.json`。延迟通过运动阶段中目标位置和
实测位置的时间移位误差估计。计算使用本机`monotonic_ns`记录的目标发布时间和ROS回调接收
时间，不要求ROS stamp与本机时钟严格同步；长时间静止的数据无法可靠估计延迟。
