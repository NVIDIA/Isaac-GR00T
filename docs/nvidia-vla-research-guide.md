# NVIDIA 에코시스템 기반 VLA 연구 가이드

> Isaac Sim, Isaac Lab, GR00T N1.6 통합 워크플로우

---

## 목차

1. [개요 및 시스템 아키텍처](#제1장-개요-및-시스템-아키텍처)
2. [Isaac Sim — 물리 시뮬레이션 엔진](#제2장-isaac-sim--물리-시뮬레이션-엔진)
3. [Isaac Lab — 학습 환경 프레임워크](#제3장-isaac-lab--학습-환경-프레임워크)
4. [GR00T N1.6 — VLA 파운데이션 모델](#제4장-gr00t-n16--vla-파운데이션-모델)
5. [데이터 포맷 — LeRobot v2 + GR00T 확장](#제5장-데이터-포맷--lerobot-v2--gr00t-확장)
6. [파인튜닝 워크플로우](#제6장-파인튜닝-워크플로우)
7. [시뮬레이션 기반 평가](#제7장-시뮬레이션-기반-평가-closed-loop)
8. [배포 아키텍처](#제8장-배포-아키텍처)
9. [종합 워크플로우](#제9장-종합-워크플로우--end-to-end-파이프라인)
10. [부록](#부록)

---

## 제1장: 개요 및 시스템 아키텍처

### 1.1 VLA(Vision-Language-Action) 연구란

VLA 모델은 **카메라 이미지(Vision)** + **자연어 명령(Language)** 을 입력으로 받아 **로봇 행동(Action)** 을 생성하는 end-to-end 모델이다.

전통적 로보틱스 파이프라인(인지 → 계획 → 제어)과 달리, 사전학습된 Vision-Language Model(VLM) 위에 행동 정책(Policy)을 올려 **하나의 모델로 인지와 행동을 통합**한다.

### 1.2 NVIDIA 3-Layer 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│  Layer 3: VLA 파운데이션 모델                              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  GR00T N1.6 (3B params)                             │ │
│  │  Eagle-2 VLM + DiT Diffusion Policy                 │ │
│  │  다중 로봇(Embodiment) 지원, ZMQ 서버-클라이언트 배포    │ │
│  └─────────────────────────────────────────────────────┘ │
│                          ↑ LeRobot v2 데이터              │
│  Layer 2: 학습 환경 프레임워크                              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Isaac Lab v2.3.2                                   │ │
│  │  Manager-based Env (RL/IL)                          │ │
│  │  GR1T2 로봇 설정, Pick-Place 태스크, 텔레오퍼레이션     │ │
│  └─────────────────────────────────────────────────────┘ │
│                          ↑ Python API                    │
│  Layer 1: 시뮬레이션 엔진                                  │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Isaac Sim v5.1.0                                   │ │
│  │  PhysX 5 (GPU 물리) + RTX 렌더링 + USD 씬 포맷        │ │
│  │  Headless 실행, WebRTC 스트리밍                       │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**핵심 관계:**
- **Isaac Sim**은 물리 엔진이다. 단독으로는 환경이 아니다.
- **Isaac Lab**은 Isaac Sim 위에서 RL/IL 환경을 정의하는 프레임워크다.
- **GR00T**은 Isaac Lab과 독립적인 VLA 모델이다. 데이터 포맷(LeRobot v2)으로 연결된다.

### 1.3 현재 인프라 구성

| 구성요소 | 컨테이너 | 이미지 | 역할 |
|---------|---------|--------|------|
| Isaac Sim | `isaac_sim` | `nvcr.io/nvidia/isaac-sim:5.1.0` | Orchard 시뮬레이션 (headless + WebRTC) |
| Isaac Lab | `isaac_lab` | `isaac-lab-base:latest` (커스텀 빌드) | GR1T2 환경, RoboCasa 클라이언트 |
| GR00T | `groot` (별도 세션) | GR00T 전용 이미지 | N1.6 추론 서버 (TCP 5555) |

모든 컨테이너는 `network_mode: host`로 실행되어 `localhost:5555`로 통신한다.

```
서버 (RTX 5090)
├── groot 컨테이너 ──── GR00T N1.6 추론 서버 (port 5555)
│                           ↕ ZMQ (TCP)
├── isaac_lab 컨테이너 ── Isaac Lab + RoboCasa 클라이언트
│                         /workspace/isaaclab (IsaacLab 소스)
│                         /workspace/Isaac-GR00T (GR00T 소스, 호스트 마운트)
│
├── isaac_sim 컨테이너 ── Orchard 시뮬레이션 (WebRTC :8211)
│
└── 호스트 파일 시스템
    └── /home/ailab/sangbum/NVIDIA/
        ├── IsaacSim/      (docker-compose.yml, 스크립트)
        ├── IsaacLab/      (소스코드)
        └── Isaac-GR00T/   (소스코드, 두 컨테이너에서 공유)
```

---

## 제2장: Isaac Sim — 물리 시뮬레이션 엔진

### 2.1 핵심 기술 스택

| 기술 | 역할 | 상세 |
|------|------|------|
| **PhysX 5** | GPU 가속 물리 엔진 | 강체, 관절, 접촉, 소프트바디 시뮬레이션 |
| **RTX 렌더링** | 포토리얼리스틱 이미지 생성 | 레이트레이싱 기반 카메라 렌더링 (RGB, Depth, Segmentation) |
| **USD** | 씬 포맷 표준 | Universal Scene Description. 모든 에셋/씬을 USD로 관리 |
| **Omniverse Kit** | 확장(Extension) 프레임워크 | Isaac Sim의 모든 기능은 Extension으로 구현 |

### 2.2 실행 방식

```bash
# Headless 실행 (서버용)
docker compose up -d isaac_sim

# WebRTC 스트리밍 접속
# http://<호스트IP>:8211/streaming/webrtc-client

# GUI 모드 (로컬 모니터 필요)
docker compose up -d isaac_sim_gui

# 개발용 bash 접속
docker compose run --rm isaac_sim bash
```

Python에서 Isaac Sim 초기화:

```python
from isaacsim import SimulationApp

CONFIG = {
    "headless": True,           # GUI 없이 실행
    "renderer": "RayTracedLighting",  # RTX 렌더러
    "width": 1280,
    "height": 720,
    "enable_livestream": True,  # WebRTC 스트리밍
    "livestream_port": 8211,
}

simulation_app = SimulationApp(CONFIG)
```

### 2.3 VLA 연구에서 Isaac Sim의 역할

1. **카메라 이미지 생성**: RTX 렌더링으로 포토리얼리스틱 RGB/Depth 이미지 → VLA 모델 입력
2. **물리 시뮬레이션**: PhysX 기반 물체 조작 (grasp, push, place) → 물리적으로 정확한 행동 결과
3. **대규모 병렬화**: GPU 가속으로 수천 개 환경 동시 실행 → 대규모 데이터 수집
4. **Domain Randomization**: 조명, 텍스처, 물리 파라미터 랜덤화 → Sim-to-Real 전이 성능 향상

---

## 제3장: Isaac Lab — 학습 환경 프레임워크

### 3.1 Manager-based 아키텍처

Isaac Lab은 환경을 **6개 Manager**로 분해하여 구성한다:

```
┌──────────────────────────────────────────────────────┐
│                ManagerBasedRLEnv                       │
│                                                       │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ ActionMgr   │  │ ObservationMgr│  │ EventMgr     │ │
│  │ (IK, Joint) │  │ (joint_pos,  │  │ (reset,      │ │
│  │             │  │  camera_img) │  │  randomize)  │ │
│  └─────────────┘  └──────────────┘  └──────────────┘ │
│                                                       │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ RewardMgr   │  │TerminationMgr│  │ RecorderMgr  │ │
│  │ (reach,     │  │ (timeout,    │  │ (데이터 기록)  │ │
│  │  grasp)     │  │  falling)    │  │              │ │
│  └─────────────┘  └──────────────┘  └──────────────┘ │
│                                                       │
│  ┌──────────────────────────────────────────────────┐ │
│  │           InteractiveScene (Isaac Sim)            │ │
│  │  Robot: ArticulationCfg  │  Object: RigidObject  │ │
│  │  Table: AssetBaseCfg     │  Camera: CameraCfg    │ │
│  └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

| Manager | 역할 | 예시 |
|---------|------|------|
| **ActionManager** | raw action → 시뮬레이션 제어 명령 변환 | Joint position, IK (PinkIK) |
| **ObservationManager** | 시뮬레이션 상태 → 관측값 생성 | 관절 위치, EEF 포즈, 카메라 이미지 |
| **EventManager** | 환경 이벤트 처리 | 리셋, Domain Randomization |
| **RewardManager** | 보상 신호 계산 (RL용) | 물체 도달, 파지, 배치 보상 |
| **TerminationManager** | 에피소드 종료 조건 | 타임아웃, 로봇 쓰러짐 |
| **RecorderManager** | 에피소드 데이터 기록 | 데모 데이터 수집 |

### 3.2 GR1T2 로봇 설정

**파일:** `IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/fourier.py`

GR1T2는 Fourier Intelligence의 휴머노이드 로봇으로, Isaac Lab에 두 가지 설정이 제공된다:

| 설정 | 용도 | 특징 |
|------|------|------|
| `GR1T2_CFG` | 범용 | 기본 물리 파라미터, gravity disabled |
| `GR1T2_HIGH_PD_CFG` | 조작 태스크 | trunk/arms에 높은 PD 게인 (stiffness=4400, damping=40) |

**액추에이터 그룹 (7개):**

| 그룹 | 관절 패턴 | 용도 |
|------|----------|------|
| head | `head_.*` | 머리 회전 |
| trunk | `waist_.*` | 허리 (3 DoF) |
| legs | `*_hip_*`, `*_knee_*`, `*_ankle_*` | 하체 보행 |
| right-arm | `right_shoulder_*`, `right_elbow_*`, `right_wrist_*` | 오른팔 (7 DoF) |
| left-arm | `left_shoulder_*`, `left_elbow_*`, `left_wrist_*` | 왼팔 (7 DoF) |
| right-hand | `R_*` | 오른손 (6 DoF, Fourier Hand) |
| left-hand | `L_*` | 왼손 (6 DoF, Fourier Hand) |

**USD 에셋 경로:**
```
{ISAAC_NUCLEUS_DIR}/Robots/FourierIntelligence/GR-1/GR1T2_fourier_hand_6dof/GR1T2_fourier_hand_6dof.usd
```

### 3.3 Pick-Place 태스크 환경

**파일:** `IsaacLab/source/isaaclab_tasks/.../pick_place/pickplace_gr1t2_env_cfg.py`

GR1T2용으로 사전 구성된 태스크들:

| 환경 설정 | 설명 |
|----------|------|
| `pickplace_gr1t2_env_cfg.py` | 기본 Pick-Place (양팔) |
| `pickplace_gr1t2_waist_enabled_env_cfg.py` | 허리 움직임 포함 |
| `nutpour_gr1t2_base_env_cfg.py` | 너트 붓기 태스크 |
| `exhaustpipe_gr1t2_base_env_cfg.py` | 파이프 조립 태스크 |

각 환경은 다음으로 구성된다:
- **Scene**: PackingTable + GR1T2_HIGH_PD_CFG + 조작 대상 Object
- **ActionManager**: PinkIK 기반 end-effector 포즈 제어
- **ObservationManager**: EEF 위치/방향, 물체 위치, 접촉력

### 3.4 Mimic — 모방학습 데이터 생성

**파일:** `IsaacLab/source/isaaclab_mimic/`

Isaac Lab Mimic은 소수의 인간 시연 데이터를 다양한 초기 조건으로 **자동 증폭**하는 시스템이다:

```
인간 시연 데이터 (10개)
     ↓ DataGenerator
자동 증폭 (1,000+ 에피소드)
     ↓ 물체 위치 변환, 궤적 재계획
LeRobot v2 포맷 데이터셋
     ↓
GR00T 파인튜닝
```

- `ManagerBasedRLMimicEnv`: 모방학습 전용 환경 클래스
- SubTask 단위로 시연 분할 (reach → grasp → place)
- Selection Strategy: 가장 가까운 시연 선택 후 물체 좌표계로 변환

### 3.5 텔레오퍼레이션 지원

| 디바이스 | 클래스 | DoF | 용도 |
|---------|--------|-----|------|
| OpenXR (VR) | `openxr/` | 6 DoF + 손가락 | GR1T2 양팔+손 조작 |
| SpaceMouse | `SE3Spacemouse` | 6 DoF | 단일 팔 조작 |
| Gamepad | `SE3Gamepad` | 6 DoF | 범용 |
| 키보드 | `SE3Keyboard` | 6 DoF | 간단한 테스트 |

GR1T2 전용 retargeter: `devices/openxr/retargeters/humanoid/fourier/gr1t2_retargeter.py`

---

## 제4장: GR00T N1.6 — VLA 파운데이션 모델

### 4.1 모델 아키텍처 (Dual-System)

```
┌──────────────── 입력 ────────────────┐
│  카메라 이미지 (RGB, uint8)            │
│  로봇 상태 (joint positions, float32) │
│  자연어 명령 ("pick up the bottle")    │
└──────────────────────────────────────┘
              ↓
┌──────────── System 2: Vision-Language ────────────┐
│  Eagle-2 VLM                                      │
│  ├── SigLIP-2 Image Encoder (이미지 임베딩)         │
│  ├── SmolLM2 LLM (텍스트 이해)                     │
│  └── Cross-Attention (이미지-텍스트 융합)            │
│                                                    │
│  출력: Vision-Language Embedding (2048D)            │
└──────────────────────────────────────────────────┘
              ↓ cross-attention
┌──────────── System 1: Action Generation ──────────┐
│  DiT (Diffusion Transformer) + Flow Matching      │
│  ├── CategorySpecificMLP (embodiment별 state 인코딩)│
│  ├── AlternateVLDiT (32 layers, N1.6)              │
│  └── Action Decoder (연속 행동 출력)                 │
│                                                    │
│  출력: Action Chunk (B, 16, D) — 16스텝 행동 시퀀스  │
└──────────────────────────────────────────────────┘
```

**핵심 특징:**
- **총 3B 파라미터** (VLM ~1.34B + DiT ~1.66B)
- **Flow Matching**: 기존 DDPM 대비 빠른 denoising (4 steps)
- **CategorySpecificMLP**: 로봇(embodiment)별로 다른 state/action 인코더 → 다중 로봇 지원
- **State-Relative Action**: 현재 상태 대비 상대적 행동 예측 → 드리프트 감소

### 4.2 추론 성능

| GPU | torch.compile | E2E 지연 | 추론 주파수 |
|-----|---------------|---------|-----------|
| **RTX 5090** | O | 37ms | 27.3 Hz |
| H100 | O | 38ms | 26.3 Hz |
| RTX 4090 | O | 44ms | 22.8 Hz |
| Jetson Thor | O | 105ms | 9.5 Hz |

- 4 denoising steps, single camera view 기준
- Action chunk: 16 steps → 실제 제어 주파수는 더 높음 (chunk 내 보간)

### 4.3 EmbodimentTag 시스템

**파일:** `Isaac-GR00T/gr00t/data/embodiment_tags.py`

사전 등록된 로봇 태그:

| 태그 | 로봇 | state 차원 | action horizon |
|------|------|-----------|----------------|
| `GR1` | Fourier GR1 | 44 | 16 |
| `UNITREE_G1` | Unitree G1 | 30+ | 30 |
| `LIBERO_PANDA` | Franka Panda | 7+1 | 16 |
| `OXE_WIDOWX` | WidowX | 7 | 8 |
| `OXE_GOOGLE` | Google Robot | 7 | 8 |
| `BEHAVIOR_R1_PRO` | Galaxea R1 Pro | 28 | 32 |
| **`NEW_EMBODIMENT`** | **커스텀 로봇** | **사용자 정의** | **사용자 정의** |

커스텀 로봇을 추가할 때는 항상 `NEW_EMBODIMENT` 태그를 사용한다.

### 4.4 Gr00tPolicy API

```python
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag

# 모델 로딩
policy = Gr00tPolicy(
    model_path="nvidia/GR00T-N1.6-3B",  # HuggingFace 또는 로컬 경로
    embodiment_tag=EmbodimentTag.GR1,
    device="cuda:0",
)

# 관측값 구성
observation = {
    "video": {
        "ego_view": np.ndarray,   # shape: (B, T, H, W, 3), dtype: uint8
    },
    "state": {
        "left_arm": np.ndarray,   # shape: (B, T, 7), dtype: float32
        "right_arm": np.ndarray,  # shape: (B, T, 7), dtype: float32
        # ... 총 44 DoF
    },
    "language": {
        "task": [["pick up the bottle"]],  # shape: (B, 1)
    },
}

# 행동 예측
action, info = policy.get_action(observation)
# action: {"left_arm": np.ndarray(B, 16, 7), "right_arm": ...}
# 행동 값은 물리 단위 (radians), 정규화되지 않음

# 에피소드 간 리셋
policy.reset()
```

### 4.5 모델 체크포인트

| 모델 | 설명 |
|------|------|
| `nvidia/GR00T-N1.6-3B` | 기본 사전학습 모델 |
| `nvidia/GR00T-N1.6-bridge` | WidowX 파인튜닝 |
| `nvidia/GR00T-N1.6-fractal` | Google Robot 파인튜닝 |
| `nvidia/GR00T-N1.6-BEHAVIOR1k` | Galaxea R1 Pro 파인튜닝 |
| `nvidia/GR00T-N1.6-G1-PnPAppleToPlate` | Unitree G1 파인튜닝 |

모든 체크포인트는 HuggingFace에서 다운로드 가능.

---

## 제5장: 데이터 포맷 — LeRobot v2 + GR00T 확장

### 5.1 디렉토리 구조

```
dataset_root/
├── meta/
│   ├── info.json              # 데이터셋 메타 정보 (fps, features, total_episodes)
│   ├── episodes.jsonl         # 에피소드 목록 (1줄 = 1 에피소드)
│   ├── tasks.jsonl            # 태스크 설명 텍스트
│   ├── modality.json          # [GR00T 전용] state/action 슬라이스 매핑
│   ├── stats.json             # 정규화 통계 (min, max, mean, std)
│   └── relative_stats.json    # 상대 행동용 통계
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet  # 관절 상태 + 행동 + 메타데이터
│       └── episode_000001.parquet
└── videos/
    └── chunk-000/
        └── observation.images.ego_view/
            ├── episode_000000.mp4   # 카메라 영상
            └── episode_000001.mp4
```

### 5.2 Parquet 파일 스키마

각 행(row)은 하나의 타임스텝:

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `observation.state` | `list[float32]` | 연결된 관절 상태 배열 (예: 44차원) |
| `action` | `list[float32]` | 연결된 행동 배열 (예: 44차원) |
| `timestamp` | `float32` | 프레임 타임스탬프 (초) |
| `frame_index` | `int64` | 에피소드 내 프레임 인덱스 |
| `episode_index` | `int64` | 에피소드 ID |
| `index` | `int64` | 전체 데이터셋 내 글로벌 인덱스 |
| `task_index` | `int64` | tasks.jsonl 내 태스크 인덱스 |
| `next.done` | `bool` | 에피소드 종료 여부 |

### 5.3 modality.json — GR00T의 핵심 설정

이 파일은 Parquet의 `observation.state`와 `action` 배열을 **의미 있는 부분으로 슬라이스**하는 매핑을 정의한다.

**GR1 로봇 예시 (44 DoF):**

```json
{
    "state": {
        "left_arm":   {"start": 0,  "end": 7},
        "left_hand":  {"start": 7,  "end": 13},
        "left_leg":   {"start": 13, "end": 19},
        "neck":       {"start": 19, "end": 22},
        "right_arm":  {"start": 22, "end": 29},
        "right_hand": {"start": 29, "end": 35},
        "right_leg":  {"start": 35, "end": 41},
        "waist":      {"start": 41, "end": 44}
    },
    "action": {
        "left_arm":   {"start": 0,  "end": 7},
        "left_hand":  {"start": 7,  "end": 13},
        "left_leg":   {"start": 13, "end": 19},
        "neck":       {"start": 19, "end": 22},
        "right_arm":  {"start": 22, "end": 29},
        "right_hand": {"start": 29, "end": 35},
        "right_leg":  {"start": 35, "end": 41},
        "waist":      {"start": 41, "end": 44}
    },
    "video": {
        "ego_view_bg_crop_pad_res256_freq20": {
            "original_key": "observation.images.ego_view"
        }
    },
    "annotation": {
        "human.action.task_description": {},
        "human.validity": {},
        "human.coarse_action": {
            "original_key": "annotation.human.action.task_description"
        }
    }
}
```

**동작 원리:**
- `state.left_arm: {start: 0, end: 7}` → `observation.state[0:7]`이 왼팔 관절 7개
- `video.ego_view_...: {original_key: "observation.images.ego_view"}` → 비디오 파일 매핑
- GR00T 데이터 로더가 이 매핑으로 각 modality를 분리하여 모델에 전달

### 5.4 ModalityConfig — Python 학습 설정

`modality.json`은 데이터 구조를, `ModalityConfig`는 학습 시 **어떻게 로딩할지**를 정의한다.

```python
from gr00t.data.types import (
    ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig
)

config = {
    "video": ModalityConfig(
        delta_indices=[0],              # 현재 프레임만 사용
        modality_keys=["ego_view"],     # modality.json의 video 키
    ),
    "state": ModalityConfig(
        delta_indices=[0],              # 현재 상태만
        modality_keys=["left_arm", "right_arm", "waist"],  # 사용할 부분만 선택
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),  # 16스텝 예측 horizon
        modality_keys=["left_arm", "right_arm", "waist"],
        action_configs=[
            ActionConfig(  # left_arm: 상대 제어
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(  # right_arm: 상대 제어
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(  # waist: 절대 제어
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}
```

**ActionConfig 3축:**

| 축 | 옵션 | 설명 |
|----|------|------|
| **Representation** | `RELATIVE` | 현재 상태 대비 상대 변화 (권장) |
| | `ABSOLUTE` | 절대 목표 위치 |
| | `DELTA` | 프레임 간 차이 |
| **Type** | `NON_EEF` | 관절 공간 제어 |
| | `EEF` | End-Effector 카르테시안 공간 |
| **Format** | `DEFAULT` | 기본 포맷 |
| | `XYZ_ROT6D` | 위치 3D + 6D 회전 |
| | `XYZ_ROTVEC` | 위치 3D + 회전 벡터 |

> **중요:** `action_configs`의 순서는 `modality_keys`의 순서와 1:1 대응해야 한다.

### 5.5 정규화 통계 생성

```bash
python gr00t/data/stats.py <dataset_path> <embodiment_tag>
```

- `meta/stats.json`: 전체 통계 (min, max, mean, std, q01, q99)
- `meta/relative_stats.json`: 상대 행동용 통계 (shape: `[horizon, dim]`)
- **주의:** `delta_indices` 변경 시 반드시 재생성해야 함 (차원 불일치 오류 발생)

---

## 제6장: 파인튜닝 워크플로우

### 6.1 데이터 준비 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| 에피소드 수 | 50 | 200+ |
| 수집 주파수 | 20 Hz | 30-50 Hz |
| 카메라 | 1대 (ego view) | 2대 (ego + wrist) |
| 에피소드 품질 | 성공한 시연만 | 다양한 초기 조건 |

### 6.2 데이터 변환 파이프라인

```
Isaac Lab 텔레오퍼레이션
(OpenXR / SpaceMouse)
        ↓ joint states + camera images
Raw 데이터 (HDF5 또는 커스텀 포맷)
        ↓ 타임스탬프 정렬, 실패 에피소드 필터링, idle 구간 트리밍
정제된 데이터
        ↓ Parquet + MP4 변환
LeRobot v2 구조
        ↓ modality.json 작성
GR00T LeRobot 데이터셋
        ↓ python gr00t/data/stats.py
학습 준비 완료
```

**LeRobot v3 → v2 변환 스크립트:**
```bash
python Isaac-GR00T/scripts/lerobot_conversion/convert_v3_to_v2.py \
    --input <v3_dataset_path> \
    --output <v2_output_path>
```

### 6.3 커스텀 ModalityConfig 작성

**파일:** `examples/SO100/so100_config.py`를 참고하여 작성

```python
# my_robot_config.py
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import *

my_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front", "wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["single_arm", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["single_arm", "gripper"],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.RELATIVE,
                        type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE,
                        type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(my_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
```

### 6.4 파인튜닝 실행

```bash
export NUM_GPUS=1
CUDA_VISIBLE_DEVICES=0 python \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path ./data/my_robot_demos \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path ./my_robot_config.py \
    --num-gpus $NUM_GPUS \
    --output-dir ./checkpoints/my_robot \
    --max-steps 5000 \
    --save-steps 1000 \
    --global-batch-size 32 \
    --dataloader-num-workers 4 \
    --use-wandb
```

**주요 파라미터:**

| 파라미터 | 설명 | 권장값 |
|---------|------|--------|
| `--base-model-path` | 사전학습 체크포인트 | `nvidia/GR00T-N1.6-3B` |
| `--max-steps` | 총 학습 스텝 | 2,000 ~ 10,000 |
| `--global-batch-size` | 배치 크기 | GPU VRAM 최대 활용 |
| `--tune-llm` | LLM 레이어 파인튜닝 여부 | 데이터 적으면 False |
| `--tune-visual` | Vision 인코더 파인튜닝 | 데이터 적으면 False |
| `--tune-diffusion-model` | DiT 파인튜닝 (항상 True) | True |
| `--state-dropout-prob` | State dropout 확률 | 0.1 (로버스트 제어) |

### 6.5 Open-Loop 평가

학습된 모델이 데이터셋의 ground-truth 행동과 얼마나 일치하는지 확인:

```bash
python gr00t/eval/open_loop_eval.py \
    --dataset-path ./data/my_robot_demos \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path ./checkpoints/my_robot/checkpoint-5000 \
    --modality-config-path ./my_robot_config.py \
    --traj-ids 0 \
    --action-horizon 16 \
    --steps 400 \
    --modality-keys single_arm gripper
```

시각화된 행동 비교 영상이 출력된다. 예측 행동이 ground-truth와 유사한 패턴을 보이면 학습이 수렴된 것.

---

## 제7장: 시뮬레이션 기반 평가 (Closed-Loop)

### 7.1 RoboCasa GR1 벤치마크 (현재 동작 확인)

**주의:** 이 벤치마크는 **MuJoCo 기반**이다. Isaac Sim을 사용하지 않는다.

24개 테이블탑 조작 태스크, 각 1,000개 시연 데이터.

**현재 설정:**
```
groot 컨테이너                    isaac_lab 컨테이너
┌──────────────────┐             ┌──────────────────┐
│ GR00T Server     │←─TCP 5555──→│ RoboCasa Client  │
│ --use-sim-policy │             │ (MuJoCo 기반)     │
│   -wrapper       │             │ rollout_policy.py │
└──────────────────┘             └──────────────────┘
```

서버 실행:
```bash
# groot 컨테이너에서
bash /workspace/gr00t/scripts/start_server.sh
```

클라이언트 실행:
```bash
# isaac_lab 컨테이너에서
bash /workspace/isaaclab/scripts/run_robocasa_client.sh
```

### 7.2 Isaac Sim 기반 평가 환경 (목표)

Isaac Lab의 GR1T2 pick_place 환경을 GR00T 평가용으로 연결하는 구조:

```
groot 컨테이너                    isaac_lab 컨테이너
┌──────────────────┐             ┌──────────────────────────┐
│ GR00T Server     │←─TCP 5555──→│ Isaac Lab GR1T2 환경      │
│ N1.6 추론        │             │ ┌──────────────────────┐ │
│                  │  observation │ │ PhysX 5 물리 시뮬      │ │
│                  │←────────────│ │ RTX 카메라 렌더링      │ │
│                  │  action     │ │ WebRTC 스트리밍       │ │
│                  │────────────→│ └──────────────────────┘ │
└──────────────────┘             └──────────────────────────┘
                                          ↕
                                  http://<IP>:8211
                                  (브라우저로 실시간 관찰)
```

**필요한 작업:**
1. Isaac Lab GR1T2 환경에 카메라(ego_view) 추가
2. Observation → GR00T Policy API 포맷 변환 래퍼 작성
3. GR00T Action → Isaac Lab ActionManager 매핑
4. `PolicyClient` 연동 루프 구현

### 7.3 ReplayPolicy로 파이프라인 검증

학습된 모델 없이, 기록된 데이터의 행동을 그대로 재생하여 파이프라인 정합성을 확인:

```bash
# 서버를 ReplayPolicy 모드로 실행
python gr00t/eval/run_gr00t_server.py \
    --dataset-path ./data/my_robot_demos \
    --embodiment-tag NEW_EMBODIMENT \
    --host 0.0.0.0 --port 5555 \
    --execution-horizon 8
```

100% 성공률이 아니면 관측/행동 포맷 불일치를 의심해야 한다.

---

## 제8장: 배포 아키텍처

### 8.1 ZMQ 서버-클라이언트

```
┌─────────────────────┐          TCP 5555          ┌─────────────────────┐
│    GR00T Server     │◄════════════════════════►│    Robot Client     │
│  (GPU 서버)          │         ZMQ REQ/REP       │  (로봇 또는 Sim)     │
│                     │                            │                     │
│  Gr00tPolicy        │    observation (msgpack)   │  PolicyClient       │
│   + SimWrapper      │◄───────────────────────── │   host, port        │
│                     │    action (msgpack)         │                     │
│                     │ ──────────────────────────►│                     │
└─────────────────────┘                            └─────────────────────┘
```

- 통신: ZeroMQ REQ/REP 패턴
- 직렬화: msgpack (numpy 배열 바이너리 전송)
- 타임아웃: 기본 15초 (`timeout_ms=15000`)

### 8.2 Closed-Loop 제어 루프

```python
from gr00t.policy.server_client import PolicyClient
import numpy as np

policy = PolicyClient(host="localhost", port=5555)

while not done:
    # 1. 관측값 수집
    image = camera.get_frame()          # (H, W, 3), uint8
    joint_pos = robot.get_joint_pos()   # (D,), float32

    # 2. GR00T 포맷으로 변환
    obs = {
        "video.ego_view": image[None, None, ...],      # (1, 1, H, W, 3)
        "state.left_arm": joint_pos[:7][None, None],   # (1, 1, 7)
        "state.right_arm": joint_pos[7:14][None, None],
        "annotation.human.coarse_action": ("pick up the bottle",),
    }

    # 3. 행동 예측
    action, info = policy.get_action(obs)

    # 4. 행동 실행 (action chunk의 첫 스텝)
    next_joint_pos = action["left_arm"][0, 0, :]  # (7,)
    robot.set_joint_pos(next_joint_pos)

    time.sleep(1.0 / 30.0)  # 30 Hz 제어
```

### 8.3 일반적인 문제와 해결

| 증상 | 원인 | 해결 |
|------|------|------|
| **Jittering** (떨림) | 모델 언더트레이닝, chunk 간 불연속 | 학습 데이터 추가, State-Relative Action 사용 |
| **Stop-and-Go** | 추론 지연 > 33ms (30 FPS) | RTC + 비동기 추론, action chunk ≥ 32 |
| **드리프트** | Absolute action 누적 오차 | Relative action으로 변경 |
| **관측 불일치** | sim/real 카메라 캘리브레이션 차이 | Domain Randomization 적용 |

---

## 제9장: 종합 워크플로우 — End-to-End 파이프라인

### 9.1 전체 파이프라인

```
Phase 1: 환경 구축                      Phase 2: 데이터 수집
┌────────────────────┐                  ┌────────────────────┐
│ Isaac Lab GR1T2    │                  │ 텔레오퍼레이션       │
│ pick_place 환경     │─────────────────→│ (OpenXR/SpaceMouse) │
│ + 카메라 설정       │                  │ 100+ 에피소드 수집   │
└────────────────────┘                  └─────────┬──────────┘
                                                  │
                                                  ↓
Phase 3: 학습                           ┌────────────────────┐
┌────────────────────┐                  │ LeRobot v2 변환     │
│ GR00T N1.6         │◄────────────────│ + modality.json     │
│ 파인튜닝            │                  │ + stats.json 생성   │
│ (2,000~10,000 steps)│                  └────────────────────┘
└─────────┬──────────┘
          │
          ↓
Phase 4: 평가                           Phase 5: 배포
┌────────────────────┐                  ┌────────────────────┐
│ Open-Loop 평가      │                  │ ZMQ 서버-클라이언트  │
│ → Sim 평가 (Isaac)  │─────────────────→│ 실 로봇 배포        │
│ → Real 평가         │                  │ (Jetson / PC)      │
└────────────────────┘                  └────────────────────┘
```

### 9.2 현재 상태와 다음 단계

**완료:**
- [x] Docker 인프라 구성 (Isaac Sim, Isaac Lab, GR00T 컨테이너)
- [x] Isaac Sim Orchard 시뮬레이션 (headless + WebRTC)
- [x] RoboCasa GR1 벤치마크 실행 (MuJoCo 기반, 서버-클라이언트)
- [x] GR00T N1.6 추론 서버 동작 확인

**다음 단계:**
1. Isaac Lab GR1T2 pick_place 환경을 headless + WebRTC로 실행
2. 환경에 카메라(ego_view) 추가 및 관측값 확인
3. GR00T ↔ Isaac Lab 연결 래퍼(wrapper) 작성
4. 텔레오퍼레이션으로 시연 데이터 수집 (100+ 에피소드)
5. LeRobot v2 포맷 변환 및 GR00T 파인튜닝
6. Isaac Lab 환경에서 closed-loop 평가
7. (선택) 실 GR1T2 로봇 배포

---

## 부록

### 부록 A: 주요 파일 경로 참조표

| 구성요소 | 파일 경로 |
|---------|---------|
| Docker 설정 | `IsaacSim/docker-compose.yml` |
| GR1T2 로봇 설정 | `IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/fourier.py` |
| GR1T2 Pick-Place 환경 | `IsaacLab/source/isaaclab_tasks/.../pick_place/pickplace_gr1t2_env_cfg.py` |
| GR1T2 Mimic 환경 | `IsaacLab/source/isaaclab_mimic/envs/pinocchio_envs/pickplace_gr1t2_mimic_env.py` |
| GR1T2 텔레오퍼레이션 | `IsaacLab/source/isaaclab/isaaclab/devices/openxr/retargeters/humanoid/fourier/` |
| EmbodimentTag 정의 | `Isaac-GR00T/gr00t/data/embodiment_tags.py` |
| ModalityConfig 등록 | `Isaac-GR00T/gr00t/configs/data/embodiment_configs.py` |
| 파인튜닝 스크립트 | `Isaac-GR00T/gr00t/experiment/launch_finetune.py` |
| 서버 실행 | `Isaac-GR00T/gr00t/eval/run_gr00t_server.py` |
| PolicyClient | `Isaac-GR00T/gr00t/policy/server_client.py` |
| 데이터 로더 | `Isaac-GR00T/gr00t/data/dataset/lerobot_episode_loader.py` |
| GR1 demo modality | `Isaac-GR00T/demo_data/gr1.PickNPlace/meta/modality.json` |
| SO-100 config 예제 | `Isaac-GR00T/examples/SO100/so100_config.py` |
| 데이터 준비 가이드 | `Isaac-GR00T/getting_started/data_preparation.md` |
| 파인튜닝 가이드 | `Isaac-GR00T/getting_started/finetune_new_embodiment.md` |
| 배포 가이드 | `Isaac-GR00T/getting_started/real_world_deployment.md` |

### 부록 B: GR1 modality.json 관절 매핑

| 부위 | 인덱스 | 차원 | 상세 |
|------|--------|------|------|
| left_arm | 0:7 | 7 | 왼팔 (shoulder 3 + elbow 2 + wrist 2) |
| left_hand | 7:13 | 6 | 왼손 (Fourier Hand 6 DoF) |
| left_leg | 13:19 | 6 | 왼다리 (hip 3 + knee 1 + ankle 2) |
| neck | 19:22 | 3 | 목 (pitch, roll, yaw) |
| right_arm | 22:29 | 7 | 오른팔 |
| right_hand | 29:35 | 6 | 오른손 |
| right_leg | 35:41 | 6 | 오른다리 |
| waist | 41:44 | 3 | 허리 (pitch, roll, yaw) |
| **합계** | 0:44 | **44** | **전신** |

### 부록 C: 외부 참고 자료

| 자료 | URL |
|------|-----|
| GR00T N1 논문 | https://arxiv.org/abs/2503.14734 |
| GR00T GitHub | https://github.com/NVIDIA/Isaac-GR00T |
| GR00T N1.6 모델 | https://huggingface.co/nvidia/GR00T-N1.6-3B |
| Isaac Lab 문서 | https://isaac-sim.github.io/IsaacLab |
| Isaac Sim 문서 | https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ |
| NVIDIA 기술 블로그 (Sim-to-Real) | https://developer.nvidia.com/blog/building-generalist-humanoid-capabilities-with-nvidia-isaac-gr00t-n1-6-using-a-sim-to-real-workflow |
| LeRobot SO-101 파인튜닝 | https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning |
| Cosmos Tokenizer | https://github.com/NVIDIA/Cosmos-Tokenizer |

### 부록 D: 트러블슈팅

| 문제 | 원인 | 해결 |
|------|------|------|
| `IndexError` in stats loading | `delta_indices` 변경 후 stats 미재생성 | `python gr00t/data/stats.py` 재실행 |
| `AssertionError: 'video' key` | 서버에 `--use-sim-policy-wrapper` 미사용 | 서버 실행 시 플래그 추가 |
| `dubious ownership` in git | Docker 내 마운트 경로 소유권 불일치 | `git config --global --add safe.directory <path>` |
| venv python `No such file` | uv python 심볼릭 링크 깨짐 | `uv python install 3.10` 재실행 |
| Docker GPU 접근 불가 | NVIDIA Container Toolkit 미설치 | `nvidia-ctk runtime configure` |
| WebRTC 연결 실패 | 방화벽 또는 포트 미개방 | 포트 8211 개방, `--net host` 확인 |

---

> **문서 버전:** 2026-03-23
> **작성 환경:** RTX 5090, Isaac Sim 5.1.0, Isaac Lab 2.3.2, GR00T N1.6-3B
