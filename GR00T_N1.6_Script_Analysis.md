# GR00T N1.6 스크립트 분석 & 커스텀 수정 가이드

> 작성일: 2026-03-18
> 목적: fine-tuning 시 어디를 수정해야 하는지, 스크립트 간 연결 관계, modality/VLM 수정 포인트 정리

---

## 1. 전체 디렉토리 구조 & 역할

```
Isaac-GR00T/
├── gr00t/                          # 핵심 라이브러리 (모델, 데이터, 학습, 평가)
│   ├── model/
│   │   └── policy.py               # ★ Gr00tPolicy — 모델 로드 & 추론 핵심 클래스
│   ├── data/
│   │   └── embodiment_tags.py      # ★ EmbodimentTag — 로봇 타입 정의 (GR1, PANDA, NEW_EMBODIMENT 등)
│   ├── configs/
│   │   └── finetune_config.py      # fine-tune 하이퍼파라미터 설정
│   ├── experiment/
│   │   ├── launch_finetune.py      # ★ fine-tune 엔트리포인트 (N1.6용)
│   │   └── launch_train.py         # 상세 학습 설정용 (고급)
│   ├── eval/
│   │   ├── run_gr00t_server.py     # ★ 추론 서버 (ZeroMQ 기반)
│   │   ├── rollout_policy.py       # 시뮬레이션 rollout 클라이언트
│   │   ├── open_loop_eval.py       # 오프라인 open-loop 평가
│   │   ├── sim/                    # 시뮬레이션 환경별 세팅 (LIBERO, SimplerEnv, RoboCasa)
│   │   └── real_robot/             # 실제 로봇 배포 코드 (SO100 등)
│   └── policy/
│       └── server_client.py        # PolicyClient — 서버에 추론 요청하는 클라이언트
│
├── scripts/
│   └── deployment/
│       ├── standalone_inference_script.py  # ★ 단독 추론 (서버 없이, 12GB에서 가능)
│       ├── export_onnx_n1d6.py            # ONNX 모델 export
│       ├── build_tensorrt_engine.py       # TensorRT 엔진 빌드
│       └── benchmark_inference.py         # 추론 벤치마크
│
├── examples/                       # 로봇/벤치마크별 설정 예시
│   ├── SO100/
│   │   └── so100_config.py         # ★ modality config 예시 (커스텀 로봇 참고용)
│   ├── LIBERO/
│   ├── SimplerEnv/
│   ├── robocasa/
│   ├── robocasa-gr1-tabletop-tasks/
│   ├── BEHAVIOR/
│   ├── GR00T-WholeBodyControl/
│   └── PointNav/
│
├── getting_started/                # 튜토리얼 & 가이드
│   ├── data_preparation.md         # 데이터 포맷 변환 가이드
│   ├── finetune_new_embodiment.md  # ★ 새 로봇 fine-tune 가이드
│   ├── policy.md                   # Policy API 상세 문서
│   ├── modality_config.md          # ★ modality 설정 가이드
│   └── hardware_recommendation.md  # 하드웨어 권장사항
│
├── demo_data/                      # 샘플 데이터셋
│   └── gr1.PickNPlace/             # GR1 데모 데이터
│
└── docker/                         # Docker 세팅
```

---

## 2. 스크립트 연결 관계 (파이프라인 흐름)

### 2.1 추론 파이프라인

```
[방법 A: 단독 추론 — 12GB 노트북 OK]
standalone_inference_script.py
    → Gr00tPolicy (gr00t/model/policy.py) 로드
    → 데이터셋에서 observation 읽기
    → policy.get_action(obs) → action 출력

[방법 B: 서버-클라이언트 추론 — 24GB+ 필요 (시뮬레이션 포함 시)]
Terminal 1: run_gr00t_server.py
    → Gr00tPolicy 로드
    → ZeroMQ 서버 대기 (port 5555)

Terminal 2: rollout_policy.py (또는 커스텀 클라이언트)
    → PolicyClient(host, port) 연결
    → env.reset() → obs 전송 → action 수신 → env.step(action) 반복
```

### 2.2 Fine-tuning 파이프라인

```
[데이터 준비]
LeRobot v2 포맷 데이터셋 준비
    → meta/modality.json 작성 (★ 여기서 로봇 센서/액션 정의)
    → meta/info.json, episodes.jsonl, tasks.jsonl

[학습]
launch_finetune.py (엔트리포인트)
    → finetune_config.py에서 하이퍼파라미터 로드
    → --base-model-path nvidia/GR00T-N1.6-3B (사전학습 모델)
    → --dataset-path (LeRobot 포맷 데이터)
    → --embodiment-tag NEW_EMBODIMENT (커스텀 로봇)
    → --modality-config-path (★ 로봇별 modality 설정 파일)
    → checkpoint 저장 → output-dir/checkpoint-XXXX

[평가]
open_loop_eval.py (오프라인 — 빠른 검증)
    → 예측 action vs GT action 비교, MSE 시각화
    → /tmp/open_loop_eval/traj_X.jpeg 저장

run_gr00t_server.py + rollout_policy.py (시뮬레이션 — 실제 성능 검증)
    → 시뮬레이션에서 closed-loop 평가
```

---

## 3. Fine-tuning 시 수정해야 할 파일들

### 3.1 Modality Config (★ 가장 중요)

**파일 위치:** `examples/SO100/so100_config.py` (참고용) → 내 로봇용으로 새로 만들기

**역할:** 로봇의 카메라, 상태(joint), 액션을 어떻게 읽고 정규화할지 정의

**수정 포인트:**
- `video`: 카메라 이름, 해상도, 개수 — 내 로봇 카메라에 맞게
- `state`: joint position 키 이름, 차원 수 — 내 로봇 관절에 맞게
- `action`: action space 키 이름, 차원 수 — 내 로봇 제어 방식에 맞게
- `language`: task instruction 텍스트

**데이터셋 내 위치:** `<DATASET_PATH>/meta/modality.json`

```json
{
  "video": {
    "ego_view": {
      "original_key": "observation.images.ego_view"
    }
  },
  "state": {
    "single_arm": {
      "original_key": "observation.state",
      "delta_indices": [0, 1, 2, 3, 4, 5]
    }
  },
  "action": {
    "single_arm": {
      "original_key": "action",
      "delta_indices": [0, 1, 2, 3, 4, 5]
    }
  }
}
```

**카메라를 추가/변경할 때:**
- `modality.json`의 `video` 섹션에 카메라 키 추가
- 데이터셋의 `videos/` 폴더에 해당 카메라 영상 포함
- modality_config.py에서 이미지 전처리 설정 맞추기

### 3.2 Embodiment Tag

**파일:** `gr00t/data/embodiment_tags.py`

**역할:** 로봇 타입 식별자. 사전 등록된 태그:
- `GR1` — Fourier GR1 휴머노이드
- `LIBERO_PANDA` — Franka Panda
- `OXE_GOOGLE` — Google Robot
- `OXE_WIDOWX` — WidowX
- `UNITREE_G1` — Unitree G1
- `NEW_EMBODIMENT` — ★ 커스텀 로봇은 이걸 사용

**커스텀 로봇 사용 시:** `--embodiment-tag NEW_EMBODIMENT`로 실행하면 됨. 별도 등록 불필요.

### 3.3 Fine-tune 하이퍼파라미터

**파일:** `gr00t/configs/finetune_config.py` + `launch_finetune.py`의 CLI 인자

**주요 인자:**
```bash
--base-model-path nvidia/GR00T-N1.6-3B  # 기반 모델
--dataset-path <경로>                    # 데이터셋
--embodiment-tag NEW_EMBODIMENT          # 로봇 타입
--modality-config-path <경로>            # modality 설정
--max-steps 2000~20000                   # 학습 스텝 (데이터 양에 따라)
--global-batch-size 32                   # 배치 사이즈 (VRAM에 맞게 조절)
--save-steps 2000                        # 체크포인트 저장 간격
--use-wandb                              # WandB 로깅
--color-jitter-params ...                # 이미지 augmentation
--dataloader-num-workers 4               # 데이터 로딩 워커
```

**VRAM 부족 시 조절:**
```bash
--global-batch-size 8          # 배치 줄이기
--dataloader-num-workers 2     # 워커 줄이기
--num-shards-per-epoch 100     # 샤딩 사용
--shard-size 512
```

---

## 4. VLM (Vision-Language Model) 수정 포인트

### 4.1 GR00T N1.6의 VLM 구조

```
입력: [카메라 이미지] + [텍스트 명령]
        ↓
Cosmos-Reason-2B VLM (내부 NVIDIA VLM)
    - Vision Encoder: 이미지 → 토큰
    - Language Encoder: 텍스트 → 토큰
    - VLM Top 4 layers: unfrozen (fine-tune 시 업데이트됨)
        ↓
DiT (Diffusion Transformer, 32 layers)
    - cross-attention으로 VLM 임베딩 참조
    - flow matching으로 continuous action 생성
        ↓
출력: [action chunk] (state-relative, 연속값)
```

### 4.2 VLM 관련 수정이 필요한 경우

**새로운 이미지 도메인 (학습에 없던 환경)을 쓸 때:**

1. **이미지 전처리 수정**
   - 파일: modality config의 video 섹션
   - 해상도, aspect ratio, color jitter 파라미터 조절
   - `--color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08`

2. **VLM backbone fine-tune 범위 조절**
   - 기본: VLM top 4 layers만 unfreeze
   - 도메인이 매우 다르면: 더 많은 layer를 unfreeze (launch_train.py에서 설정)
   - 데이터가 적으면: LoRA 적용 고려 (현재 기본은 full parameter update)

3. **텍스트 프롬프트 (language instruction)**
   - `meta/tasks.jsonl`에서 task description 정의
   - 추론 시 `--lang_instruction "your task description"` 전달
   - VLM이 이해할 수 있는 자연어로 작성

### 4.3 VLM을 다른 모델로 교체하려면

현재 GR00T N1.6은 Cosmos-Reason-2B를 사용하며, VLM 교체는 다음 파일을 수정해야 함:
- `gr00t/model/` 하위의 모델 정의 파일
- Vision encoder, language encoder 아키텍처 변경
- DiT와의 cross-attention 차원 매칭
- **⚠️ 주의: 대규모 수정이 필요하며, 사전학습 weight를 활용할 수 없게 됨**

---

## 5. 데이터셋 준비 체크리스트 (내 로봇용)

### 5.1 필수 폴더 구조

```
my_robot_dataset/
├── meta/
│   ├── info.json           # 데이터셋 메타정보 (fps, codec, channels 등)
│   ├── episodes.jsonl      # 에피소드별 정보
│   ├── tasks.jsonl         # task description
│   └── modality.json       # ★ GR00T 전용 — 로봇 센서/액션 매핑
├── data/
│   └── chunk-000/
│       └── *.parquet       # state, action 데이터
└── videos/
    └── chunk-000/
        └── observation.images.{카메라명}/
            └── episode_XXXXXX.mp4
```

### 5.2 info.json 필수 항목

```json
{
  "fps": 30,
  "video.fps": 30,
  "video.codec": "avc1",
  "video.pix_fmt": "yuv420p",
  "video.channels": 3,       // ★ 이거 빠지면 에러남
  "video.is_depth_map": false,
  "has_audio": false
}
```

### 5.3 데이터 변환

- LeRobot v2 → GR00T: `modality.json`만 추가하면 됨
- LeRobot v3 → v2: 변환 스크립트 제공됨
- 기타 포맷: `getting_started/data_preparation.md` 참고

---

## 6. 실행 명령어 빠른 참조

### 추론 (단독, 12GB OK)
```bash
uv run python scripts/deployment/standalone_inference_script.py \
  --model-path nvidia/GR00T-N1.6-3B \
  --dataset-path demo_data/gr1.PickNPlace \
  --embodiment-tag GR1 \
  --traj-ids 0 1 2 \
  --inference-mode pytorch \
  --action-horizon 8
```

### 추론 서버 (24GB+)
```bash
# Terminal 1: 서버
uv run python gr00t/eval/run_gr00t_server.py \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path <CHECKPOINT_PATH> \
  --device cuda:0 --host 0.0.0.0 --port 5555

# Terminal 2: 클라이언트 (시뮬레이션 등)
python gr00t/eval/rollout_policy.py \
  --policy_client_host 127.0.0.1 --policy_client_port 5555 \
  --env_name <ENV> --n_episodes 10
```

### Fine-tuning (48GB+)
```bash
CUDA_VISIBLE_DEVICES=0 uv run python \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path <DATASET_PATH> \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path <CONFIG.py> \
  --num-gpus 1 \
  --output-dir <OUTPUT_DIR> \
  --max-steps 10000 \
  --global-batch-size 32
```

### Open-loop 평가
```bash
uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path <DATASET_PATH> \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path <CHECKPOINT_PATH> \
  --traj-ids 0 --action-horizon 16
```

### TensorRT 최적화 (배포용)
```bash
# 1. ONNX export
python scripts/deployment/export_onnx_n1d6.py \
  --model-path nvidia/GR00T-N1.6-3B \
  --dataset-path <DATA> --embodiment-tag GR1 --output-dir ./onnx

# 2. TensorRT 빌드
python scripts/deployment/build_tensorrt_engine.py \
  --onnx ./onnx/dit_model.onnx \
  --engine ./onnx/dit_model_bf16.trt --precision bf16

# 3. TensorRT 추론
python scripts/deployment/standalone_inference_script.py \
  --inference-mode tensorrt --trt-engine-path ./onnx/dit_model_bf16.trt ...
```

---

## 7. 핵심 수정 포인트 요약표

| 목적 | 수정 파일 | 설명 |
|------|-----------|------|
| 내 로봇 센서/액션 정의 | `meta/modality.json` + `*_config.py` | 카메라 수, joint 수, action space |
| 로봇 타입 지정 | CLI `--embodiment-tag` | 커스텀이면 `NEW_EMBODIMENT` 사용 |
| 학습 하이퍼파라미터 | `launch_finetune.py` CLI 인자 | batch size, steps, lr 등 |
| 이미지 augmentation | `--color-jitter-params` | 새 도메인 적응 시 조절 |
| task 설명 텍스트 | `meta/tasks.jsonl` | VLM이 읽는 자연어 명령 |
| VLM unfreeze 범위 | `launch_train.py` (고급) | 도메인 차이 클 때 더 많이 unfreeze |
| 추론 속도 최적화 | `export_onnx` → `build_tensorrt` | 배포 시 TensorRT 사용 |
| VRAM 절약 | batch size ↓, sharding, workers ↓ | 12GB에서는 standalone만 가능 |

---

## 8. 아키텍처 요약

```
┌─────────────────────────────────────────────────────┐
│                    GR00T N1.6                        │
│                                                     │
│  ┌──────────────┐    ┌──────────────┐              │
│  │ Camera Image  │    │ Language Cmd │              │
│  └──────┬───────┘    └──────┬───────┘              │
│         │                    │                      │
│         ▼                    ▼                      │
│  ┌──────────────────────────────────┐              │
│  │   Cosmos-Reason-2B VLM           │              │
│  │   (Vision Encoder + LM)          │              │
│  │   Top 4 layers: trainable        │  ← VLM 수정  │
│  └──────────────┬───────────────────┘              │
│                 │ VLM embeddings                    │
│                 ▼                                   │
│  ┌──────────────────────────────────┐              │
│  │   DiT (Diffusion Transformer)    │              │
│  │   32 layers                      │              │
│  │   - self-attn (state, action)    │              │
│  │   - cross-attn (VLM features)    │              │
│  │   - flow matching denoising      │  ← Action    │
│  └──────────────┬───────────────────┘    수정      │
│                 │                                   │
│                 ▼                                   │
│  ┌──────────────────────────────────┐              │
│  │   Action Chunk Output            │              │
│  │   (state-relative, continuous)   │              │
│  │   horizon = 8~16 steps           │  ← Modality  │
│  └──────────────────────────────────┘    수정      │
└─────────────────────────────────────────────────────┘
```
