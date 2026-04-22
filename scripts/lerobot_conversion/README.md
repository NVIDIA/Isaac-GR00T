# Converting from LeRobot v3 to v2

## Setup

### 1. Create and Activate Virtual Environment
```bash
uv venv
source .venv/bin/activate
uv pip install -e . --verbose
```

### 2. Run Conversion Script

Inside the uv environment, run:
```bash
python convert_v3_to_v2.py --repo-id BobShan/double_folding_towel_v3.0
```

> **Note:** You may need to install lerobot with `GIT_LFS_SKIP_SMUDGE=1`:
> 
> ```bash
> GIT_LFS_SKIP_SMUDGE=1 uv pip install "lerobot @ git+https://github.com/huggingface/lerobot.git@c75455a6de5c818fa1bb69fb2d92423e86c70475"
> ```

## Validate or Repair v2 Metadata

Some public datasets are mostly v2-compatible but miss metadata fields expected by GR00T (for example, `meta/episodes.jsonl` or `chunk_index` values in episode metadata). Use this helper before training/evaluation:

```bash
python validate_v2_dataset.py --dataset-root /path/to/lerobot_v2_dataset
```

To write automatic metadata fixes in-place:

```bash
python validate_v2_dataset.py --dataset-root /path/to/lerobot_v2_dataset --write-fixes
```

What it currently fixes:
- Reconstruct `meta/episodes.jsonl` from `data/chunk-*/episode_*.parquet` when missing
- Ensure each episode has `tasks`, `length`, and `chunk_index`
