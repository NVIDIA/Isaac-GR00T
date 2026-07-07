# Converting from LeRobot v3 to v2

## Setup

### 1. Create and Activate Virtual Environment

Run these from the `scripts/lerobot_conversion` directory (it has its own
`pyproject.toml`; installing from the repo root would install the `gr00t`
package instead):
```bash
cd scripts/lerobot_conversion
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
