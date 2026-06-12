# CLAUDE.md — Isaac GR00T N1.7

## Project overview

Isaac GR00T N1.7 is an open vision-language-action (VLA) model for generalized humanoid robot skills.
The repo contains the model, training pipeline, evaluation harness, and deployment tooling.

- **Language:** Python 3.12–3.14 (`requires-python = ">=3.12,<3.15"`)
- **Package manager:** [uv](https://docs.astral.sh/uv/)
- **Build system:** setuptools (see `pyproject.toml`)
- **CI:** internal GitLab CI (`.gitlab-ci.yml` + includes under `ci/`, not shipped to the public GitHub EA repo); public GitHub Actions (`.github/workflows/`)

## Quick-start commands

```bash
# Install (Linux; regenerates lock with: uv lock --prerelease=allow)
uv sync --python 3.12

# Lint and format (uses ruff via pre-commit)
pre-commit run --all-files

# Run CPU tests
python -m pytest tests/ -m "not gpu" -v --timeout=300

# Run GPU tests
python -m pytest tests/ -m gpu -v --timeout=300

# Build package
uv build

# Validate lockfile (on Linux)
uv lock --locked --prerelease=allow
```

## Code style

- Formatter: `ruff format` (double quotes, spaces, line-length 100)
- Linter: `ruff check` with rules E, F, I (ignores E501)
- Config lives in `pyproject.toml` under `[tool.ruff]`
- Run `pre-commit run --all-files` before committing

## Directory layout

```
gr00t/              # Main package
  configs/          #   Training, data, and model configs
  data/             #   Data loading, embodiment tags, dataset processing
  eval/             #   Evaluation (run_gr00t_server.py)
  experiment/       #   Training pipeline (launch_finetune.py, trainer.py)
  model/            #   Model architecture (N1.7, base, modules)
  policy/           #   Policy inference (Gr00tPolicy, server/client)
examples/           # Per-embodiment example configs and READMEs
scripts/            # Deployment, conversion, and utility scripts
  deployment/       #   Platform install scripts (dgpu, orin, thor, spark)
tests/              # pytest suite (markers: gpu, not gpu)
getting_started/    # User-facing guides and notebooks
```

## Key entry points

- **Fine-tune:** `bash examples/finetune.sh --base-model-path <path> --dataset-path <path> --embodiment-tag <tag> --output-dir <dir>`
- **Inference server:** `python gr00t/eval/run_gr00t_server.py --model-path <path> --embodiment-tag <tag>`
- **ONNX export:** `python scripts/deployment/export_onnx_n1d7.py`
- **TensorRT build:** `python scripts/deployment/build_trt_pipeline.py`
- **Benchmark:** `python scripts/deployment/benchmark_inference.py`

## Testing

- Test markers: `gpu` (requires GPU), default is CPU-safe
- Fixtures live in `tests/fixtures/` and `demo_data/`
- CI runs CPU and GPU tests in separate jobs with 300s timeout

## Deployment platforms

- **dGPU (H100, A100, RTX, GB200):** CUDA 13.0 — `bash scripts/deployment/dgpu/install_deps.sh`, container via top-level `docker/Dockerfile`
- **Jetson Orin:** CUDA 13.0 (JetPack 7.2+) — `scripts/deployment/orin/install_deps.sh`, `scripts/activate_orin.sh`
- **Jetson Thor:** CUDA 13.0 — `scripts/deployment/thor/install_deps.sh`, `scripts/activate_thor.sh`
- **DGX Spark:** CUDA 13.0 — `scripts/deployment/spark/install_deps.sh`, `scripts/activate_spark.sh`

LeRobot v0.5.1 is used only in `scripts/lerobot_conversion/` and `gr00t/eval/real_robot/SO100/` (not the core `gr00t` package).
