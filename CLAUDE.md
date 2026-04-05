# CLAUDE.md — Isaac GR00T N1.7

## Project overview

Isaac GR00T N1.7 is an open vision-language-action (VLA) model for generalized humanoid robot skills.
The repo contains the model, training pipeline, evaluation harness, and deployment tooling.

- **Language:** Python 3.10
- **Package manager:** [uv](https://docs.astral.sh/uv/)
- **Build system:** hatchling (via pyproject.toml)
- **CI:** GitLab CI (`.gitlab-ci.yml` + modular includes under `ci/`)

## Quick-start commands

```bash
# Install (dev mode with all extras)
uv sync --all-extras

# Lint and format (uses ruff via pre-commit)
pre-commit run --all-files

# Run CPU tests
python -m pytest tests/ groot_infra/tests/ -m "not gpu" -v --timeout=300

# Run GPU tests
python -m pytest tests/ groot_infra/tests/ -m gpu -v --timeout=300

# Build package
uv build

# Validate lockfile
uv lock --locked

# Regenerate ATTRIBUTIONS.md
python scripts/generate_attributions.py
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
ci/                 # GitLab CI job definitions
groot_infra/        # CI utilities (separate uv workspace member)
```

## Key entry points

| Task | Command |
|------|---------|
| Fine-tune | `bash examples/finetune.sh --base-model-path <path> --dataset-path <path> --embodiment-tag <tag> --output-dir <dir>` |
| Inference server | `python gr00t/eval/run_gr00t_server.py --model-path <path> --embodiment-tag <tag>` |
| ONNX export | `python scripts/deployment/export_onnx_n1d7.py` |
| TensorRT build | `python scripts/deployment/build_trt_pipeline.py` |
| Benchmark | `python scripts/deployment/benchmark_inference.py` |

## Testing

- Test markers: `gpu` (requires GPU), default is CPU-safe
- Fixtures live in `tests/fixtures/` and `demo_data/`
- CI runs CPU and GPU tests in separate jobs with 300s timeout

## Deployment platforms

| Platform | CUDA | Install script | Dockerfile |
|----------|------|----------------|------------|
| dGPU (H100, A100, RTX) | 12.8 | `scripts/deployment/dgpu/install_deps.sh` | `scripts/deployment/dgpu/Dockerfile` |
| Jetson Orin | 12.6 | `scripts/deployment/orin/install_deps.sh` | `scripts/deployment/orin/Dockerfile` |
| Jetson Thor | 13.0 | `scripts/deployment/thor/install_deps.sh` | `scripts/deployment/thor/Dockerfile` |
| DGX Spark | 13.0 | `scripts/deployment/spark/install_deps.sh` | `scripts/deployment/spark/Dockerfile` |

Each platform has a corresponding `activate_*.sh` script to set up environment variables.
