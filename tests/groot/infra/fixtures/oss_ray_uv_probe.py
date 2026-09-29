"""Exercise OSS Ray entrypoints from a driver launched by ``uv run --no-sync``."""

from __future__ import annotations

import importlib.machinery
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


def _stub_unavailable_optional_dependencies() -> None:
    for name in ("av", "cv2", "torchvision"):
        try:
            __import__(name)
        except (ImportError, RuntimeError):
            module = ModuleType(name)
            module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            sys.modules[name] = module


_stub_unavailable_optional_dependencies()

from groot.infra import oss  # noqa: E402
from groot.infra.closed_loop import cli as closed_loop_cli  # noqa: E402


def _run_ray_worker(endpoint: str) -> int:
    import ray

    ray.init(include_dashboard=False, num_cpus=1)

    @ray.remote
    def worker(env_key: str) -> tuple[str, str | None]:
        import ray as worker_ray

        return worker_ray.__version__, os.environ.get(env_key)

    try:
        version, setting = ray.get(
            worker.remote(oss.OSS_RAY_UV_RUN_RUNTIME_ENV),
            timeout=30,
        )
        assert setting == "0", setting
        print(f"endpoint={endpoint} worker_ray={version} setting={setting}")  # noqa: T201
    finally:
        ray.shutdown()
    return 0


def _run_closed_loop() -> int:
    closed_loop_cli.run = lambda _config, *, dry_run: (
        int(dry_run) or _run_ray_worker("closed-loop-eval")
    )
    return oss.main(
        [
            "closed-loop-eval",
            "--checkpoint",
            "checkpoint",
            "--backend",
            "probe:Backend",
            "--suite",
            "suite",
            "--embodiment-tag",
            "probe",
        ]
    )


def _run_closed_loop_benchmark() -> int:
    checkpoint = Path("checkpoint")

    class Benchmark:
        spec = SimpleNamespace(evaluation={"closed_loop": {}})
        stage = ""

        def record_resolution(self, _resolved: object) -> None:
            pass

        def record_training(self, _resolved: object) -> None:
            pass

        def evaluate(self, evaluated_checkpoint: Path) -> int:
            assert evaluated_checkpoint == checkpoint
            return _run_ray_worker("train-eval-closed-loop-benchmark")

    def run_benchmark(args: object, *, run_train_eval) -> int:
        return run_train_eval(args, benchmark=Benchmark())

    oss.run_benchmark = run_benchmark
    oss._prepare_train_dataset = lambda *_args, **_kwargs: None  # noqa: SLF001
    oss.resolve_train = lambda _args: SimpleNamespace(final_checkpoint=checkpoint)
    oss._run_train = lambda *_args, **_kwargs: 0  # noqa: SLF001
    return oss.main(
        [
            "train-eval",
            "--benchmark-recipe",
            "probe.json",
            "--output-dir",
            "output",
        ]
    )


if __name__ == "__main__":
    endpoint = sys.argv[1]
    if endpoint == "closed-loop-eval":
        raise SystemExit(_run_closed_loop())
    if endpoint == "train-eval-closed-loop-benchmark":
        raise SystemExit(_run_closed_loop_benchmark())
    raise ValueError(f"unknown endpoint {endpoint!r}")
