"""Real-worker regression coverage for Ray launched through uv-backed OSS commands."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

import pytest

from groot.infra import oss


@pytest.mark.parametrize(
    "endpoint",
    ["closed-loop-eval", "train-eval-closed-loop-benchmark"],
)
def test_oss_local_ray_worker_reuses_uv_driver_environment(
    endpoint: str,
    tmp_path: Path,
) -> None:
    """A packaged Ray worker must not replay uv into a new empty environment."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv executable is unavailable")

    project = tmp_path / "uv-project"
    project.mkdir()
    environment = project / ".venv"
    current_environment = Path(sys.prefix)
    if (
        current_environment != Path(sys.base_prefix)
        and (current_environment / "pyvenv.cfg").is_file()
    ):
        environment.symlink_to(current_environment, target_is_directory=True)
    else:
        venv.EnvBuilder(with_pip=False, system_site_packages=True).create(environment)

    (project / "pyproject.toml").write_text(
        '[project]\nname = "groot-ray-uv-probe"\nversion = "0.0.0"\n',
        encoding="utf-8",
    )
    probe = Path(__file__).with_name("fixtures") / "oss_ray_uv_probe.py"
    shutil.copyfile(probe, project / "probe.py")

    env = dict(os.environ)
    env.pop("RAY_ADDRESS", None)
    env.pop("VIRTUAL_ENV", None)
    env[oss.OSS_RAY_UV_RUN_RUNTIME_ENV] = "1"
    env["PYTHONPATH"] = str(Path(oss.__file__).resolve().parents[2])
    env["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")
    completed = subprocess.run(
        [uv, "run", "--no-sync", "python", "probe.py", endpoint],
        cwd=project,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
    assert f"endpoint={endpoint} worker_ray=" in completed.stdout
    assert "setting=0" in completed.stdout
