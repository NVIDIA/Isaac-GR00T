# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for OSS post-training open-loop eval orchestration."""

from __future__ import annotations

from pathlib import Path
import sys
import types
from typing import Any

import pytest

from groot.infra.eval import oss_open_loop_evaluator as evaluator_mod
from groot.infra.eval import policy_server_process as server_mod
from groot.infra.eval.hf_model_uri import HfModelUri
from groot.infra.eval.metric_summary import MetricSummary
from groot.infra.eval.oss_open_loop_eval_config import OssOpenLoopEvalConfig
from groot.infra.eval.oss_open_loop_evaluator import OssOpenLoopEvaluator
from groot.infra.eval.policy_client_adapter import PolicyClientAdapter
from groot.infra.eval.policy_server_process import PolicyServerProcess


def test_parse_hf_model_uri() -> None:
    parsed = HfModelUri.init_from_uri(
        "hf://model/nvidia/GR00T-N2-MinWAM-Stack-Bowls-PostTrain@main/checkpoint-100"
    )

    assert parsed.repo_id == "nvidia/GR00T-N2-MinWAM-Stack-Bowls-PostTrain"
    assert parsed.revision == "main"
    assert parsed.checkpoint_name == "checkpoint-100"


def test_parse_hf_model_uri_rejects_missing_checkpoint() -> None:
    with pytest.raises(ValueError, match="checkpoint path"):
        HfModelUri.init_from_uri("hf://model/org/model@main")


def test_resolve_artifacts_from_golden_run_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOLDEN_RUN_FINAL_MODEL_REPO_ID", "org/final")
    monkeypatch.setenv("GOLDEN_RUN_FINAL_MODEL_REVISION", "main")
    monkeypatch.setenv("GOLDEN_RUN_DATASET_REPO_ID", "org/data")
    monkeypatch.setenv("GOLDEN_RUN_DATASET_REVISION", "release/test")
    monkeypatch.setenv("GOLDEN_RUN_DATASET_LOCAL_NAME", "dataset_local")

    calls: list[tuple[str, str, str, str]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(
            (
                kwargs["repo_id"],
                kwargs["repo_type"],
                kwargs["revision"],
                kwargs["local_dir"],
            )
        )
        Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)
        return kwargs["local_dir"]

    evaluator = OssOpenLoopEvaluator(
        OssOpenLoopEvalConfig(cache_root=str(tmp_path)),
        snapshot_download_fn=fake_snapshot_download,
    )
    artifacts = evaluator.resolve_artifacts()

    assert artifacts.model_path == tmp_path / "models" / "org--final" / "main" / "checkpoint-100"
    assert artifacts.dataset_path == (
        tmp_path / "datasets" / "org--data" / "release--test" / "dataset_local"
    )
    assert calls == [
        ("org/final", "model", "main", str(tmp_path / "models" / "org--final" / "main")),
        (
            "org/data",
            "dataset",
            "release/test",
            str(tmp_path / "datasets" / "org--data" / "release--test" / "dataset_local"),
        ),
    ]


def test_model_uri_overrides_default_checkpoint(tmp_path: Path) -> None:
    def fake_snapshot_download(**kwargs: Any) -> str:
        Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)
        return kwargs["local_dir"]

    evaluator = OssOpenLoopEvaluator(
        OssOpenLoopEvalConfig(
            model_uri="hf://model/org/final@dev/checkpoint-7",
            cache_root=str(tmp_path),
        ),
        snapshot_download_fn=fake_snapshot_download,
    )
    path = evaluator.resolve_model_path()

    assert path == tmp_path / "models" / "org--final" / "dev" / "checkpoint-7"


def test_resolve_model_path_falls_back_to_loadable_checkpoint(tmp_path: Path) -> None:
    def fake_snapshot_download(**kwargs: Any) -> str:
        root = Path(kwargs["local_dir"])
        (root / "checkpoint-100" / "dcp_state").mkdir(parents=True)
        (root / "checkpoint-100" / "dcp_state" / ".metadata").touch()
        (root / "checkpoint-10").mkdir()
        (root / "checkpoint-10" / "model.safetensors.index.json").write_text(
            "{}", encoding="utf-8"
        )
        return kwargs["local_dir"]

    evaluator = OssOpenLoopEvaluator(
        OssOpenLoopEvalConfig(
            model_uri="hf://model/org/final@main/checkpoint-100",
            cache_root=str(tmp_path),
        ),
        snapshot_download_fn=fake_snapshot_download,
    )
    path = evaluator.resolve_model_path()

    assert path == tmp_path / "models" / "org--final" / "main" / "checkpoint-10"


def test_validate_results_rejects_nonfinite_metric() -> None:
    summary = MetricSummary(
        per_trajectory=[{"traj_id": 0, "mse": float("nan")}],
        average={"mse": float("nan")},
    )

    with pytest.raises(RuntimeError, match="not finite"):
        OssOpenLoopEvaluator(OssOpenLoopEvalConfig()).validate_results({"local": summary})


def test_validate_results_rejects_server_parity_drift() -> None:
    local = MetricSummary(per_trajectory=[], average={"mse": 0.01, "mae": 0.02})
    server = MetricSummary(per_trajectory=[], average={"mse": 0.03, "mae": 0.02})

    with pytest.raises(RuntimeError, match="local/server mse delta"):
        OssOpenLoopEvaluator(
            OssOpenLoopEvalConfig(server_parity_tolerance=0.01),
        ).validate_results(
            {"local": local, "server": server}
        )


def test_validate_results_accepts_matching_modes() -> None:
    local = MetricSummary(per_trajectory=[], average={"mse": 0.01, "mae": 0.02})
    server = MetricSummary(per_trajectory=[], average={"mse": 0.011, "mae": 0.021})

    OssOpenLoopEvaluator(
        OssOpenLoopEvalConfig(max_mse=0.02, server_parity_tolerance=0.01),
    ).validate_results({"local": local, "server": server})


def test_modality_config_from_policy_rebuilds_wire_dicts(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModalityConfig:
        def __init__(self, delta_indices: list[int], modality_keys: list[str]):
            self.delta_indices = delta_indices
            self.modality_keys = modality_keys

    class FakePolicy:
        def get_modality_config(self) -> dict[str, dict[str, list[int] | list[str]]]:
            return {
                "language": {
                    "delta_indices": [0],
                    "modality_keys": ["annotation.task"],
                }
            }

    fake_types = types.ModuleType("groot.core.data.types")
    fake_types.ModalityConfig = FakeModalityConfig
    monkeypatch.setitem(sys.modules, "groot.core.data.types", fake_types)

    modality = OssOpenLoopEvaluator(OssOpenLoopEvalConfig()).modality_config_from_policy(
        FakePolicy()
    )

    assert isinstance(modality["language"], FakeModalityConfig)
    assert modality["language"].modality_keys == ["annotation.task"]


def test_policy_client_adapter_translates_nested_observation_and_flat_action() -> None:
    events: list[dict[str, Any]] = []

    class FakeActionArray:
        ndim = 2

        def __getitem__(self, item: Any) -> tuple[str, Any]:
            return ("batched", item)

    class FakeClient:
        def get_action(
            self,
            observation: dict[str, Any],
            options: dict[str, Any] | None = None,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            events.append(observation)
            return {"action.left": FakeActionArray()}, {"ok": True}

    adapter = PolicyClientAdapter(FakeClient())

    action, info = adapter.get_action(
        {
            "video": {"cam": "video-value"},
            "state": {"joint": "state-value"},
            "language": {"annotation.task": "task-value"},
        }
    )

    assert events == [
        {
            "video.cam": "video-value",
            "state.joint": "state-value",
            "annotation.task": "task-value",
        }
    ]
    assert action["left"] == ("batched", (None, Ellipsis))
    assert info == {"ok": True}


def test_close_policy_and_release_device_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class FakePolicy:
        def close(self) -> None:
            events.append("close")

    class FakeCuda:
        def is_available(self) -> bool:
            return True

        def empty_cache(self) -> None:
            events.append("empty_cache")

    def fake_import_module(name: str) -> object:
        assert name == "torch"
        return types.SimpleNamespace(cuda=FakeCuda())

    monkeypatch.setattr(evaluator_mod.importlib, "import_module", fake_import_module)

    evaluator = OssOpenLoopEvaluator(OssOpenLoopEvalConfig())
    evaluator.close_policy(FakePolicy())
    evaluator.release_device_cache()

    assert events == ["close", "empty_cache"]


def test_policy_server_process_starts_and_stops_with_client_ping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeProc:
        def __init__(self, cmd, **kwargs):
            self.cmd = cmd
            self.kwargs = kwargs
            self.returncode = None
            events.append("popen")

        def poll(self):
            return self.returncode

        def terminate(self):
            events.append("terminate")
            self.returncode = -15

        def kill(self):
            events.append("kill")
            self.returncode = -9

        def wait(self, timeout=None):
            events.append("wait")
            self.returncode = 0
            return 0

    class FakePolicyClient:
        def __init__(self, address: str, timeout_ms: int, **_kwargs):
            self.address = address
            self.timeout_ms = timeout_ms

        def ping(self) -> bool:
            events.append(f"ping:{self.address}")
            return True

        def kill_server(self) -> None:
            events.append("kill_server")

        def close(self) -> None:
            events.append("close")

    fake_server_client = types.ModuleType("groot.core.policy.server_client")
    fake_server_client.PolicyClient = FakePolicyClient
    monkeypatch.setitem(sys.modules, "groot.core.policy.server_client", fake_server_client)
    monkeypatch.setattr(server_mod.subprocess, "Popen", FakeProc)

    config = OssOpenLoopEvalConfig(
        device="cpu",
        embodiment_tag="xdof_relative_eef_relative_joint",
        server_port=6000,
        server_log_path=str(tmp_path / "server.log"),
    )
    server = PolicyServerProcess(tmp_path / "checkpoint-100", config, device="cpu")

    server.start()
    assert server.proc is not None
    assert "--model-path" in server.proc.cmd
    assert str(tmp_path / "checkpoint-100") in server.proc.cmd
    assert "--embodiment-tag" in server.proc.cmd
    assert "XDOF_RELATIVE_EEF_RELATIVE_JOINT" in server.proc.cmd
    assert "--port" in server.proc.cmd
    assert "6000" in server.proc.cmd

    server.stop()

    assert events == [
        "popen",
        "ping:tcp://127.0.0.1:6000",
        "close",
        "kill_server",
        "close",
        "wait",
    ]


def test_policy_server_readiness_failure_includes_log_tail(tmp_path: Path) -> None:
    log_path = tmp_path / "server.log"
    log_path.write_text("first line\nlast diagnostic\n", encoding="utf-8")
    config = OssOpenLoopEvalConfig(server_log_path=str(log_path))
    server = PolicyServerProcess(tmp_path / "checkpoint-100", config, device="cpu")

    message = server.readiness_failure_message("Policy server exited before becoming ready")

    assert str(log_path) in message
    assert "Policy server log tail:" in message
    assert "last diagnostic" in message
