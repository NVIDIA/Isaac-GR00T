# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Policy server subprocess lifecycle."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time
from types import TracebackType
from typing import TextIO

from typing_extensions import Self

from groot.core.data import schema
from groot.infra.eval.oss_open_loop_eval_config import OssOpenLoopEvalConfig


class PolicyServerProcess:
    def __init__(self, model_path: Path, config: OssOpenLoopEvalConfig, device: str):
        self.model_path = model_path
        self.config = config
        self.device = device
        self.proc: subprocess.Popen | None = None
        self._log_file: TextIO | None = None

    @property
    def address(self) -> str:
        return f"tcp://{self.config.server_host}:{self.config.server_port}"

    def start(self) -> None:
        Path(self.config.server_log_path).parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.config.server_log_path, "w")  # noqa: SIM115
        cmd = [
            sys.executable,
            "-m",
            "groot.core.policy.run_gr00t_server",
            "--model-path",
            str(self.model_path),
            "--embodiment-tag",
            self.server_embodiment_tag(),
            "--host",
            self.config.server_bind_host,
            "--port",
            str(self.config.server_port),
            "--device",
            self.device,
            "--num-inference-timesteps",
            str(self.config.denoising_steps),
        ]
        if self.config.trt_engine_dir is not None:
            cmd.extend(["--trt-engine-dir", self.config.trt_engine_dir])
        if self.config.use_cuda_graph:
            cmd.append("--use-cuda-graph")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            self._wait_until_ready()
        except BaseException:
            # A raise here (readiness timeout, server exit, spawn error)
            # escapes __enter__, so __exit__/stop() never runs. Reap the
            # subprocess and close the log file before re-raising so a failed
            # startup does not orphan a GPU-holding process or the log handle.
            self.stop()
            raise

    def server_embodiment_tag(self) -> str:
        try:
            return schema.EmbodimentTag[self.config.embodiment_tag].name
        except KeyError:
            return schema.EmbodimentTag(self.config.embodiment_tag).name

    def stop(self) -> None:
        from groot.core.policy.server_client import PolicyClient

        try:
            if self.proc is not None and self.proc.poll() is None:
                client = PolicyClient(address=self.address, timeout_ms=5_000)
                try:
                    client.kill_server()
                except Exception:  # noqa: BLE001
                    self.proc.terminate()
                finally:
                    client.close()
                try:
                    self.proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=30)
        finally:
            # Always release the log handle, including the Popen-failed path
            # where self.proc is still None. Idempotent so a second stop()
            # (e.g. start() cleanup then __exit__) does not double-close.
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None

    def _wait_until_ready(self) -> None:
        from groot.core.policy.server_client import PolicyClient

        started_at = time.monotonic()
        client = PolicyClient(address=self.address, timeout_ms=5_000)
        try:
            while time.monotonic() - started_at < self.config.server_start_timeout_seconds:
                if self.proc is not None and self.proc.poll() is not None:
                    raise RuntimeError(
                        self.readiness_failure_message("Policy server exited before becoming ready")
                    )
                if client.ping():
                    return
                time.sleep(2)
        finally:
            client.close()
        raise TimeoutError(
            self.readiness_failure_message(
                f"Policy server did not become ready within {self.config.server_start_timeout_seconds}s"
            )
        )

    def readiness_failure_message(self, reason: str) -> str:
        message = f"{reason}. See {self.config.server_log_path}"
        log_tail = self.server_log_tail()
        if log_tail:
            message = f"{message}\n\nPolicy server log tail:\n{log_tail}"
        return message

    def server_log_tail(self, max_chars: int = 4_000) -> str:
        if self._log_file is not None:
            self._log_file.flush()
        log_path = Path(self.config.server_log_path)
        if not log_path.exists():
            return ""
        return log_path.read_text(encoding="utf-8", errors="replace")[-max_chars:].strip()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        self.stop()
        return False
