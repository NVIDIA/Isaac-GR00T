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

from __future__ import annotations

import asyncio
import datetime as dt
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any

from botocore.exceptions import ClientError

from groot.infra.ci import log
from groot.infra.ci.db import Database
from groot.infra.ci.main.osmo_cache import blocks_to_yaml, parse_yaml_blocks

logger = log.init(__name__)

_PREFIX = "rws/ci/postmerge_test_rate_limit/v1/"


class PostmergeTestRateLimit:
    """Per-workflow postmerge result rate limit backed by the PDX S3 bucket.

    The nightly high-priority postmerge pipeline uses this to skip workflows that
    already produced a result in the last 24 hours, usually from the normal
    low-priority main-branch postmerge run.
    """

    def __init__(self, window_hours: int = 24, db: Database | None = None) -> None:
        self._window_hours = window_hours
        self._db = db if db is not None else Database()

    def _slug(self, workflow_name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", workflow_name.lower()).strip("_")

    def _key(self, workflow_name: str) -> str:
        return f"{_PREFIX}{self._slug(workflow_name)}.json"

    def has_recent_result(self, workflow_name: str, now: datetime | None = None) -> bool:
        key = self._key(workflow_name)
        try:
            raw = asyncio.run(self._db.async_read(key))
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in ("NoSuchKey", "404", "NoSuchBucket"):
                return False
            raise

        data = json.loads(raw)
        last = datetime.fromisoformat(data["recorded_at"])
        now = now if now is not None else datetime.now(dt.timezone.utc)
        recent = now - last < dt.timedelta(hours=self._window_hours)
        if recent:
            logger.info(
                "[postmerge-rate-limit] %s has a result at %s (< %dh); skipping",
                workflow_name,
                last.isoformat(),
                self._window_hours,
            )
        return recent

    def filter_yaml(
        self,
        input_path: Path,
        output_path: Path,
        now: datetime | None = None,
    ) -> list[str]:
        """Write only workflows without a recent result and return their names."""
        text = input_path.read_text()
        env_block, blocks = parse_yaml_blocks(text)
        due = [
            workflow_name
            for workflow_name in blocks
            if not self.has_recent_result(workflow_name, now=now)
        ]
        output_path.write_text(blocks_to_yaml(env_block, blocks, due))
        return due

    def record_workflow_result(
        self,
        workflow_name: str,
        payload: dict[str, Any],
        recorded_at: datetime | None = None,
    ) -> None:
        recorded_at = recorded_at if recorded_at is not None else datetime.now(dt.timezone.utc)
        body = {
            "workflow_name": workflow_name,
            "recorded_at": recorded_at.isoformat(),
            **payload,
        }
        asyncio.run(self._db.async_write(self._key(workflow_name), json.dumps(body, indent=2)))
        logger.info("[postmerge-rate-limit] recorded result for %s", workflow_name)

    def record_log(
        self,
        log_file: Path,
        workflow_type: str,
        commit_sha: str,
        pipeline_url: str,
        recorded_at: datetime | None = None,
    ) -> int:
        log_data = json.loads(log_file.read_text())
        workflows = log_data.get("integration", {}).get("workflows", [])
        count = 0
        for workflow in workflows:
            workflow_name = workflow.get("name")
            status = workflow.get("status")
            if not workflow_name or not status:
                continue
            self.record_workflow_result(
                workflow_name=workflow_name,
                payload={
                    "workflow_type": workflow_type,
                    "status": status,
                    "commit_sha": commit_sha,
                    "pipeline_url": pipeline_url,
                    "workflow_id": workflow.get("workflow_id", ""),
                    "start_time": workflow.get("start_time", ""),
                    "end_time": workflow.get("end_time", ""),
                },
                recorded_at=recorded_at,
            )
            count += 1
        return count
