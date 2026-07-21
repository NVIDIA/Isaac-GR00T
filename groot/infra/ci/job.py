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
"""GitLab CI job utilities."""

from datetime import datetime
import enum

from groot.infra.ci import log

logger = log.init(__name__)


class DurationType(str, enum.Enum):
    """Duration type enumeration for job duration metrics."""

    QUEUE = "queue"
    EXECUTION = "execution"
    TOTAL = "total"


class JobStatus(enum.IntEnum):
    """Job status enumeration matching GitLab job status values.

    Reference: https://docs.gitlab.com/ci/jobs/#available-job-statuses
    """

    SUCCESS = 0
    FAILED = 1
    CANCELED = 2
    CANCELING = 3
    SKIPPED = 4
    MANUAL = 5
    CREATED = 6
    PENDING = 7
    RUNNING = 8
    PREPARING = 9
    SCHEDULED = 10
    WAITING_FOR_CALLBACK = 11
    WAITING_FOR_RESOURCE = 12
    UNKNOWN = 13


# Mapping from GitLab status strings to JobStatus enum
# Reference: https://docs.gitlab.com/ci/jobs/#available-job-statuses
GITLAB_STATUS_MAP = {
    "success": JobStatus.SUCCESS,
    "failed": JobStatus.FAILED,
    "canceled": JobStatus.CANCELED,
    "canceling": JobStatus.CANCELING,
    "skipped": JobStatus.SKIPPED,
    "manual": JobStatus.MANUAL,
    "created": JobStatus.CREATED,
    "pending": JobStatus.PENDING,
    "running": JobStatus.RUNNING,
    "preparing": JobStatus.PREPARING,
    "scheduled": JobStatus.SCHEDULED,
    "waiting_for_callback": JobStatus.WAITING_FOR_CALLBACK,
    "waiting_for_resource": JobStatus.WAITING_FOR_RESOURCE,
}


class Job:
    """GitLab CI job information and utilities.

    Args:
        job_data: Job dictionary from GitLab API.
    """

    def __init__(self, job_data: dict):
        self.id = str(job_data.get("id", ""))
        self.name = job_data.get("name", "unknown")
        self.status = job_data.get("status", "unknown")
        self.stage = job_data.get("stage", "unknown")
        self.web_url = job_data.get("web_url", "")
        self.created_at = job_data.get("created_at")
        self.started_at = job_data.get("started_at")
        self.finished_at = job_data.get("finished_at")
        self.allow_failure = job_data.get("allow_failure", False)
        self.retried = job_data.get("retried", False)
        self.failure_reason = job_data.get("failure_reason")

    def get_status(self) -> JobStatus:
        """Get the job status as JobStatus enum.

        Returns:
            JobStatus enum value
        """
        return GITLAB_STATUS_MAP.get(self.status.lower(), JobStatus.UNKNOWN)

    def get_execution_duration(self) -> int:
        """Get the job execution duration in seconds (started_at to finished_at).

        Returns:
            Job execution duration in seconds, or -1 if not available
        """
        if self.started_at and self.finished_at:
            try:
                start = datetime.fromisoformat(self.started_at)
                finish = datetime.fromisoformat(self.finished_at)
                return int((finish - start).total_seconds())
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to calculate duration for job {self.name}: {e}")
                return -1

        return -1

    def get_queued_duration(self) -> int:
        """Get the job queued duration in seconds (created_at to started_at).

        Returns:
            Queued duration in seconds, or -1 if not available
        """
        if self.created_at and self.started_at:
            try:
                created = datetime.fromisoformat(self.created_at)
                started = datetime.fromisoformat(self.started_at)
                return int((started - created).total_seconds())
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to calculate queued duration for job {self.name}: {e}")
                return -1

        return -1

    def get_total_duration(self) -> int:
        """Get the total job duration in seconds (created_at to finished_at).

        Returns:
            Total duration in seconds, or -1 if not available
        """
        if self.created_at and self.finished_at:
            try:
                created = datetime.fromisoformat(self.created_at)
                finish = datetime.fromisoformat(self.finished_at)
                return int((finish - created).total_seconds())
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to calculate total duration for job {self.name}: {e}")
                return -1

        return -1

    def __repr__(self) -> str:
        """String representation of the job."""
        return f"Job(name={self.name}, status={self.status}, duration={self.get_execution_duration()}s)"

    @classmethod
    def init_from_dict(cls, data: dict) -> "Job":
        return cls(data)
