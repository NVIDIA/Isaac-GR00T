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
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GitLab CI pipeline utilities."""

import datetime as dt
from datetime import datetime
import enum
import os

import requests

from groot.infra.ci import job as ci_job
from groot.infra.ci import log

logger = log.init(__name__)

# Stages that count toward pipeline status and wall-clock duration
# for a regular (non-golden-run) pipeline.
_REGULAR_PIPELINE_STAGES = {"premerge", "postmerge"}

# Stages that count for a golden run pipeline.
_GOLDEN_RUN_STAGES = {"staging"}


class PipelineStatus(enum.IntEnum):
    """Pipeline status enumeration."""

    SUCCESS = 0
    FAILED = 1
    UNKNOWN = 2


class Pipeline:
    """GitLab CI pipeline information and utilities.

    Initialize pipeline with GitLab CI environment variables.
    """

    def __init__(self):
        self.ci_api_v4_url = os.getenv("CI_API_V4_URL")
        self.ci_project_id = os.getenv("CI_PROJECT_ID")
        self.ci_pipeline_id = os.getenv("CI_PIPELINE_ID")
        self.ci_pipeline_source = os.getenv("CI_PIPELINE_SOURCE", "unknown")
        self.ci_job_id = os.getenv("CI_JOB_ID")
        self.ci_commit_branch = os.getenv("CI_COMMIT_BRANCH")
        self.ci_default_branch = os.getenv("CI_DEFAULT_BRANCH")
        self.ci_is_golden_run = any(
            os.getenv(name) == "true"
            for name in (
                "GOLDEN_RUN_SCHEDULED",
                "GOLDEN_RUN_E2E_LEAF",
            )
        )
        self.ci_pipeline_osmo_workflow_priority = os.getenv("OSMO_PRIORITY", "normal")
        self._auth_token = os.getenv("GITLAB_TOKEN")
        self._jobs_cache = None

    @classmethod
    def init_from_dict(cls, data: dict) -> "Pipeline":
        instance = cls()
        instance.id = str(data.get("id", ""))
        instance.status = data.get("status", "unknown")
        instance.ref = data.get("ref", "")
        instance.sha = data.get("sha", "")
        instance.web_url = data.get("web_url", "")
        instance.coverage = data.get("coverage")
        return instance

    def is_golden_run(self) -> bool:
        """Check if the current pipeline is a golden run.

        Returns:
            True if running on a golden run, False otherwise
        """
        return self.ci_is_golden_run

    def get_pipeline_source(self) -> str:
        """Get the source of the current pipeline.

        Returns:
            The source of the current pipeline
        """
        return self.ci_pipeline_source

    def get_pipeline_osmo_workflow_priority(self) -> str:
        """Get the OSMO workflow priority of the current pipeline.

        Returns:
            The priority of the current pipeline
        """
        return self.ci_pipeline_osmo_workflow_priority

    def is_main_branch(self) -> bool:
        """Check if the current pipeline is running on the main branch.

        Returns:
            True if running on main branch, False otherwise
        """
        return bool(self.ci_commit_branch) and self.ci_commit_branch == self.ci_default_branch

    def _get_pipeline_phase_jobs(self) -> list[ci_job.Job]:
        """Return jobs for the relevant pipeline phase.

        - Golden run pipelines: staging stage only
        - Regular pipelines: premerge + postmerge stages only
        """
        stages = _GOLDEN_RUN_STAGES if self.is_golden_run() else _REGULAR_PIPELINE_STAGES
        return [j for j in self.get_jobs() if j.stage in stages]

    def get_jobs(self) -> list[ci_job.Job]:
        """Get all jobs in the pipeline (memoized).

        Returns:
            List of Job objects from GitLab API, or empty list if unable to fetch
        """
        if self._jobs_cache is not None:
            return self._jobs_cache

        if not all([self.ci_api_v4_url, self.ci_project_id, self.ci_pipeline_id, self._auth_token]):
            logger.warning("Missing GitLab CI variables, cannot fetch pipeline jobs")
            self._jobs_cache = []
            return self._jobs_cache

        try:
            url = f"{self.ci_api_v4_url}/projects/{self.ci_project_id}/pipelines/{self.ci_pipeline_id}/jobs"
            headers = {"PRIVATE-TOKEN": self._auth_token}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            jobs_data = response.json()
            self._jobs_cache = [ci_job.Job(job_data) for job_data in jobs_data]
            logger.debug(f"Fetched {len(self._jobs_cache)} jobs from pipeline")
            return self._jobs_cache

        except Exception:
            # Do not cache the error to account for transient errors
            logger.exception("Failed to fetch pipeline jobs")
            return []

    def get_duration(self) -> int:
        """Get the pipeline duration in seconds by finding the earliest job creation time.

        Returns:
            Pipeline duration in seconds, or -1 if unable to determine
        """
        jobs = self._get_pipeline_phase_jobs()
        if not jobs:
            logger.warning("No jobs found in pipeline")
            return -1

        # Find the earliest job creation time
        earliest_start = None
        for job in jobs:
            if job.created_at:
                # Parse ISO 8601 timestamp: "2026-01-18T22:23:10.188-08:00"
                job_start = datetime.fromisoformat(job.created_at)
                if earliest_start is None or job_start < earliest_start:
                    earliest_start = job_start

        if earliest_start is None:
            logger.warning("No job creation timestamps found")
            return -1

        # Calculate duration from earliest job start to now
        duration = int((datetime.now(dt.timezone.utc) - earliest_start).total_seconds())
        logger.debug(f"Pipeline started at {earliest_start}, duration: {duration}s")
        return duration

    def get_status(self) -> PipelineStatus:
        """Get the overall pipeline status by checking job statuses via GitLab API.

        Returns:
            PipelineStatus.SUCCESS if all jobs succeeded,
            PipelineStatus.FAILED if any job failed,
            PipelineStatus.UNKNOWN on error
        """
        jobs = self._get_pipeline_phase_jobs()
        if not jobs:
            logger.warning("Cannot determine pipeline status")
            return PipelineStatus.UNKNOWN

        # Check if any job failed (excluding the current .post job)
        for job in jobs:
            if job.id == self.ci_job_id:
                continue  # Skip the current observability job

            logger.debug(f"Job {job.name}: {job.status}")

            if job.status == "failed" and not job.allow_failure:
                logger.info(f"Pipeline has failed job: {job.name}")
                return PipelineStatus.FAILED
            elif job.status == "failed" and job.allow_failure:
                logger.info(
                    f"Pipeline has allowed-to-fail job: {job.name} (not counting as failure)"
                )

        logger.info("All pipeline jobs succeeded")
        return PipelineStatus.SUCCESS
