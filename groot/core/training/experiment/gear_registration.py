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
"""GEAR Dashboard API experiment registration helper.

Called by:
  - submit.py shared submit utility (before Ray job submission) for cluster jobs
  - experiment.py run() as a fallback for local training jobs

Uses an env-var guard (GEAR_EXPERIMENT_ID / GEAR_EXPERIMENT_REF) to avoid
double-registration when a launcher has already minted the ID before the Ray
job is submitted.

On any failure the functions log a warning and return empty strings so training
is never blocked by a registration error.
"""

from __future__ import annotations

import getpass
import json
import logging
import os
from pathlib import Path
import platform
import ssl
import subprocess
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

try:
    import truststore
except ImportError:
    truststore = None

logger = logging.getLogger(__name__)

API_URL_ENV = "GEAR_DASHBOARD_API_URL"
SKIP_SSL_VERIFY_ENV = "GEAR_DASHBOARD_SKIP_SSL_VERIFY"
# Scoped CA-bundle override: takes precedence over generic SSL_CERT_FILE.
CA_BUNDLE_ENV = "GEAR_DASHBOARD_CA_BUNDLE"
TIMEOUT_ENV = "GEAR_DASHBOARD_TIMEOUT"
FORCE_OUTBOX_ENV = "GEAR_DASHBOARD_FORCE_OUTBOX"
PRESERVE_CHECKPOINT_ROOT_ENV = "GEAR_PRESERVE_CHECKPOINT_ROOT"
PRESERVE_OUTPUT_DIR_ENV = "GEAR_PRESERVE_OUTPUT_DIR"
# Production API. ``GEAR_DASHBOARD_API_URL`` overrides this -- point it at
# ``DEV_API_URL`` (or any other host) for development and integration testing
# so local runs never write to the prod experiments table by accident.
DEFAULT_API_URL = "https://gear-dashboard.nvidia.com/api"
DEV_API_URL = "https://gear-dashboard-01.nvidia.com/api"
# HTTP timeout for every dashboard call. 15s by default rather than 5 because
# Ray workers on the AWS DGXC pool reach the dashboard through a corp tunnel
# whose latency budget is in the high single digits even when healthy. Set
# GEAR_DASHBOARD_TIMEOUT to override (operator-tunable per cluster).
DEFAULT_TIMEOUT = 15.0

# macOS machines commonly trust NVIDIA internal CAs through Keychain, while
# Python's default OpenSSL/certifi store does not. Linux cluster images should
# use their system trust store or an explicit CA bundle.
if platform.system() == "Darwin" and truststore is not None:
    truststore.inject_into_ssl()


def preserve_checkpoint_root_enabled(env: dict[str, str] | None = None) -> bool:
    """Whether launcher/worker should keep user-provided checkpoint roots."""
    env = env or os.environ
    for key in (PRESERVE_CHECKPOINT_ROOT_ENV, PRESERVE_OUTPUT_DIR_ENV):
        if env.get(key, "").lower() in ("1", "true", "yes"):
            return True
    return False


def _get_api_url() -> str:
    return os.environ.get(API_URL_ENV, DEFAULT_API_URL)


def _get_ssl_context() -> ssl.SSLContext | None:
    """Build an SSL context for talking to the gear dashboard.

    Resolution order (most-specific first, so emergency overrides win):

    1. ``GEAR_DASHBOARD_SKIP_SSL_VERIFY=1`` -- disable verification entirely
       (dev / smoke tests on machines without the corp CA).
    2. ``GEAR_DASHBOARD_CA_BUNDLE`` -- scoped operator override; lets an admin
       rotate the cert without a code commit. Mirrors the
       ``YAM_EVAL_CA_BUNDLE`` pattern from the gitlab-CI yam-eval client.
    3. ``SSL_CERT_FILE`` -- generic Python ssl env. Some Ray images set this
       at the system level; honor it but rank it BELOW the scoped override.
    4. Python's system trust store.

    All branches are scoped to *this* SSL context: nothing here mutates
    process-wide trust, so boto3 / HuggingFace / other HTTPS clients on the
    same worker keep their default trust stores intact.
    """
    if os.environ.get(SKIP_SSL_VERIFY_ENV, "").lower() in ("1", "true", "yes"):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    cert_file = os.environ.get(CA_BUNDLE_ENV) or os.environ.get("SSL_CERT_FILE")
    if cert_file:
        ctx = ssl.create_default_context(cafile=cert_file)
        # The dashboard's TLS chain anchors at BGCA223-CA, an *intermediate*
        # signed by HQNVCA121-CA (the corp root). HQNVCA121 uses RSA-PSS
        # signatures which Python's strict chain builder rejects on some
        # OpenSSL builds even when the root is in the bundle. PARTIAL_CHAIN
        # tells OpenSSL that any cert in the bundle is a valid trust anchor,
        # so the leaf -> BGCA223 chain succeeds without traversing the
        # PSS-signed root. Curl, Java, and macOS Keychain all accept this
        # chain by default for the same reason.
        ctx.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN
        return ctx
    return None


def _get_timeout() -> float:
    """Resolve the per-call HTTP timeout, honoring GEAR_DASHBOARD_TIMEOUT."""
    raw = os.environ.get(TIMEOUT_ENV, "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            logger.warning(
                "%s=%r is not a number; falling back to %ss",
                TIMEOUT_ENV,
                raw,
                DEFAULT_TIMEOUT,
            )
    return DEFAULT_TIMEOUT


def _force_outbox() -> bool:
    return os.environ.get(FORCE_OUTBOX_ENV, "").lower() in ("1", "true", "yes")


def _post(
    url: str,
    payload: dict,
    *,
    outbox_event: str | None = None,
    experiment_id: str | None = None,
    training_run_id: str | None = None,
) -> tuple[dict | None, int | None]:
    """POST to the dashboard.

    Returns (parsed JSON, status_code) or (None, code).

    If ``outbox_event`` is set, network-layer failures are persisted to the
    PDX outbox for later replay. HTTP errors and UUID-allocating calls are
    not replayed.
    """
    if outbox_event and _force_outbox():
        _outbox(
            outbox_event,
            "POST",
            url,
            payload,
            experiment_id=experiment_id,
            training_run_id=training_run_id,
        )
        return None, None
    try:
        data = json.dumps(payload).encode()
        req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=_get_timeout(), context=_get_ssl_context()) as resp:
            return json.loads(resp.read().decode()), resp.status
    except HTTPError as exc:
        # Read and return the body so callers can recover from 409 Conflict, etc.
        try:
            body = json.loads(exc.read().decode())
        except Exception:  # noqa: BLE001
            body = None
        logger.warning("GEAR Dashboard API POST %s failed (non-fatal): %s", url, exc)
        return body, exc.code
    except Exception as exc:  # noqa: BLE001
        logger.warning("GEAR Dashboard API POST %s failed (non-fatal): %s", url, exc)
        if outbox_event:
            _outbox(
                outbox_event,
                "POST",
                url,
                payload,
                experiment_id=experiment_id,
                training_run_id=training_run_id,
            )
        return None, None


def _get(url: str) -> dict | None:
    """Fire-and-forget GET. Returns parsed JSON or None."""
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=_get_timeout(), context=_get_ssl_context()) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001
        logger.warning("GEAR Dashboard API GET %s failed (non-fatal): %s", url, exc)
        return None


def _patch(
    url: str,
    payload: dict,
    *,
    outbox_event: str | None = None,
    experiment_id: str | None = None,
    training_run_id: str | None = None,
) -> dict | None:
    """PATCH the dashboard.

    Returns parsed JSON or None. If ``outbox_event`` is set, network-layer
    failures are persisted to the PDX outbox for later replay.
    """
    if outbox_event and _force_outbox():
        _outbox(
            outbox_event,
            "PATCH",
            url,
            payload,
            experiment_id=experiment_id,
            training_run_id=training_run_id,
        )
        return None
    try:
        data = json.dumps(payload).encode()
        req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="PATCH")
        with urlopen(req, timeout=_get_timeout(), context=_get_ssl_context()) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001
        logger.warning("GEAR Dashboard API PATCH %s failed (non-fatal): %s", url, exc)
        if outbox_event:
            _outbox(
                outbox_event,
                "PATCH",
                url,
                payload,
                experiment_id=experiment_id,
                training_run_id=training_run_id,
            )
        return None


def _outbox(
    event: str,
    verb: str,
    url: str,
    payload: dict,
    *,
    experiment_id: str | None,
    training_run_id: str | None,
) -> None:
    """Best-effort write of a failed dashboard call to the PDX outbox."""
    try:
        from groot.core.training.experiment import event_outbox

        event_outbox.write_event(
            event=event,
            verb=verb,
            url=url,
            body=payload,
            experiment_id=experiment_id,
            training_run_id=training_run_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("GEAR Dashboard event-outbox dispatch failed (non-fatal): %s", exc)


def _first_env_value(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _normalize_branch(branch: str) -> str:
    branch = branch.strip()
    for prefix in ("refs/heads/", "origin/"):
        if branch.startswith(prefix):
            branch = branch.removeprefix(prefix)
    if branch == "HEAD":
        return ""
    return branch


def _first_branch_env(*names: str) -> str:
    for name in names:
        branch = _normalize_branch(os.environ.get(name, ""))
        if branch:
            return branch
    return ""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _get_groot_commit() -> str:
    """Resolve the source commit, preferring CI metadata over local git state."""
    env_commit = _first_env_value(
        "GROOT_COMMIT_HASH",
        "CI_GIT_COMMIT",
        "CI_COMMIT_SHA",
        "CI_COMMIT_SHORT_SHA",
    )
    if env_commit:
        return env_commit
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=_repo_root(),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def _get_branch() -> str:
    """Resolve the source branch, preferring CI metadata over local git state.

    GitLab CI checks out commits in detached-HEAD mode, so plain
    ``git rev-parse --abbrev-ref HEAD`` returns ``HEAD`` for the dominant
    golden-run path. Use explicit CI variables first and only fall back to git
    for local launches.
    """
    env_branch = _first_branch_env(
        "GROOT_BRANCH",
        "GROOT_COMMIT_BRANCH",
        "CI_GIT_BRANCH",
        "CI_MERGE_REQUEST_SOURCE_BRANCH_NAME",
        "CI_COMMIT_BRANCH",
        "CI_COMMIT_REF_NAME",
        "GIT_BRANCH",
    )
    if env_branch:
        return env_branch

    for cmd in (
        ["git", "branch", "--show-current"],
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
    ):
        try:
            branch = (
                subprocess.check_output(
                    cmd,
                    cwd=_repo_root(),
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
            branch = _normalize_branch(branch)
            if branch:
                return branch
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not resolve git branch via %s: %s", " ".join(cmd), exc)
            continue
    return ""


def _infer_gpu_type(cluster_type: str) -> str:
    mapping = {"gb200": "GB200", "gb300": "GB300", "h100": "H100", "l40": "L40", "l40s": "L40S"}
    return mapping.get(cluster_type.lower(), cluster_type.upper())


def register_experiment(
    *,
    experiment_name: str,
    who: str | None = None,
    wandb_project: str = "groot_omni",
    lfs_prefix: str | None = None,
    checkpoint_root_source: str | None = None,
    requested_lfs_checkpoint_path: str | None = None,
    requested_msc_checkpoint_url: str | None = None,
    actual_lfs_checkpoint_path: str | None = None,
    actual_msc_checkpoint_url: str | None = None,
    repo: str | None = None,
    identity_version: int | None = None,
    identity_hash: str | None = None,
    identity_payload: dict[str, Any] | None = None,
    config_hash: str | None = None,
    config_snapshot_uri: str | None = None,
    code_source_type: str | None = None,
    code_snapshot_uri: str | None = None,
    code_package_hash: str | None = None,
    code_package_fingerprint: str | None = None,
    code_commit: str | None = None,
    code_branch: str | None = None,
    code_is_dirty: bool | None = None,
    code_main_ref: str | None = None,
    code_main_merge_base: str | None = None,
    code_uncommitted_diff_uri: str | None = None,
    groot_commit: str = "",
    branch: str = "",
    start_from_checkpoint: str | None = None,
    datasets: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Register an experiment via POST /experiments.

    The dashboard is the authority on canonical output paths: it computes
    ``checkpoint_path`` (always), ``lfs_checkpoint_path`` (when ``lfs_prefix``
    is provided), and ``wandb_run_name`` and returns them in the response.
    Clients send identity metadata and intent (``wandb_project``,
    ``lfs_prefix``, ``parent_checkpoint_path``); they do not send
    ``checkpoint_path``.

    Returns a dict with the API-returned record fields:
        experiment_id, experiment_ref, checkpoint_path,
        lfs_checkpoint_path, wandb_run_name
    Returns an empty dict on failure. Skips re-registration if
    ``GEAR_EXPERIMENT_ID`` is already set in the environment.
    """
    if os.environ.get("GEAR_EXPERIMENT_ID"):
        return {
            "experiment_id": os.environ.get("GEAR_EXPERIMENT_ID", ""),
            "experiment_ref": os.environ.get("GEAR_EXPERIMENT_REF", ""),
            "checkpoint_path": os.environ.get("GEAR_CHECKPOINT_PATH", ""),
            "lfs_checkpoint_path": os.environ.get("GEAR_LFS_CHECKPOINT_PATH", ""),
            "wandb_run_name": os.environ.get("GEAR_WANDB_RUN_NAME", ""),
        }

    api_url = _get_api_url()
    who = who or getpass.getuser()
    branch = branch or os.environ.get("GROOT_BRANCH", "") or _get_branch()

    payload: dict[str, Any] = {
        "experiment_name": experiment_name,
        "who": who,
        "wandb_project": wandb_project,
        "groot_commit": groot_commit,
        "branch": branch,
        "datasets": datasets or [],
    }
    optional_fields = {
        "checkpoint_root_source": checkpoint_root_source,
        "requested_lfs_checkpoint_path": requested_lfs_checkpoint_path,
        "requested_msc_checkpoint_url": requested_msc_checkpoint_url,
        "actual_lfs_checkpoint_path": actual_lfs_checkpoint_path,
        "actual_msc_checkpoint_url": actual_msc_checkpoint_url,
        "repo": repo,
        "identity_version": identity_version,
        "identity_hash": identity_hash,
        "identity_payload": identity_payload,
        "config_hash": config_hash,
        "config_snapshot_uri": config_snapshot_uri,
        "code_source_type": code_source_type,
        "code_snapshot_uri": code_snapshot_uri,
        "code_package_hash": code_package_hash,
        "code_package_fingerprint": code_package_fingerprint,
        "code_commit": code_commit,
        "code_branch": code_branch,
        "code_is_dirty": code_is_dirty,
        "code_main_ref": code_main_ref,
        "code_main_merge_base": code_main_merge_base,
        "code_uncommitted_diff_uri": code_uncommitted_diff_uri,
    }
    payload.update({key: val for key, val in optional_fields.items() if val is not None})
    if lfs_prefix:
        payload["lfs_prefix"] = lfs_prefix
    if start_from_checkpoint:
        # Server resolves parent_experiment_id and parent_checkpoint_step from
        # the raw path -- we just pass it through.
        payload["parent_checkpoint_path"] = start_from_checkpoint

    result, status = _post(f"{api_url}/experiments", payload)

    # 409 Conflict on (who, experiment_name) returns the existing record with
    # the same shape as 201, so reuse is symmetric.
    if status == 409 and result:
        record = result.get("detail") if "detail" in result else result
        if isinstance(record, dict) and record.get("experiment_id"):
            _record_to_env(record)
            return _extract_record(record)

    if not result:
        return {}

    if result.get("experiment_id"):
        _record_to_env(result)
        os.environ.setdefault("GROOT_BRANCH", branch)

    return _extract_record(result)


def _extract_record(record: dict[str, Any]) -> dict[str, str]:
    """Pluck the canonical fields gr00t cares about out of an API response."""
    return {
        "experiment_id": record.get("experiment_id", "") or "",
        "experiment_ref": record.get("experiment_ref", "") or "",
        "checkpoint_path": record.get("checkpoint_path", "") or "",
        "lfs_checkpoint_path": record.get("lfs_checkpoint_path", "") or "",
        "wandb_run_name": record.get("wandb_run_name", "") or "",
    }


def _record_to_env(record: dict[str, Any]) -> None:
    """Mirror the API-returned record onto GEAR_* env vars.

    Launchers and Ray workers read these values downstream.
    """
    fields = _extract_record(record)
    if fields["experiment_id"]:
        os.environ["GEAR_EXPERIMENT_ID"] = fields["experiment_id"]
    if fields["experiment_ref"]:
        os.environ["GEAR_EXPERIMENT_REF"] = fields["experiment_ref"]
    if fields["checkpoint_path"]:
        os.environ["GEAR_CHECKPOINT_PATH"] = fields["checkpoint_path"]
    if fields["lfs_checkpoint_path"]:
        os.environ["GEAR_LFS_CHECKPOINT_PATH"] = fields["lfs_checkpoint_path"]
    if fields["wandb_run_name"]:
        os.environ["GEAR_WANDB_RUN_NAME"] = fields["wandb_run_name"]


def register_training_run(
    *,
    experiment_id: str,
    osmo_workflow_id: str = "",
    pool: str = "",
    gpu_type: str = "",
    num_gpus_per_node: int | None = None,
    num_run_gpus: int | None = None,
    total_gpus: int | None = None,
    launch_command: str = "",
    started_at: str | None = None,
    status: str = "running",
    image: str = "",
    config_json: dict | None = None,
    code_snapshot_uri: str | None = None,
    code_package_hash: str | None = None,
    config_hash: str | None = None,
    config_snapshot_uri: str | None = None,
    resume_checkpoint_path: str | None = None,
    resume_source: str | None = None,
) -> str:
    """Register a training run via POST /experiments/{id}/training-runs.

    Returns training_run_id or "" on failure.
    """
    if not experiment_id:
        return ""

    api_url = _get_api_url()
    payload: dict[str, Any] = {
        "osmo_workflow_id": osmo_workflow_id,
        "status": status,
    }
    if pool:
        payload["pool"] = pool
    if gpu_type:
        payload["gpu_type"] = gpu_type
    if num_gpus_per_node is not None:
        payload["num_gpus_per_node"] = num_gpus_per_node
    if num_run_gpus is not None:
        payload["num_run_gpus"] = num_run_gpus
    if total_gpus is not None:
        payload["total_gpus"] = total_gpus
    if launch_command:
        payload["launch_command"] = launch_command
    if started_at:
        payload["started_at"] = started_at
    if image:
        payload["image"] = image
    if config_json:
        payload["config_json"] = config_json
    optional_fields = {
        "code_snapshot_uri": code_snapshot_uri,
        "code_package_hash": code_package_hash,
        "config_hash": config_hash,
        "config_snapshot_uri": config_snapshot_uri,
        "resume_checkpoint_path": resume_checkpoint_path,
        "resume_source": resume_source,
    }
    payload.update({key: val for key, val in optional_fields.items() if val is not None})

    result, _status = _post(f"{api_url}/experiments/{experiment_id}/training-runs", payload)
    if not result:
        return ""

    run_id = result.get("training_run_id", "")
    if run_id:
        pass
    return run_id


def update_experiment(
    experiment_id: str,
    **fields: Any,
) -> None:
    """PATCH /experiments/{id} with the given fields.

    Used by experiment.py to enrich an experiment record after the launcher has
    already registered it — launchers don't know the resolved training config
    (datasets, start_from_checkpoint, checkpoint_path, wandb_run_name) but the
    training process does.
    """
    if not experiment_id:
        return
    api_url = _get_api_url()
    result = _patch(
        f"{api_url}/experiments/{experiment_id}",
        fields,
        outbox_event="experiment_patched",
        experiment_id=experiment_id,
    )
    if result is not None:
        pass


def update_training_run(
    experiment_id: str,
    training_run_id: str,
    **fields: Any,
) -> None:
    """PATCH /experiments/{id}/training-runs/{run_id} with the given fields.

    Use this for non-status scalar fields (``ray_submission_id``, ``ended_at``,
    ``config_json`` etc.). For ``status`` transitions and audit events, prefer
    :func:`append_training_run_status_event` which atomically appends to the
    ``status_updates`` array instead of overwriting it.
    """
    if not experiment_id or not training_run_id:
        return
    api_url = _get_api_url()
    result = _patch(
        f"{api_url}/experiments/{experiment_id}/training-runs/{training_run_id}",
        fields,
        outbox_event="training_run_patched",
        experiment_id=experiment_id,
        training_run_id=training_run_id,
    )
    if result is not None:
        pass


def append_training_run_status_event(
    experiment_ref_or_id: str,
    run_id: str,
    *,
    status: str,
    timestamp: str | None = None,
    message: str | None = None,
    metadata: dict[str, Any] | None = None,
    set_status: bool = True,
) -> None:
    """Append a single status event via POST .../training-runs/{run_id}/status-updates.

    The endpoint is atomic and append-only -- concurrent clients (launcher,
    rank 0, evaluator) can add status events without first reading the
    existing array. With ``set_status=True`` (the default and the common
    case), the run's scalar ``status`` field is updated to ``status`` in the
    same transaction.

    ``run_id`` accepts either the ``training_run_id`` UUID or the
    ``osmo_workflow_id`` -- matching the dashboard's GET / PATCH endpoints.

    Fails closed: logs a warning and returns on any error so a missing
    endpoint or transient network blip never blocks the launcher.
    """
    if not experiment_ref_or_id or not run_id:
        return
    api_url = _get_api_url()
    payload: dict[str, Any] = {"status": status, "set_status": set_status}
    if timestamp:
        payload["timestamp"] = timestamp
    if message:
        payload["message"] = message
    if metadata:
        payload["metadata"] = metadata

    result, _status = _post(
        f"{api_url}/experiments/{experiment_ref_or_id}/training-runs/{run_id}/status-updates",
        payload,
        outbox_event="training_run_status_update",
        experiment_id=experiment_ref_or_id,
        training_run_id=run_id,
    )
    if result is not None:
        pass
