# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""S3 backend using boto3 library.

IMPORTANT: This is an INTERNAL component of the AsyncDataServingService.
"""

from pathlib import Path

import boto3
from boto3.s3 import transfer
from botocore import exceptions

from groot.core.data.hydrator.backends import base
from groot.core.data.hydrator.utils import aws, io
from groot.infra.job.data_credentials import DataCredentials


class S3Boto3Backend(base.StorageBackend):
    """AWS S3 backend using boto3 library.

    Default backend for Phase 1. Reliable and well-tested.

    Args:
        working_dir: Local working directory root where files will be downloaded
        credentials_profile: Profile name in ~/.gear/data_credentials (e.g. "aws" or "default").
        transfer_config: boto3 TransferConfig parameters dict:
            - multipart_threshold: File size to use multipart (default 25MB)
            - max_concurrency: Max threads for parallel uploads/downloads (default 10)
            - multipart_chunksize: Size of each part in multipart transfer (default 25MB)
            - use_threads: Whether to use threads (default True)
        credentials_file: Path to data credentials file (default: ~/.gear/data_credentials).
        s5cmd_numworkers: Number of goroutines s5cmd uses per sync process (default 256).
            Each letter-based sync task spawns one s5cmd process; this controls
            per-process concurrency.  Increase (e.g. 512) when small-file S3 workloads
            are I/O-bound within a single process.
    """

    def __init__(
        self,
        working_dir: str,
        credentials_profile: str,
        transfer_config: dict | None = None,
        credentials_file: str = DataCredentials.DEFAULT_CREDENTIALS_FILE,
        s5cmd_numworkers: int = 256,
    ):
        super().__init__(working_dir)
        self.s5cmd_numworkers = s5cmd_numworkers

        self._credentials_profile = credentials_profile
        self._credentials_file = credentials_file
        self._credentials: dict | None = None  # loaded lazily on first use
        self.transfer_config_params = transfer_config or {}

        # Build boto3 TransferConfig if parameters provided
        if self.transfer_config_params:
            self.transfer_config = transfer.TransferConfig(**self.transfer_config_params)
        else:
            # Use defaults optimized for video files
            self.transfer_config = transfer.TransferConfig(
                multipart_threshold=26214400,  # 25 MB
                max_concurrency=10,
                multipart_chunksize=26214400,  # 25 MB
                use_threads=True,
            )

    def _load_credentials(self) -> dict:
        """Load credentials lazily for the configured profile."""
        if self._credentials is None:
            self._credentials = aws.get_boto3_client_config(
                profile=self._credentials_profile,
                credentials_file=self._credentials_file,
            )
        return self._credentials

    def _get_s3_client(self, bucket: str | None = None):
        """Create boto3 S3 client with configured credentials.

        If *bucket* follows the ``-{region}-nvidia-gear`` naming scheme, the
        region and endpoint are derived from the bucket name, overriding the
        credentials default to avoid PermanentRedirect errors.

        Args:
            bucket: Optional bucket name used to derive regional endpoint.

        Returns:
            Configured boto3 S3 client.

        Raises:
            PermissionError: If credentials for the configured profile are unavailable.
        """
        try:
            config_ = dict(self._load_credentials())
        except (ValueError, FileNotFoundError) as e:
            raise PermissionError(
                f"S3 credentials unavailable for profile {self._credentials_profile!r}: {e}"
            ) from e

        region = None
        if bucket:
            region, endpoint_style = io.resolve_s3_region_and_endpoint_style(
                bucket, self._credentials_profile
            )
            if region and endpoint_style == "regional":
                config_["region_name"] = region
                config_["endpoint_url"] = f"https://s3.{region}.amazonaws.com"
        if config_.get("endpoint_url"):
            from botocore import config

            config_.setdefault("config", config.Config(s3={"addressing_style": "path"}))
        return boto3.client("s3", **config_)

    def download_file(
        self,
        remote_path: str,
    ) -> str:
        """Download file from S3 to local working directory.

        Uses hash-based naming to avoid path length limits:
        {self.working_dir}/{hash}{extension}
        (e.g., /tmp/working/a1b2c3d4e5f6g7h8.mp4)

        Args:
            remote_path: S3 URI (e.g., s3://bucket/path/to/file.mp4)

        Returns:
            local_path: Full path to the downloaded file

        Raises:
            ValueError: If remote_path is not a valid S3 URI
            FileNotFoundError: If S3 object doesn't exist
            PermissionError: If credentials are invalid or access denied
            RuntimeError: For other download errors
        """
        # Parse S3 URI
        try:
            bucket, key = io.parse_s3_uri(remote_path)
        except ValueError as e:
            raise ValueError(f"Invalid S3 URI: {remote_path}") from e

        # Generate local path using hash-based naming (avoids path length limits)
        # This MUST use the base class method to ensure consistency with cache manager
        local_path = self.generate_local_path(self.working_dir, remote_path)

        # Ensure working directory exists (parent dir is always working_dir for hashed paths)
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)

        # Create S3 client
        try:
            s3_client = self._get_s3_client(bucket)
        except (exceptions.NoCredentialsError, exceptions.ClientError) as e:
            raise PermissionError(f"Failed to create S3 client: {e}") from e

        # Download file
        try:
            s3_client.download_file(
                Bucket=bucket,
                Key=key,
                Filename=local_path,
                Config=self.transfer_config,
            )
        except exceptions.ClientError as e:
            # Extract error code from ClientError
            error_code = "Unknown"
            if hasattr(e, "response"):
                resp = e.response
                if isinstance(resp, dict):
                    error_code = resp.get("Error", {}).get("Code", "Unknown")

            if error_code in {"404", "NoSuchKey"}:
                raise FileNotFoundError(f"S3 object not found: {remote_path}") from e
            if error_code in {"403", "AccessDenied"}:
                raise PermissionError(f"Access denied to S3 object: {remote_path}") from e
            raise RuntimeError(f"Failed to download {remote_path}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error downloading {remote_path}: {e}") from e

        return local_path

    def get_file_size(
        self,
        remote_path: str,
        credentials: dict | None = None,  # noqa: ARG002
    ) -> int:
        """Get S3 object size using HEAD request.

        Args:
            remote_path: S3 URI (e.g., s3://bucket/path/to/file.mp4)
            credentials: Unused (kept for API compatibility with base class)

        Returns:
            File size in bytes

        Raises:
            ValueError: If remote_path is not a valid S3 URI
            FileNotFoundError: If S3 object doesn't exist
            PermissionError: If credentials are invalid or access denied
            RuntimeError: For other errors
        """
        # Parse S3 URI
        try:
            bucket, key = io.parse_s3_uri(remote_path)
        except ValueError as e:
            raise ValueError(f"Invalid S3 URI: {remote_path}") from e

        # Create S3 client
        try:
            s3_client = self._get_s3_client(bucket)
        except (exceptions.NoCredentialsError, exceptions.ClientError) as e:
            raise PermissionError(f"Failed to create S3 client: {e}") from e

        # Get object metadata
        try:
            response = s3_client.head_object(Bucket=bucket, Key=key)
            return response["ContentLength"]
        except exceptions.ClientError as e:
            # Extract error code from ClientError
            error_code = "Unknown"
            if hasattr(e, "response"):
                resp = e.response
                if isinstance(resp, dict):
                    error_code = resp.get("Error", {}).get("Code", "Unknown")

            if error_code in {"404", "NoSuchKey"}:
                raise FileNotFoundError(f"S3 object not found: {remote_path}") from e
            if error_code in {"403", "AccessDenied"}:
                raise PermissionError(f"Access denied to S3 object: {remote_path}") from e
            raise RuntimeError(f"Failed to get size of {remote_path}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error getting size of {remote_path}: {e}") from e

    def supports_direct_read(self, modality: str) -> bool:
        """Check if modality supports direct S3 read.

        Args:
            modality: File modality ("video", "parquet", "json", etc.)

        Returns:
            True if files of this modality can be read directly from S3 without download
        """
        # Parquet and JSON can be read directly from S3 (pandas read_parquet/read_json + storage_options)
        return modality in ("parquet", "json")

    def upload_file(self, remote_path: str, content: bytes | Path) -> None:
        """Upload content to S3.

        Args:
            remote_path: S3 URI (e.g. s3://bucket/key/meta/stats.json)
            content: ``bytes`` — uploaded directly via ``put_object``.
                ``Path`` — streamed from disk via ``upload_file`` (multipart
                for files >8 MB, no full-file buffering anywhere). Caller
                must keep the file alive on disk until the upload returns.

        Raises:
            ValueError: If remote_path is not a valid S3 URI
            RuntimeError: On upload failure
        """
        try:
            bucket, key = io.parse_s3_uri(remote_path)
        except ValueError as e:
            raise ValueError(f"Invalid S3 URI: {remote_path}") from e
        try:
            s3_client = self._get_s3_client(bucket)
            if isinstance(content, Path):
                s3_client.upload_file(
                    Filename=str(content),
                    Bucket=bucket,
                    Key=key,
                    Config=self.transfer_config,
                )
            else:
                s3_client.put_object(Bucket=bucket, Key=key, Body=content)
        except exceptions.ClientError as e:
            raise RuntimeError(f"Failed to upload {remote_path}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error uploading {remote_path}: {e}") from e

    def list_directory(self, remote_path: str) -> list[str]:
        """List all object URIs under an S3 prefix (metadata only, no download).

        Args:
            remote_path: S3 URI prefix (e.g. s3://bucket/prefix/)

        Returns:
            List of full S3 URIs for all objects under the prefix

        """
        return io.glob_paths(remote_path, "*", profile=self._credentials_profile)

    def sync_directory(
        self,
        remote_path: str,
        local_dir: str,
        on_pair: base.OnPair | None = None,
    ) -> list[tuple[str, str]]:
        """Sync all objects under an S3 prefix to a local directory using s5cmd.

        Parses s5cmd's stdout to extract the exact set of (s3_uri, local_path) pairs
        that were copied.  Returns this list so callers can register each file as READY
        immediately after the sync completes rather than relying on a post-hoc glob.

        Args:
            remote_path: S3 URI prefix (e.g. s3://bucket/prefix/)
            local_dir: Local directory to sync files into
            on_pair: Optional callback invoked for each copied source/destination pair.

        Returns:
            List of (s3_uri, local_path) pairs for every file successfully copied.

        Raises:
            subprocess.CalledProcessError: If s5cmd exits non-zero for a reason
                other than no matching objects (e.g. auth failure, I/O error).
        """
        import os
        import subprocess
        import tempfile

        os.makedirs(local_dir, exist_ok=True)
        normalized = io.normalize_path(remote_path)
        # If caller already provided a glob (e.g. "s3://bucket/prefix/a*" for
        # letter-based distributed sync), use it as-is; otherwise append "/*".
        s3_glob = normalized if "*" in normalized else normalized.rstrip("/") + "/*"

        # Resolve credentials once; _load_credentials is lazy but triggers I/O
        # on first call, so avoid re-invoking it per field below.
        creds = self._load_credentials()

        # Write resolved credentials to a temp file so s5cmd uses the correct
        # account/endpoint regardless of what the process environment has set.
        lines = ["[default]"]
        if creds.get("aws_access_key_id"):
            lines.append(f"aws_access_key_id = {creds['aws_access_key_id']}")
        if creds.get("aws_secret_access_key"):
            lines.append(f"aws_secret_access_key = {creds['aws_secret_access_key']}")
        if creds.get("region_name"):
            lines.append(f"region = {creds['region_name']}")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".credentials", delete=False) as f:
            f.write("\n".join(lines) + "\n")
            creds_file = f.name

        cmd = [
            "s5cmd",
            "--numworkers",
            str(self.s5cmd_numworkers),
            "--credentials-file",
            creds_file,
        ]
        if creds.get("endpoint_url"):
            cmd += ["--endpoint-url", creds["endpoint_url"]]
        cmd += ["sync", s3_glob, local_dir + "/"]

        # Stream stdout line by line and parse s5cmd's default per-operation log
        # format: ``cp <src> <dst>`` (one line per successfully-copied object).
        # Any line that does not match this 3-token shape is logged at INFO and
        # skipped; if s5cmd ever changes its stdout format (e.g. adds JSON
        # output via --json), the returned pair list will silently become empty
        # or partial. Lock this to s5cmd >= 2.2 where the "cp" prefix is stable
        # and covered by the unit test mocking Popen.stdout below.
        import logging as _logging

        _s5log = _logging.getLogger(__name__)
        pairs: list[tuple[str, str]] = []
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            assert proc.stdout is not None
            for raw in proc.stdout:
                parts = raw.rstrip().split()
                if len(parts) == 3 and parts[0] == "cp":
                    pairs.append((parts[1], parts[2]))
                    if on_pair is not None:
                        on_pair(parts[1], parts[2])
                else:
                    _s5log.info("[s5cmd] unexpected stdout line: %r", raw.rstrip())
            _, stderr_out = proc.communicate()
        finally:
            os.unlink(creds_file)
        if proc.returncode != 0:
            combined = (stderr_out or "").lower()
            if "no object found" in combined or "no match found" in combined:
                return []
            raise subprocess.CalledProcessError(proc.returncode, cmd, stderr=stderr_out)
        return pairs
