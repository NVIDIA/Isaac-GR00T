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

import asyncio
import os

import boto3
from botocore.config import Config

S3_BUCKET_NAME = "GearHome"


class Database:
    """Database client for CI/CD pipeline."""

    def __init__(self):
        # Configure boto3 with increased connection pool
        boto_config = Config(
            max_pool_connections=50,  # Increase from default 10
            retries={"max_attempts": 3, "mode": "adaptive"},
        )

        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.environ["BUCKET_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["BUCKET_SECRET_ACCESS_KEY"],
            region_name=os.environ["BUCKET_REGION"],
            endpoint_url=os.environ["BUCKET_ENDPOINT_URL"],
            config=boto_config,
        )

        # Semaphore to limit concurrent requests
        self._semaphore = asyncio.Semaphore(20)  # Max 20 concurrent requests

    async def async_write(self, key: str, body: str) -> None:
        """Write a string to the database."""
        await asyncio.to_thread(
            self.s3_client.put_object,
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=body,
        )

    async def async_read(self, key: str) -> str:
        """Read a single object from S3 with concurrency control."""
        async with self._semaphore:
            response = await asyncio.to_thread(
                self.s3_client.get_object,
                Bucket=S3_BUCKET_NAME,
                Key=key,
            )
            return response["Body"].read().decode("utf-8")

    async def async_list(self, prefix: str) -> list[str]:
        """List all objects with given prefix."""
        async with self._semaphore:
            keys = []
            kwargs: dict = {"Bucket": S3_BUCKET_NAME, "Prefix": prefix}
            while True:
                response = await asyncio.to_thread(self.s3_client.list_objects_v2, **kwargs)
                keys.extend(obj["Key"] for obj in response.get("Contents", []))
                if not response.get("IsTruncated"):
                    break
                kwargs["ContinuationToken"] = response["NextContinuationToken"]
            return keys

    async def async_delete(self, key: str) -> None:
        """Delete a single object from S3."""
        await asyncio.to_thread(
            self.s3_client.delete_object,
            Bucket=S3_BUCKET_NAME,
            Key=key,
        )
