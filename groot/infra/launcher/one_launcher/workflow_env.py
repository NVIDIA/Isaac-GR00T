# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Environment variable names managed by one-launcher workflow setup."""

AUTHORITATIVE_WORKFLOW_ENV_NAMES = {
    "BATCH_SIZE",
    "CLUSTER_TYPE",
    "EXPECTED_NUM_GPUS",
    "EXPECTED_NUM_NODES",
    "GEAR_DASHBOARD_FORCE_OUTBOX",
    "GRADIENT_ACCUMULATION_STEPS",
    "GROOT_BRANCH",
    "GROOT_COMMIT_HASH",
    "GROOT_OL_ENABLE_RETRY_RESUME",
    "GROOT_OL_LFS_PREFIX",
    "GROOT_OL_LAUNCH_ID",
    "GROOT_OL_MAX_RETRIES",
    "GROOT_OL_RETRY_POLICY",
    "GROOT_OL_RETRY_RESCHEDULE_EXIT_CODE",
    "HF_HUB_CACHE",
    "JOB_NAME",
    "LFS_CACHE_PATH",
    "MSC_CACHE_PATH",
    "NUM_GPUS",
    "RAY_ENABLE_UV_RUN_RUNTIME_ENV",
    "RAY_WAIT_TIMEOUT_SECONDS",
    "TRAINING_OUTPUT_DIR",
    "USER_SLUG",
}
