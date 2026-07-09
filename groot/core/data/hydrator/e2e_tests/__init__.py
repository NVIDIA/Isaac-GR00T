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

"""E2E test utilities for the hydrator (shared by tests under tests/).

This package provides shared utilities (e.g. utils.py) for hydrator E2E tests.
The actual E2E test modules live under tests/groot/core/data/hydrator/e2e_tests/
and are run with pytest, marked with requires_s3.

Run E2E tests:
    pytest tests/groot/core/data/hydrator/e2e_tests/ -v -m requires_s3
"""
