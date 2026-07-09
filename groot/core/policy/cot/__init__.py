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
"""Robot-agnostic chain-of-thought (System-2) policy wrappers.

A remote VLM planner translates a high-level goal plus current camera frames
into a simplified next-step instruction, which is written into the observation
before the wrapped (System-1) policy computes actions. Camera-frame extraction
and the instruction key are injected so any embodiment can reuse these wrappers.
"""
