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
"""No-op model used by the GPU data-loading dry-run.

Returns a zero loss with a single learnable parameter so that ``loss.backward()``
and ``optimizer.step()`` still exercise the real training loop without consuming
compute on a real model. Use this to attribute stalls to data loading vs.
host→device vs. trainer overhead.
"""
import os
import time

import torch
from torch import nn


class IdentityModel(nn.Module):
    """A no-op model that returns a zero loss with a real gradient.

    ``forward`` ignores its inputs entirely. The loss is the sum of a single
    learnable parameter initialized to zero, so:
      - ``compute_loss`` gets a scalar tensor with ``requires_grad=True``
      - ``loss.backward()`` produces a non-empty grad graph (grad = 1 on _param)
      - ``optimizer.step()`` has something to update (parameter drifts; nothing
        else does anything)

    Env vars:
      IDENTITY_FORWARD_SLEEP_MS: blocking sleep at end of forward, ms.
        Lands in both model_forward_time and training_step_time (forward is
        a subset of step). Use to simulate model compute when probing whether
        the dataloader pipeline keeps up.
    """

    def __init__(self) -> None:
        super().__init__()
        self._param = nn.Parameter(torch.zeros(1))
        self._forward_sleep_sec = float(os.environ.get("IDENTITY_FORWARD_SLEEP_MS", "0")) / 1000.0

    def forward(self, **_inputs) -> dict[str, torch.Tensor]:
        if self._forward_sleep_sec > 0:
            time.sleep(self._forward_sleep_sec)
        return {"loss": self._param.sum()}
