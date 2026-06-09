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

"""CPU-only regression pin for ``_init_distributed_process_group``.

The load-bearing invariant: under ``torchrun`` (``WORLD_SIZE > 1``) every
rank must call ``torch.cuda.set_device(LOCAL_RANK)`` first and *then*
``torch.distributed.init_process_group(backend="nccl",
device_id=torch.device("cuda:LOCAL_RANK"))``. If that ordering breaks or
``device_id=`` is dropped, NCCL defers communicator construction to the
first collective and guesses the device, which on restricted K8s GPU pods
surfaces as an opaque ``ncclUnhandledCudaError``. PyTorch >=2.4 documents
``device_id=`` as the recommended pattern.

This test patches out ``torch.distributed`` / ``torch.cuda`` so it runs on
every CPU CI shard without standing up a real process group.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from gr00t.experiment.experiment import _init_distributed_process_group
import torch


def test_torchrun_path_binds_device_id_and_orders_set_device_first(monkeypatch):
    """Under ``torchrun`` (``WORLD_SIZE>1``): ``set_device(LOCAL_RANK)`` must
    run before ``init_process_group(backend="nccl", device_id=cuda:LOCAL_RANK)``.

    Both the ``device_id=`` kwarg and the call ordering matter; this fails
    with a precise message instead of waiting for a multi-GPU CI flake.
    """
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "2")

    # ``manager.attach_mock`` is the canonical way to record call ordering
    # across two independent ``patch()`` targets; without it each mock tracks
    # only its own calls and cross-mock ordering is invisible.
    manager = MagicMock()
    init_pg = MagicMock()
    set_device = MagicMock()
    manager.attach_mock(init_pg, "init_process_group")
    manager.attach_mock(set_device, "set_device")

    with (
        patch("torch.distributed.is_initialized", return_value=False),
        patch("torch.distributed.get_rank", return_value=2),
        patch("torch.distributed.init_process_group", new=init_pg),
        patch("torch.cuda.set_device", new=set_device),
    ):
        rank = _init_distributed_process_group()

    assert rank == 2

    set_device.assert_called_once_with(2)
    init_pg.assert_called_once()
    init_kwargs = init_pg.call_args.kwargs
    assert init_kwargs.get("backend") == "nccl", (
        f"init_process_group must pin backend=nccl, got kwargs={init_kwargs}"
    )
    device_id = init_kwargs.get("device_id")
    assert device_id is not None, (
        "init_process_group must pass device_id= so NCCL binds the rank to its GPU "
        "at communicator construction (see module docstring)."
    )
    assert device_id == torch.device("cuda:2"), (
        f"device_id must be cuda:LOCAL_RANK (cuda:2 here), got {device_id!r}"
    )

    # Ordering pin: set_device(LOCAL_RANK) must precede init_process_group.
    # Without it NCCL has no current-device hint and can still race even when
    # device_id= is passed.
    call_order = [name for (name, _, _) in manager.mock_calls]
    assert call_order == ["set_device", "init_process_group"], (
        f"torch.cuda.set_device must be called BEFORE dist.init_process_group; "
        f"got call order: {call_order}"
    )
