# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""WriteSession protocol — the interface that PackDS writer classes depend on.

Both the Lance backend (BaseLanceWriteSession) and the DeltaCAT backend
(DeltaCatWriteSession) satisfy this protocol.  PackDSLanceDB and
PackDSDeltaCat call only the methods listed here, so the backends are
swappable without changing the writer logic.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pyarrow as pa


@runtime_checkable
class WriteSession(Protocol):
    """Minimal write-session contract used by PackDS dataset classes.

    Four methods — write, flush, finalize, close:

    - ``write_batch``: Write a prepared Arrow table.
    - ``flush``: Make all pending writes durable and wait for async ops.
    - ``finalize``: Post-write housekeeping (indices, metadata, publish).
    - ``close``: Release resources.

    Sessions are ready to use after construction (no ``open()`` call).
    Schema tracking, episode counting, and other bookkeeping are internal
    to the session or owned by the dataset class — not on this protocol.
    """

    def write_batch(self, data: pa.Table) -> None:
        """Write a prepared Arrow table.  May buffer internally."""
        ...

    def flush(self) -> None:
        """Make all pending writes durable; wait for any async operations."""
        ...

    def finalize(self, metadata: dict[str, str]) -> None:
        """Post-write housekeeping: indices, metadata persistence, publish.

        Called once after all writes are complete but before ``close()``.
        *metadata* is the ``packds:*`` key-value dict to persist alongside
        the dataset.
        """
        ...

    def close(self) -> None:
        """Release resources.  No further writes after this."""
        ...
