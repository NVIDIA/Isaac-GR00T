# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""PackDS: High-performance dataset format for VLA training.

This package provides a unified interface for reading, writing, and validating
PackDS datasets. PackDS provides a backend-agnostic read-only reader plus a
LanceDB-backed read/write implementation. State/action arrays are stored as
nested fixed-size-list Arrow arrays.

All metadata is stored in the LanceDB schema metadata (no external meta/ directory).

Example:
    from groot.core.data.packds import (
        PackDS, PackDSStep, encode_steps, decode_steps,
    )

    # Create a step (single action per row)
    step = PackDSStep(
        episode_id="ep_000000",
        step_index=0,
        step_time_ns=0,
        original_step_time_ns=0,
        annotations={"task": "pick up the red cup", "action_source": "human"},
        decoded_video={"cam1": frame_array},
        state={"joint_pos": np.array([0.1, 0.2, 0.3])},
        action={"joint_pos": np.array([0.2, 0.3, 0.4])},
    )

    # Encode video and write
    encoded = encode_steps([step], encode_jpeg=True)
    with PackDS("output.packds", mode="w") as ds:
        ds.write_steps(encoded)

    # Read back
    with PackDS("output.packds") as ds:
        table = ds.read_episodes(["ep_000000"])["ep_000000"]
        batch = from_table(table).decode_video()
        print(batch.decoded_video["cam1"][0].shape)
        print(batch[0].annotations["task"])  # "pick up the red cup"
        print(batch[0].annotations["action_source"])  # "human"
"""

# Main class
from groot.core.data.packds.backends import BACKENDS, TABLE_BACKENDS
from groot.core.data.packds.dataset import PackDS, PackDSLanceDB, PackDSReader
from groot.core.data.packds.sources import SOURCE_RESOLVERS, ResolvedSource, resolve_source

# Step dataclass and batch container
from groot.core.data.packds.step import PackDSStep, PackDSStepBatch

# Utility functions
from groot.core.data.packds.utils import (
    IntoSteps,
    column_to_key,
    extract_scalar,
    from_table,
    gear_creds_to_lance_storage_options,
    infer_schema,
    into_step_batch,
    into_table,
    is_nested_fixed_size_list_tensor_type,
    key_to_column,
    nested_fixed_size_list_shape,
    numpy_shape_to_fixed_size_list_type,
    to_table,
)

# Video encoding/decoding for PackDSStep
from groot.core.data.packds.video import (
    decode_steps,
    encode_steps,
)

__all__ = [
    "BACKENDS",
    "SOURCE_RESOLVERS",
    "TABLE_BACKENDS",
    "IntoSteps",
    "PackDS",
    "PackDSLanceDB",
    "PackDSReader",
    "PackDSStep",
    "PackDSStepBatch",
    "ResolvedSource",
    "column_to_key",
    "decode_steps",
    "encode_steps",
    "extract_scalar",
    "from_table",
    "gear_creds_to_lance_storage_options",
    "infer_schema",
    "into_step_batch",
    "into_table",
    "is_nested_fixed_size_list_tensor_type",
    "key_to_column",
    "nested_fixed_size_list_shape",
    "numpy_shape_to_fixed_size_list_type",
    "resolve_source",
]
