# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tactile-force and deform-map conditioning for GrootN2 (``tactile_cfg`` / ``deform_cfg``).

GrootN2, its processor and its pipeline call this package through a few small seams and
are unchanged for models without tactile configs. The vendor HB1 implementation is
internal-only and is loaded only for models with deformation enabled.
Submodules never import model.py or processor.py at module level: both import this package.
"""

from groot.core.training.models.n2.tactile.callbacks import (
    TactileLossWarmupCallback,
    trainer_callbacks,
)
from groot.core.training.models.n2.tactile.guards import (
    MissingTactileInputError,
    TactileUnsupportedError,
)
from groot.core.training.models.n2.tactile.infer import (
    check_future_inference,
    inference_pack_kwargs,
    prepare_future_inference,
)
from groot.core.training.models.n2.tactile.modules import (
    TactileSpec,
    attach,
    complete_encoder_load,
    encode_deform,
    encode_tactile,
    record_encoder_load,
)
from groot.core.training.models.n2.tactile.train import (
    TactileTrainInputs,
    add_training_loss,
    prepare_training_inputs,
    training_pack_kwargs,
)

__all__ = [
    "MissingTactileInputError",
    "TactileLossWarmupCallback",
    "TactileSpec",
    "TactileTrainInputs",
    "TactileUnsupportedError",
    "add_training_loss",
    "attach",
    "check_future_inference",
    "complete_encoder_load",
    "encode_deform",
    "encode_tactile",
    "inference_pack_kwargs",
    "prepare_future_inference",
    "prepare_training_inputs",
    "record_encoder_load",
    "trainer_callbacks",
    "training_pack_kwargs",
]
