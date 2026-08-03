# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""N2 Turbo LIBERO-10 golden-run experiment.

Launch on H100 with:

    uv run ol train --experiment n2_turbo_libero_10 --compute-type h100
"""

from groot.infra.experiments.training.configs._n2_turbo_libero_config import (
    make_n2_turbo_libero_experiment,
)
from groot.infra.experiments.training.configs.n2_libero_10 import (
    EXPERIMENT as BASE_LIBERO_EXPERIMENT,
)

EXPERIMENT = make_n2_turbo_libero_experiment(BASE_LIBERO_EXPERIMENT)
