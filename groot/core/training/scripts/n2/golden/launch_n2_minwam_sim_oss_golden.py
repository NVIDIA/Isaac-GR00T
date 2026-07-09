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

"""OSS-native Min-WAM stack-bowls simulation golden run."""

from groot.core.training.scripts.n2.golden.minwam_sim_oss import Recipe, main

RECIPE = Recipe(
    label="N2 Min-WAM Sim OSS Golden Posttrain",
    data_yaml="groot/core/training/configs/experiments/n2/yam/workstation_stack_bowls_200traj_hist.yaml",
    default_job_name="n2_minwam-yam-stack-bowls-il-sim-oss-golden-bidir-sd0.5-10k",
    max_n_cameras=3,
)


if __name__ == "__main__":
    main(RECIPE, launcher_module="groot.core.training.scripts.n2.golden.launch_n2_minwam_sim_oss_golden")
