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

"""CPU regressions for frozen benchmark inputs and paired preprocessing samples."""

from pathlib import Path

import pytest


def _load_benchmark_helpers():
    """Exercise the public benchmark helpers without importing GPU dependencies."""
    import ast
    import json
    import math
    from types import SimpleNamespace

    import numpy as np

    source = Path(__file__).resolve().parents[3] / "scripts/deployment/benchmark_inference.py"
    names = {"load_observation", "save_data_processing_reference", "load_data_processing_reference"}
    tree = ast.parse(source.read_text())
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == names
    namespace = {"json": json, "math": math, "Path": Path, "np": np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def test_preprocessing_reference_reuses_samples_and_rejects_other_inputs(tmp_path):
    import copy
    import json

    contract_module = _load_benchmark_helpers()
    contract = {
        "device": "NVIDIA Thor",
        "model_path": "/model",
        "configuration": {"num_iterations": 3, "processed_inputs": {"image": "hash"}},
    }
    path = tmp_path / "processing.json"
    samples = [7.0, 8.0, 9.0]
    contract_module.save_data_processing_reference(path, contract, samples)
    assert contract_module.load_data_processing_reference(path, contract) == samples
    wrong_input = copy.deepcopy(contract)
    wrong_input["configuration"]["processed_inputs"]["image"] = "other-image"
    with pytest.raises(ValueError, match="does not match"):
        contract_module.load_data_processing_reference(path, wrong_input)
    for invalid in ([7.0], [7.0, float("nan"), 9.0], [7.0, 0.0, 9.0]):
        payload = json.loads(path.read_text())
        payload["times_ms"] = invalid
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="complete finite positive"):
            contract_module.load_data_processing_reference(path, contract)


def test_frozen_input_preserves_raw_pixels_and_ignores_processed_cache(tmp_path):
    import numpy as np

    loader = _load_benchmark_helpers()
    template = {
        "video": {"image": np.zeros((1, 1, 2, 2, 3), dtype=np.uint8)},
        "state": {"x": np.zeros((1, 1), dtype=np.float32)},
        "language": {"task": [["move"]]},
    }
    path = tmp_path / "input.npz"
    frozen = template["video"]["image"] + 2
    np.savez(
        path,
        **{
            "observation.video.image": frozen,
            "observation.state.x": template["state"]["x"],
            "observation.language.task": np.asarray(template["language"]["task"]),
            "processed.pixel_values": np.ones(7),
        },
    )
    observation = loader.load_observation(path, template)
    np.testing.assert_array_equal(observation["video"]["image"], frozen)
    assert set(observation) == {"video", "state", "language"}
    assert observation["language"]["task"] == [["move"]]
    template["video"]["image"] = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape/dtype"):
        loader.load_observation(path, template)
