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

"""Run GR00T N1.7 with the ExecuTorch programs written by ``export_executorch_n1d7.py``.

``ExecuTorchEngine`` has the interface of ``trt_torch.Engine``, so the
full-pipeline forward functions of ``trt_model_forward`` drive the ExecuTorch
programs unchanged: the same glue ops stay in PyTorch (embed_tokens,
masked_scatter, get_rope_index, VLLN, the denoising loop) and every component
runs as an ExecuTorch program.
"""

from functools import partial
import importlib
import json
import os

from gr00t.deployment.action_padding import assert_dit_engine_matches_action_padding
import torch

from ._trt_contract import assert_engine_matches_policy
from .trt_model_forward import (
    _validate_vl_attention_engine,
    action_head_tensorrt_forward,
    qwen3_backbone_full_trt_forward,
)


EXECUTORCH_IO_KEY = "executorch_io"
_REQUIRED_COMPONENTS = ("vit", "llm", "state_encoder", "action_encoder", "dit", "action_decoder")
_runtime = None


def _executorch_runtime():
    global _runtime
    if _runtime is None:
        # Registers the TensorRT backend with the ExecuTorch runtime.
        from executorch.runtime import Runtime
        import torch_tensorrt_executorch_runtime  # noqa: F401

        runtime = Runtime.get()
        # The components run one after another, so their engines can share
        # one activation scratch buffer instead of holding one each.
        runtime.backend_registry.set_option(
            "TensorRTBackend", {"use_shared_activation_scratch": True}
        )
        _runtime = runtime
    return _runtime


class ExecuTorchEngine:
    """The ``forward`` method of one exported component, called like ``trt_torch.Engine``."""

    def __init__(self, program_dir, io):
        self.file = os.path.join(program_dir, io["file"])
        data_file = io.get("data_file")
        program = _executorch_runtime().load_program(
            self.file, data_path=os.path.join(program_dir, data_file) if data_file else None
        )
        self._method = program.load_method("forward")
        self.in_meta = [
            [item["name"], tuple(item["shape"]), getattr(torch, item["dtype"])]
            for item in io["inputs"]
        ]
        self.out_meta = [
            [item["name"], None, getattr(torch, item["dtype"])] for item in io["outputs"]
        ]

    def dtype_of(self, tensor_name):
        for name, _shape, dtype in self.in_meta:
            if name == tensor_name:
                return dtype
        raise KeyError(f"Input tensor '{tensor_name}' not found in {self.file}.")

    def set_runtime_tensor_shape(self, name, shape):
        """Shapes come with the input tensors, so there is nothing to set."""

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return_list = kwargs.pop("return_list", False)
        names = [item[0] for item in self.in_meta]
        if len(args) > len(names):
            raise ValueError(f"{self.file}: {len(args)} inputs given, the program takes {names}")
        bound = dict(zip(names, args))
        bound.update((name, value) for name, value in kwargs.items() if name in names)
        missing = [name for name in names if name not in bound]
        if missing:
            raise ValueError(f"{self.file}: missing inputs {missing}")
        inputs = []
        for name, _shape, dtype in self.in_meta:
            x = bound[name]
            assert x.dtype == dtype, f"{self.file}: {name} is {x.dtype}, expected {dtype}"
            assert x.is_cuda, f"{self.file}: {name} is on {x.device}, expected cuda"
            inputs.append(x.contiguous())
        outputs = self._method.execute(inputs)
        if return_list:
            return list(outputs)
        return {item[0]: out for item, out in zip(self.out_meta, outputs, strict=True)}

    def close(self):
        self._method = None


def executorch_components(program_dir, metadata):
    """Return the program I/O in ``metadata``, refusing an export that lacks a component."""
    io = (metadata or {}).get(EXECUTORCH_IO_KEY)
    if io is None:
        raise ValueError(f"{program_dir} holds no ExecuTorch export (export_executorch_n1d7.py)")
    required = list(_REQUIRED_COMPONENTS)
    if metadata.get("vl_self_attention_exported", True):
        required.append("vl_self_attention")
    missing = [
        name
        for name in required
        if name not in io
        or not all(
            os.path.exists(os.path.join(program_dir, f))
            for f in (io[name]["file"], io[name].get("data_file"))
            if f
        )
    ]
    if missing:
        raise ValueError(
            f"{program_dir} lacks {missing}. Export every component with export_executorch_n1d7.py."
        )
    return io


def check_executorch_export(program_dir):
    """Fail before any slow step if the runtime packages do not load or the export is incomplete."""
    for module in ("executorch.runtime", "torch_tensorrt_executorch_runtime"):
        try:
            importlib.import_module(module)
        except ImportError as e:
            raise ImportError(
                f"ExecuTorch programs need {module}; see the deployment README."
            ) from e
    metadata_path = os.path.join(program_dir, "export_metadata.json")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"No ExecuTorch export in {program_dir}")
    with open(metadata_path) as f:
        return executorch_components(program_dir, json.load(f))


def setup_executorch_programs(policy, program_dir):
    """Replace the backbone and action head with the ExecuTorch programs in ``program_dir``.

    Mirrors ``trt_model_forward._setup_n17_full_pipeline`` for the baseline
    (non-optimized) profile, including its checks against the loaded policy.
    """
    source = "setup_executorch_programs"
    metadata = assert_engine_matches_policy(policy, program_dir, source=source)
    io = executorch_components(program_dir, metadata)
    backbone = policy.model.backbone
    qwen_model = backbone.model
    action_head = policy.model.action_head

    backbone._embedding_layer = qwen_model.model.language_model.get_input_embeddings()
    backbone._image_token_id = qwen_model.config.image_token_id
    backbone._vit_baked_grid_thw = metadata.get("vit_grid_thw")
    backbone.vit_engine = ExecuTorchEngine(program_dir, io["vit"])
    del qwen_model.model.visual
    torch.cuda.empty_cache()
    backbone.llm_engine = ExecuTorchEngine(program_dir, io["llm"])
    del qwen_model.model.language_model.layers
    del qwen_model.model.language_model.norm
    torch.cuda.empty_cache()
    backbone.forward = partial(qwen3_backbone_full_trt_forward, backbone)

    action_head._trt_optimization_runner = None
    action_head.vl_sa_engine = None
    if "vl_self_attention" in io:
        vl_sa_engine = ExecuTorchEngine(program_dir, io["vl_self_attention"])
        _validate_vl_attention_engine(vl_sa_engine, mask_vl_padding=action_head.mask_vl_padding)
        action_head.vl_sa_engine = vl_sa_engine
        del action_head.vl_self_attention
    del action_head.model
    del action_head.state_encoder
    del action_head.action_encoder
    del action_head.action_decoder
    torch.cuda.empty_cache()

    assert action_head.action_dim == action_head.config.max_action_dim
    action_head.state_encoder_engine = ExecuTorchEngine(program_dir, io["state_encoder"])
    action_head.action_encoder_engine = ExecuTorchEngine(program_dir, io["action_encoder"])
    action_head.dit_engine = ExecuTorchEngine(program_dir, io["dit"])
    assert_dit_engine_matches_action_padding(
        action_head.config, {item[0] for item in action_head.dit_engine.in_meta}, source
    )
    action_head.action_decoder_engine = ExecuTorchEngine(program_dir, io["action_decoder"])
    action_head.get_action = partial(action_head_tensorrt_forward, action_head)
    print(f"N1.7 ExecuTorch programs loaded from {program_dir}")


def close_executorch_programs(policy):
    """Release the programs ``setup_executorch_programs`` attached to ``policy``."""
    for module in policy.model.modules():
        for value in list(vars(module).values()):
            if isinstance(value, ExecuTorchEngine):
                value.close()
