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

"""TensorRT forward functions for GR00T N1.7 inference.

This module provides TRT-accelerated forward functions that replace the
PyTorch backbone and action head during inference.

Architecture (n17_full_pipeline mode):
  Backbone: ViT (TRT) → embed_tokens + masked_scatter + get_rope_index (PyTorch)
            → LLM (TRT, with deepstack injection)
  Action Head: VLLN (PyTorch) → State Encoder (TRT) → denoising loop:
               [ Action Encoder (TRT) → DiT (TRT) → Action Decoder (TRT) ]

Architecture (vit_llm_only mode):
  Backbone: ViT (TRT) → embed_tokens + masked_scatter + get_rope_index (PyTorch)
            → LLM (TRT, with deepstack injection)
  Action Head: stays in PyTorch
  Use when DiT cannot be exported with dynamic vl_seq_len (e.g. torch 2.10 / sm121).

Architecture (action_head mode):
  Backbone: stays in PyTorch (Qwen3-VL)
  Action Head: VLLN (PyTorch) → State Encoder (TRT) → denoising loop:
               [ Action Encoder (TRT) → DiT (TRT) → Action Decoder (TRT) ]
"""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import gc
import json
import logging
import os
import sys
from typing import Optional

from gr00t.deployment.modes import InferenceMode
import torch
from transformers.feature_extraction_utils import BatchFeature


logger = logging.getLogger(__name__)


# Ensure sibling modules are importable (scripts/deployment is not a package)
_deploy_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _deploy_dir not in sys.path:
    sys.path.insert(0, _deploy_dir)
from ._trt_contract import (  # noqa: E402
    assert_engine_bundle_present,
    assert_engine_matches_policy,
    assert_grid_thw_matches,
)
from .n1d7_optimization_config import (  # noqa: E402
    ACTION_HEAD_FORMAT_VERSION,
    OptimizationArtifacts,
    format_quantization_summary,
)
from .trt_torch import Engine  # noqa: E402


class GraphSafeEngine(Engine):
    """Engine variant with stable output addresses and asynchronous execution."""

    def __init__(self, file, plugins=None):
        super().__init__(file, plugins=plugins or [])
        self._output_cache = {}
        self._closed = False

    def close(self):
        """Release cached buffers, then use the upstream Engine lifecycle."""
        if self._closed:
            return
        self._output_cache.clear()
        super().close()
        self.plugins = []

    def _get_output_tensor(self, name, shape, dtype, device, cache):
        if not cache:
            return torch.empty(tuple(shape), dtype=dtype, device=device)
        key = (name, tuple(shape), dtype, device.type, device.index)
        output = self._output_cache.get(key)
        if output is None:
            output = torch.empty(tuple(shape), dtype=dtype, device=device)
            self._output_cache[key] = output
        return output

    def set_runtime_tensor_shape(self, name, shape):
        if self._closed:
            raise RuntimeError(f"TensorRT engine is closed: {self.file}")
        if not self.execution_context.set_input_shape(name, shape):
            raise RuntimeError(f"Failed to set runtime shape for {name}: {tuple(shape)}")

    def forward(self, *args, **kwargs):
        if self._closed:
            raise RuntimeError(f"TensorRT engine is closed: {self.file}")
        return_list = kwargs.pop("return_list", False)
        synchronize = kwargs.pop("synchronize", False)
        # Stable (cached) output addresses are required under CUDA graph
        # capture; transient eager calls pass cache_outputs=False so one-off
        # shapes don't grow the cache.
        cache_outputs = kwargs.pop("cache_outputs", True)
        references = []
        stream = torch.cuda.current_stream()

        for index, tensor in enumerate(args):
            name, _shape, dtype = self.in_meta[index]
            self._bind_input(name, dtype, tensor, references)
        for name, _shape, dtype in self.in_meta:
            if name in kwargs:
                self._bind_input(name, dtype, kwargs[name], references)

        if len(references) != len(self.in_meta):
            raise ValueError(
                f"Expected {len(self.in_meta)} input tensors, received {len(references)}"
            )

        for name, _shape, dtype in self.out_meta:
            runtime_shape = self.execution_context.get_tensor_shape(name)
            output = self._get_output_tensor(
                name, runtime_shape, dtype, references[0].device, cache_outputs
            )
            self.execution_context.set_tensor_address(name, output.data_ptr())
            references.append(output)

        if not self.execution_context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError(f"TensorRT execution failed for {self.file}")
        if synchronize:
            stream.synchronize()

        outputs = references[len(self.in_meta) :]
        if return_list:
            return outputs
        return {item[0]: outputs[index] for index, item in enumerate(self.out_meta)}

    def _bind_input(self, name, dtype, tensor, references):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Input {name} must be a torch.Tensor")
        runtime_shape = self.execution_context.get_tensor_shape(name)
        if runtime_shape != tensor.shape:
            raise ValueError(f"Invalid input shape for {name}: {tensor.shape} != {runtime_shape}")
        if tensor.dtype != dtype:
            raise TypeError(f"Invalid input dtype for {name}: {tensor.dtype} != {dtype}")
        if not tensor.is_cuda:
            raise ValueError(f"Input {name} must be a CUDA tensor")
        tensor = tensor.contiguous()
        self.execution_context.set_tensor_address(name, tensor.data_ptr())
        references.append(tensor)


@dataclass
class ActionHeadGraphState:
    backbone_features: torch.Tensor
    image_mask: torch.Tensor
    backbone_attention_mask: torch.Tensor
    state: torch.Tensor
    embodiment_id: torch.Tensor
    actions: torch.Tensor
    initial_actions: torch.Tensor
    time_embeddings: torch.Tensor
    adaln_modulations: torch.Tensor
    output_modulations: torch.Tensor
    position_embeddings: Optional[torch.Tensor]
    graph: Optional[torch.cuda.CUDAGraph] = None
    output: Optional[torch.Tensor] = None


class OptimizedActionHeadRunner:
    """Own optimized action-head engines, constants, buffers, graph, and teardown."""

    def __init__(self, action_head, engine_dir: str, metadata: dict, dit_engine_path: str):
        artifacts = OptimizationArtifacts.from_metadata(
            engine_dir, metadata, artifact_kind="engine"
        )
        if artifacts is None:
            raise ValueError("OptimizedActionHeadRunner requires optimized metadata")
        if artifacts.format_version != ACTION_HEAD_FORMAT_VERSION:
            raise ValueError(f"Unsupported action-head format version: {artifacts.format_version}")
        for path in (artifacts.cross_kv, artifacts.constants):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing optimized action-head artifact: {path}")

        self.action_head = action_head
        self.engine_dir = engine_dir
        self.metadata = metadata
        self.closed = False
        # One CUDA graph for the first input signature; other signatures run
        # eagerly with transient buffers (see module docstring).
        self._graph_state: Optional[ActionHeadGraphState] = None
        self._graph_signature = None
        self._capture_failed = False
        self._eager_signatures_warned = set()

        # Reused directories may contain a stale full-pipeline VL attention engine.
        # Action-head-only exports retain this module in PyTorch.
        vl_sa_path = os.path.join(engine_dir, "vl_self_attention.engine")
        self.vl_sa_engine = (
            GraphSafeEngine(vl_sa_path)
            if metadata.get("export_mode") == "full_pipeline" and os.path.exists(vl_sa_path)
            else None
        )
        self.state_encoder_engine = GraphSafeEngine(
            os.path.join(engine_dir, "state_encoder.engine")
        )
        self.action_encoder_engine = GraphSafeEngine(
            os.path.join(engine_dir, "action_encoder.engine")
        )
        self.cross_kv_engine = GraphSafeEngine(artifacts.cross_kv)
        self.dit_engine = GraphSafeEngine(dit_engine_path)
        self.action_decoder_engine = GraphSafeEngine(
            os.path.join(engine_dir, "action_decoder.engine")
        )
        self._validate_interfaces()
        self.constants = self._load_constants(artifacts.constants)
        self.position_embeddings = self._make_position_embeddings()

    @classmethod
    def maybe_create(
        cls,
        action_head,
        engine_dir: str,
        metadata: Optional[dict],
        dit_engine_path: str,
    ) -> Optional["OptimizedActionHeadRunner"]:
        artifacts = OptimizationArtifacts.from_metadata(
            engine_dir, metadata, artifact_kind="engine"
        )
        if artifacts is None:
            return None
        return cls(action_head, engine_dir, metadata or {}, dit_engine_path)

    def _validate_interfaces(self):
        action_inputs = {item[0] for item in self.action_encoder_engine.in_meta}
        dit_inputs = {item[0] for item in self.dit_engine.in_meta}
        expected_dit = {
            "sa_embs",
            "cross_keys",
            "cross_values",
            "adaln_modulations",
            "output_modulation",
            "image_mask",
            "backbone_attention_mask",
        }
        if "timestep_embedding" not in action_inputs or not expected_dit.issubset(dit_inputs):
            raise RuntimeError(
                "Optimized metadata does not match the Action Encoder/DiT interfaces"
            )

    def _load_constants(self, path: str):
        constants = torch.load(path, map_location="cpu", weights_only=True)
        if constants.get("format_version") != ACTION_HEAD_FORMAT_VERSION:
            raise ValueError(f"Unsupported constants format: {constants.get('format_version')}")
        device = next(self.action_head.vlln.parameters()).device
        result = {
            "action_time_embeddings": constants["action_time_embeddings"]
            .to(
                device=device,
                dtype=self.action_encoder_engine.dtype_of("timestep_embedding"),
            )
            .contiguous(),
            "adaln_modulations": constants["adaln_modulations"]
            .to(device=device, dtype=self.dit_engine.dtype_of("adaln_modulations"))
            .contiguous(),
            "output_modulations": constants["output_modulations"]
            .to(device=device, dtype=self.dit_engine.dtype_of("output_modulation"))
            .contiguous(),
        }
        if result["action_time_embeddings"].shape[0] != int(
            self.action_head.num_inference_timesteps
        ):
            raise ValueError("Exported and runtime denoising step counts differ")
        return result

    def _make_position_embeddings(self):
        if not self.action_head.config.add_pos_embed:
            return None
        device = next(self.action_head.vlln.parameters()).device
        position_ids = torch.arange(
            self.action_head.config.action_horizon, dtype=torch.long, device=device
        )
        return (
            self.action_head.position_embedding(position_ids)
            .unsqueeze(0)
            .to(self.action_encoder_engine.dtype_of("actions"))
            .contiguous()
        )

    def __call__(self, backbone_output, action_input, options=None):
        if "action" in action_input:
            raise NotImplementedError(
                "Optimized TensorRT action-head graph does not support RTC action overlap"
            )
        del options
        if self.closed:
            raise RuntimeError("Optimized action-head runner is closed")
        if backbone_output.image_mask is None:
            raise ValueError("Optimized action head requires image_mask")
        if backbone_output.backbone_attention_mask is None:
            raise ValueError("Optimized action head requires backbone_attention_mask")

        signature = self._signature(backbone_output, action_input)

        # Capture the single CUDA graph on the first call.
        if self._graph_state is None and not self._capture_failed:
            state = self._make_graph_state(backbone_output, action_input)
            try:
                self._capture_graph(state)
                self._graph_state = state
                self._graph_signature = signature
                logger.info("Captured action-head CUDA graph for signature %s", signature)
            except Exception:
                logger.exception("CUDA graph capture failed; running eager TensorRT path")
                self._capture_failed = True
                return BatchFeature(data={"action_pred": self._execute(state, cache_outputs=False)})
            # The fresh state already holds this call's inputs and diffusion
            # noise. Replay without the noise refresh below, so a seeded run
            # (verify_n1d7_trt) consumes the same first RNG draw as the
            # PyTorch reference it is compared against.
            state.graph.replay()
            return BatchFeature(data={"action_pred": state.output})

        if self._graph_state is not None and signature == self._graph_signature:
            state = self._graph_state
            self._copy_inputs(state, backbone_output, action_input)
            # Latest N1.7 samples fresh diffusion noise on every inference.
            # The graph reads this persistent buffer, so refresh it before replay.
            if not hasattr(self.action_head, "init_actions"):
                state.initial_actions.normal_()
            state.graph.replay()
            return BatchFeature(data={"action_pred": state.output})

        # Different signature (or capture failed): eager TensorRT fallback with
        # transient buffers — nothing is cached, memory stays bounded.
        if signature not in self._eager_signatures_warned:
            self._eager_signatures_warned.add(signature)
            logger.warning(
                "Optimized action head running eager TensorRT fallback for input "
                "signature %s (CUDA graph is captured only for %s)",
                signature,
                self._graph_signature,
            )
        state = self._make_graph_state(backbone_output, action_input)
        return BatchFeature(data={"action_pred": self._execute(state, cache_outputs=False)})

    def _signature(self, backbone_output, action_input):
        return (
            tuple(backbone_output.backbone_features.shape),
            backbone_output.backbone_features.dtype,
            tuple(backbone_output.image_mask.shape),
            tuple(backbone_output.backbone_attention_mask.shape),
            tuple(action_input.state.shape),
            tuple(action_input.embodiment_id.shape),
            backbone_output.backbone_features.device.index,
        )

    @staticmethod
    def _reshape_state(state):
        if state.ndim == 3 and state.shape[1] > 1:
            return state.view(state.shape[0], 1, -1)
        return state

    def _copy_inputs(self, state, backbone_output, action_input):
        state.backbone_features.copy_(backbone_output.backbone_features)
        state.image_mask.copy_(backbone_output.image_mask)
        state.backbone_attention_mask.copy_(backbone_output.backbone_attention_mask)
        state.state.copy_(self._reshape_state(action_input.state))
        state.embodiment_id.copy_(action_input.embodiment_id)

    def _make_graph_state(self, backbone_output, action_input):
        device = backbone_output.backbone_features.device
        batch_size = backbone_output.backbone_features.shape[0]
        constants_batch = self.constants["action_time_embeddings"].shape[1]
        if constants_batch != batch_size:
            raise ValueError(
                f"Runtime batch {batch_size} != exported constants batch {constants_batch}"
            )
        vlln_parameter = next(self.action_head.vlln.parameters(), None)
        vlln_dtype = (
            vlln_parameter.dtype
            if vlln_parameter is not None
            else backbone_output.backbone_features.dtype
        )
        auxiliary_dtype = self.state_encoder_engine.dtype_of("state")
        action_state = self._reshape_state(action_input.state)
        actions = torch.empty(
            batch_size,
            self.action_head.config.action_horizon,
            self.action_head.action_dim,
            device=device,
            dtype=auxiliary_dtype,
        )
        if hasattr(self.action_head, "init_actions"):
            initial_actions = (
                self.action_head.init_actions.expand_as(actions)
                .to(device=device, dtype=auxiliary_dtype)
                .contiguous()
            )
        else:
            initial_actions = torch.randn_like(actions)

        state = ActionHeadGraphState(
            backbone_features=torch.empty(
                backbone_output.backbone_features.shape,
                device=device,
                dtype=vlln_dtype,
            ),
            image_mask=torch.empty_like(backbone_output.image_mask),
            backbone_attention_mask=torch.empty_like(backbone_output.backbone_attention_mask),
            state=torch.empty(action_state.shape, device=device, dtype=auxiliary_dtype),
            embodiment_id=torch.empty(
                action_input.embodiment_id.shape, device=device, dtype=torch.int64
            ),
            actions=actions,
            initial_actions=initial_actions,
            time_embeddings=self.constants["action_time_embeddings"],
            adaln_modulations=self.constants["adaln_modulations"],
            output_modulations=self.constants["output_modulations"],
            position_embeddings=self.position_embeddings,
        )
        self._copy_inputs(state, backbone_output, action_input)
        return state

    def _execute(self, state, cache_outputs=True):
        backbone_features = self.action_head.vlln(state.backbone_features)
        if self.vl_sa_engine is not None:
            backbone_features = backbone_features.to(self.vl_sa_engine.dtype_of("hidden_states"))
            self.vl_sa_engine.set_runtime_tensor_shape("hidden_states", backbone_features.shape)
            vl_embs = self.vl_sa_engine(backbone_features, cache_outputs=cache_outputs)["output"]
        else:
            vl_embs = self.action_head.vl_self_attention(backbone_features)

        vl_embs = vl_embs.to(self.cross_kv_engine.dtype_of("vl_embs"))
        self.cross_kv_engine.set_runtime_tensor_shape("vl_embs", vl_embs.shape)
        cross_kv = self.cross_kv_engine(vl_embs, cache_outputs=cache_outputs)

        self.state_encoder_engine.set_runtime_tensor_shape("state", state.state.shape)
        self.state_encoder_engine.set_runtime_tensor_shape(
            "embodiment_id", state.embodiment_id.shape
        )
        state_features = self.state_encoder_engine(
            state.state, state.embodiment_id, cache_outputs=cache_outputs
        )["output"]

        actions = state.actions
        actions.copy_(state.initial_actions)
        dit_dtype = self.dit_engine.dtype_of("sa_embs")
        time_delta = 1.0 / self.action_head.num_inference_timesteps
        for step in range(self.action_head.num_inference_timesteps):
            timestep_embedding = state.time_embeddings[step]
            self.action_encoder_engine.set_runtime_tensor_shape("actions", actions.shape)
            self.action_encoder_engine.set_runtime_tensor_shape(
                "timestep_embedding", timestep_embedding.shape
            )
            self.action_encoder_engine.set_runtime_tensor_shape(
                "embodiment_id", state.embodiment_id.shape
            )
            action_features = self.action_encoder_engine(
                actions,
                timestep_embedding,
                state.embodiment_id,
                cache_outputs=cache_outputs,
            )["output"]
            if state.position_embeddings is not None:
                action_features = action_features + state.position_embeddings

            dit_inputs = {
                "sa_embs": torch.cat((state_features, action_features), dim=1).to(dit_dtype),
                "cross_keys": cross_kv["cross_keys"],
                "cross_values": cross_kv["cross_values"],
                "adaln_modulations": state.adaln_modulations[step],
                "output_modulation": state.output_modulations[step],
                "image_mask": state.image_mask,
                "backbone_attention_mask": state.backbone_attention_mask,
            }
            for name, tensor in dit_inputs.items():
                self.dit_engine.set_runtime_tensor_shape(name, tensor.shape)
            model_output = self.dit_engine(cache_outputs=cache_outputs, **dit_inputs)["output"]

            self.action_decoder_engine.set_runtime_tensor_shape("model_output", model_output.shape)
            self.action_decoder_engine.set_runtime_tensor_shape(
                "embodiment_id", state.embodiment_id.shape
            )
            prediction = self.action_decoder_engine(
                model_output, state.embodiment_id, cache_outputs=cache_outputs
            )["output"]
            actions.add_(prediction[:, -self.action_head.action_horizon :], alpha=time_delta)
        return actions

    def _capture_graph(self, state):
        current_stream = torch.cuda.current_stream()
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                self._execute(state)
        warmup_stream.synchronize()
        current_stream.wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(current_stream)
        with torch.cuda.graph(graph, stream=capture_stream):
            output = self._execute(state)
        current_stream.wait_stream(capture_stream)
        state.graph = graph
        state.output = output

    def close(self):
        if self.closed:
            return
        self.closed = True
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self._graph_state is not None:
            self._graph_state.graph = None
            self._graph_state.output = None
            self._graph_state = None
        gc.collect()
        for engine in self.engines:
            if engine is not None:
                engine.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def engines(self):
        return (
            self.vl_sa_engine,
            self.state_encoder_engine,
            self.action_encoder_engine,
            self.cross_kv_engine,
            self.dit_engine,
            self.action_decoder_engine,
        )


def release_replaced_action_modules(action_head, runner: OptimizedActionHeadRunner):
    """Delete only PyTorch modules replaced by the optimized runner."""
    if runner.vl_sa_engine is not None and hasattr(action_head, "vl_self_attention"):
        del action_head.vl_self_attention
    for name in ("model", "state_encoder", "action_encoder", "action_decoder"):
        if hasattr(action_head, name):
            delattr(action_head, name)
    torch.cuda.empty_cache()


def close_optimized_action_head(policy):
    """Destroy the runner's CUDA graph before its TensorRT execution contexts.

    Engines stored as plain module attributes are closed by
    ``trt_model_forward.close_tensorrt_engines``, which calls this first.
    """
    action_head = getattr(getattr(policy, "model", None), "action_head", None)
    runner = getattr(action_head, "_trt_optimization_runner", None)
    if runner is not None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        runner.close()
        action_head._trt_optimization_runner = None


_COMPONENT_PRECISION_KEYS = {
    "vit": "vit_precision",
    "llm": "llm_precision",
    "dit": "dit_precision",
}
_ENGINE_METADATA_CACHE = {}


def _load_engine_metadata(engine_dir):
    """Load per-component precision metadata.

    The metadata file (``export_metadata.json``) is written by
    ``export_onnx_n1d7.py`` into the ONNX output directory. We look in the
    following locations, in order:

    1. ``<engine_dir>/export_metadata.json``        — standalone-shipped engines
    2. ``<engine_dir>/../onnx/export_metadata.json`` — default build layout
       produced by ``build_trt_pipeline.py``

    Returns the parsed dict, or ``None`` if no metadata is found.
    """
    if os.path.isfile(engine_dir):
        engine_dir = os.path.dirname(os.path.abspath(engine_dir))
    if engine_dir in _ENGINE_METADATA_CACHE:
        return _ENGINE_METADATA_CACHE[engine_dir]

    candidates = [
        os.path.join(engine_dir, "export_metadata.json"),
        os.path.join(os.path.dirname(os.path.abspath(engine_dir)), "onnx", "export_metadata.json"),
    ]
    metadata = None
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                metadata = json.load(f)
            logger.info(f"Loaded engine metadata from {path}")
            break
    _ENGINE_METADATA_CACHE[engine_dir] = metadata
    return metadata


def _engine_path(engine_dir, component, default_precision="bf16"):
    """Build the engine file path for a given component.

    Resolution order:
    1. If ``export_metadata.json`` is present, use the per-component precision
       recorded there (auto-detection). A quantized bundle resolves to that
       name even if the file is missing, so the caller fails with the right
       expected filename instead of silently loading a mismatched engine.
    2. Otherwise, fall back to the legacy BF16 engine names.
    """
    metadata = _load_engine_metadata(engine_dir)
    precision_key = _COMPONENT_PRECISION_KEYS.get(component)
    if metadata is not None and precision_key is not None and precision_key in metadata:
        resolved = os.path.join(engine_dir, f"{component}_{metadata[precision_key]}.engine")
        if os.path.exists(resolved):
            logger.info(
                f"  [{component}] precision={metadata[precision_key]} (from export_metadata.json)"
            )
            return resolved
        if metadata.get("quantization", "none") != "none":
            return resolved

    if component == "vit":
        return _resolve_vit_engine_path(engine_dir)
    return os.path.join(engine_dir, f"{component}_{default_precision}.engine")


def _assert_supports_trt_padding_strip(attention_mask: torch.Tensor) -> None:
    """Hard-fail if the TRT padding-strip path would silently corrupt B>1 batches.

    transformers 4.57+ strips padding tokens before calling ``language_model``.
    The TRT forward path replicates that with ``valid_mask = attention_mask[0] == 1``
    so engine inputs match export-time captured shapes. With B=1 (or B>1 where
    every sample has the same mask, e.g. tiled benchmarking via
    ``verify_n1d7_trt.py --batch_size N``) this is correct. With B>1 *and*
    heterogeneous masks (different valid lengths per sample), samples 1..B-1
    would be silently mis-stripped using sample 0's padding template, producing
    wrong inputs to the LLM TRT engine — no error, no warning, just degraded
    output that's caught only via downstream eval drift.

    Raise rather than warn so the caller can't miss it in the multi-rank log
    torrent. To support heterogeneous B>1, switch to the all-batch
    common-padding mask ``(attention_mask == 1).all(dim=0)`` (only strips
    columns padded across *all* samples) — that is a semantic change requiring
    modeling sign-off, so we hard-fail here for now.

    Note on the build-time batch coupling: the ONNX export pipeline
    (``scripts/deployment/trt/export_onnx_n1d7.py``) bakes the batch dim as a
    static shape — only ``seq_len`` is registered in ``dynamic_axes`` —
    so a TRT engine built at ``--batch-size N`` only accepts B=N at
    runtime. Lifting heterogeneous B>1 inference therefore requires both
    rebuilding the engine at that batch size *and* relaxing this
    assertion. Production today is B=1; this check covers the runtime
    half of that contract.
    """
    if attention_mask.shape[0] <= 1:
        return
    if not (attention_mask == attention_mask[0:1]).all():
        raise ValueError(
            "TRT backbone padding-strip requires B=1 or homogeneous attention "
            f"masks across the batch (got B={attention_mask.shape[0]} with "
            "heterogeneous masks). attention_mask[0] would silently mis-strip "
            "samples 1..B-1 using sample 0's padding template, producing wrong "
            "inputs to the LLM TRT engine. Switch to "
            "(attention_mask == 1).all(dim=0) to only strip columns padded "
            "across all samples — but that is a semantic change requiring "
            "modeling sign-off."
        )


def _resolve_vit_engine_path(trt_engine_path: str) -> str:
    """Locate the ViT engine, tolerating the legacy filename.

    Older builds emitted ``vit_bf16.engine`` even when the source ONNX
    was FP32 — a name that obscured the actual engine precision. New
    builds emit ``vit.engine`` (precision-neutral; the real dtype lives
    in ``export_metadata.json`` and the ONNX tensor types). Existing
    engine directories with the legacy name are still accepted with a
    warning emitted on each call (no per-process suppression: the
    warning content is path-dependent and call sites are few), to
    encourage migration to the precision-neutral name. When no engine
    is present we return the canonical name so the caller can put it
    in any "not found" error.
    """
    new = os.path.join(trt_engine_path, "vit.engine")
    legacy = os.path.join(trt_engine_path, "vit_bf16.engine")
    if os.path.exists(new):
        return new
    if os.path.exists(legacy):
        logger.warning(
            f"Loading ViT engine from legacy path {legacy}; new builds "
            f"emit vit.engine. Rebuild the pipeline to drop the misleading "
            f"precision tag from the filename."
        )
        return legacy
    return new


# ============================================================
# N1.7 Backbone TRT Forward (ViT TRT + LLM TRT)
# ============================================================


def _qwen3_vit_and_scatter(self, vl_input):
    """Shared logic: ViT TRT + embed_tokens + scatter + get_rope_index.

    Returns all inputs needed by either PyTorch LLM or LLM TRT engine.
    These ops stay in PyTorch because they involve dynamic Python logic
    (get_rope_index, masked_scatter, get_placeholder_mask).
    """
    qwen_model = self.model  # Qwen3VLForConditionalGeneration
    inner_model = qwen_model.model  # Qwen3VLModel

    pixel_values = vl_input["pixel_values"]
    grid_thw = vl_input["image_grid_thw"]
    # Scatter into the dtype of the downstream LLM, not the ViT output.
    llm_engine = getattr(self, "llm_engine", None)
    engine_dtype = (
        llm_engine.dtype_of("inputs_embeds")
        if llm_engine is not None
        else self._embedding_layer.weight.dtype
    )

    # The ViT engine baked pos/rotary buffers for the export-time grid; a
    # different runtime grid would silently corrupt vision features.
    assert_grid_thw_matches(
        getattr(self, "_vit_baked_grid_thw", None), grid_thw, source="ViT TRT forward"
    )

    # --- ViT TRT Engine ---
    # Detect ViT engine dtype (FP32 for accuracy or BF16 for speed)
    vit_dtype = self.vit_engine.dtype_of("pixel_values")
    if isinstance(pixel_values, (list, tuple)):
        pv = torch.cat(pixel_values, dim=0)
    else:
        pv = pixel_values
    if pv.dtype != vit_dtype:
        pv = pv.to(vit_dtype)

    self.vit_engine.set_runtime_tensor_shape("pixel_values", pv.shape)
    vit_result = self.vit_engine(pv)
    image_embeds = vit_result["image_embeds"]
    deepstack_features = vit_result.get("deepstack_features")

    # Unpack deepstack: [num_layers, N, D] → list of [N, D]
    deepstack_list = []
    if deepstack_features is not None and deepstack_features.numel() > 1:
        deepstack_list = list(deepstack_features.to(engine_dtype).unbind(0))

    # --- PyTorch: embed_tokens + scatter ---
    input_ids = vl_input["input_ids"]
    inputs_embeds = self._embedding_layer(input_ids)

    if inputs_embeds.dtype != engine_dtype:
        inputs_embeds = inputs_embeds.to(engine_dtype)
    if image_embeds.dtype != engine_dtype:
        image_embeds = image_embeds.to(engine_dtype)

    image_embeds_cat = torch.cat([image_embeds], dim=0)
    image_mask, _ = inner_model.get_placeholder_mask(
        input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds_cat
    )
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_cat)

    visual_pos_masks = image_mask[..., 0] if image_mask is not None else None

    # Compute 3D position IDs (stays in PyTorch — complex Python logic)
    attention_mask = vl_input["attention_mask"]
    position_ids, rope_deltas = inner_model.get_rope_index(
        input_ids, grid_thw, video_grid_thw=None, attention_mask=attention_mask
    )
    inner_model.rope_deltas = rope_deltas

    image_mask_out = input_ids == self._image_token_id
    backbone_attention_mask = attention_mask == 1

    # transformers 4.57+ strips padding tokens before calling language_model internally.
    # Apply the same stripping so TRT engine inputs match export-time captured shapes.
    _assert_supports_trt_padding_strip(attention_mask)
    valid_mask = attention_mask[0] == 1  # [full_seq_len]
    if not valid_mask.all():
        inputs_embeds = inputs_embeds[:, valid_mask, :]
        attention_mask = attention_mask[:, valid_mask]
        position_ids = position_ids[:, :, valid_mask]
        if visual_pos_masks is not None:
            visual_pos_masks = visual_pos_masks[:, valid_mask]
        image_mask_out = image_mask_out[:, valid_mask]
        backbone_attention_mask = backbone_attention_mask[:, valid_mask]

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "visual_pos_masks": visual_pos_masks,
        "deepstack_list": deepstack_list,
        "image_mask_out": image_mask_out,
        "backbone_attention_mask": backbone_attention_mask,
    }


def qwen3_backbone_tensorrt_forward(self, vl_input):
    """Replace Qwen3Backbone.forward() with ViT TRT + PyTorch LLM.

    ViT is replaced with a TRT engine. The LLM stays in PyTorch.
    Used when LLM TRT engine is not available.

    Args:
        self: Qwen3Backbone instance (monkey-patched)
        vl_input: BatchFeature with keys: input_ids, attention_mask, pixel_values, image_grid_thw
    """
    self.set_frozen_modules_to_eval_mode()

    keys_to_use = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
    vl_input = {k: vl_input[k] for k in keys_to_use}

    prepared = _qwen3_vit_and_scatter(self, vl_input)

    qwen_model = self.model
    inner_model = qwen_model.model

    # LLM forward (PyTorch)
    outputs = inner_model.language_model(
        input_ids=None,
        position_ids=prepared["position_ids"],
        attention_mask=prepared["attention_mask"],
        inputs_embeds=prepared["inputs_embeds"],
        visual_pos_masks=prepared["visual_pos_masks"],
        deepstack_visual_embeds=prepared["deepstack_list"] or None,
        output_hidden_states=True,
    )

    return BatchFeature(
        data={
            "backbone_features": outputs.last_hidden_state,
            "backbone_attention_mask": prepared["backbone_attention_mask"],
            "image_mask": prepared["image_mask_out"],
        }
    )


def qwen3_backbone_llm_trt_forward(self, vl_input):
    """Replace Qwen3Backbone.forward() with PyTorch ViT + LLM TRT.

    ViT stays in PyTorch. LLM is replaced with a TRT engine.
    Used when ViT TRT has accuracy issues but LLM TRT is accurate.
    """
    self.set_frozen_modules_to_eval_mode()

    keys_to_use = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
    vl_input = {k: vl_input[k] for k in keys_to_use}

    # Run PyTorch ViT + scatter + rope (original backbone logic up to LLM)
    qwen_model = self.model
    inner_model = qwen_model.model

    # ViT forward (PyTorch — kept for accuracy)
    pixel_values = vl_input["pixel_values"]
    grid_thw = vl_input["image_grid_thw"]
    image_embeds_split, deepstack_image_embeds = inner_model.get_image_features(
        pixel_values, grid_thw
    )
    # get_image_features returns a tuple of per-image tensors; concat for scatter
    image_embeds = torch.cat(list(image_embeds_split), dim=0)

    # Scatter image embeddings into text embeddings
    input_ids = vl_input["input_ids"]
    inputs_embeds = qwen_model.get_input_embeddings()(input_ids)
    image_mask, _ = inner_model.get_placeholder_mask(
        input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
    )
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    visual_pos_masks = image_mask[..., 0] if image_mask is not None else None
    deepstack_list = list(deepstack_image_embeds) if deepstack_image_embeds else []

    # Compute position IDs
    attention_mask = vl_input["attention_mask"]
    position_ids, rope_deltas = inner_model.get_rope_index(
        input_ids, grid_thw, video_grid_thw=None, attention_mask=attention_mask
    )
    inner_model.rope_deltas = rope_deltas

    image_mask_out = input_ids == qwen_model.config.image_token_id
    backbone_attention_mask = attention_mask == 1

    # Strip padding tokens (transformers 4.57+)
    _assert_supports_trt_padding_strip(attention_mask)
    valid_mask = attention_mask[0] == 1
    if not valid_mask.all():
        inputs_embeds = inputs_embeds[:, valid_mask, :]
        attention_mask = attention_mask[:, valid_mask]
        position_ids = position_ids[:, :, valid_mask]
        if visual_pos_masks is not None:
            visual_pos_masks = visual_pos_masks[:, valid_mask]
        image_mask_out = image_mask_out[:, valid_mask]
        backbone_attention_mask = backbone_attention_mask[:, valid_mask]

    # LLM forward (TRT)
    llm_float_dtype = self.llm_engine.dtype_of("inputs_embeds")

    if inputs_embeds.dtype != llm_float_dtype:
        inputs_embeds = inputs_embeds.to(llm_float_dtype)
    if attention_mask.dtype != torch.int64:
        attention_mask = attention_mask.to(torch.int64)
    if position_ids.dtype != torch.int64:
        position_ids = position_ids.to(torch.int64)

    self.llm_engine.set_runtime_tensor_shape("inputs_embeds", inputs_embeds.shape)
    self.llm_engine.set_runtime_tensor_shape("attention_mask", attention_mask.shape)
    self.llm_engine.set_runtime_tensor_shape("position_ids", position_ids.shape)

    llm_kwargs = {}
    if visual_pos_masks is not None and deepstack_list:
        self.llm_engine.set_runtime_tensor_shape("visual_pos_masks", visual_pos_masks.shape)
        llm_kwargs["visual_pos_masks"] = visual_pos_masks
        for i, ds in enumerate(deepstack_list):
            name = f"deepstack_{i}"
            if ds.dtype != llm_float_dtype:
                ds = ds.to(llm_float_dtype)
            self.llm_engine.set_runtime_tensor_shape(name, ds.shape)
            llm_kwargs[name] = ds

    backbone_features = self.llm_engine(inputs_embeds, attention_mask, position_ids, **llm_kwargs)[
        "embeddings"
    ]

    # A retained PyTorch action head needs its parameter dtype at this boundary.
    # Full TRT keeps the LLM engine output dtype for downstream engine adapters.
    llm_out_dtype = getattr(self, "_trt_action_head_dtype", self.llm_engine.out_meta[0][2])
    if backbone_features.dtype != llm_out_dtype:
        backbone_features = backbone_features.to(llm_out_dtype)

    return BatchFeature(
        data={
            "backbone_features": backbone_features,
            "backbone_attention_mask": backbone_attention_mask,
            "image_mask": image_mask_out,
        }
    )


def qwen3_backbone_full_trt_forward(self, vl_input):
    """Replace Qwen3Backbone.forward() with ViT TRT + LLM TRT.

    Both ViT and LLM are replaced with TRT engines.
    PyTorch ops kept: embed_tokens, masked_scatter, get_rope_index (lightweight).

    Args:
        self: Qwen3Backbone instance (monkey-patched)
        vl_input: BatchFeature with keys: input_ids, attention_mask, pixel_values, image_grid_thw
    """
    self.set_frozen_modules_to_eval_mode()

    keys_to_use = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
    vl_input = {k: vl_input[k] for k in keys_to_use}

    prepared = _qwen3_vit_and_scatter(self, vl_input)

    inputs_embeds = prepared["inputs_embeds"]
    attention_mask = prepared["attention_mask"]
    position_ids = prepared["position_ids"]

    # Detect LLM engine's expected float dtype from its first input binding.
    # Handles both BF16 engines (default) and FP32 engines gracefully.
    llm_float_dtype = self.llm_engine.dtype_of("inputs_embeds")

    if inputs_embeds.dtype != llm_float_dtype:
        inputs_embeds = inputs_embeds.to(llm_float_dtype)
    if attention_mask.dtype != torch.int64:
        attention_mask = attention_mask.to(torch.int64)
    if position_ids.dtype != torch.int64:
        position_ids = position_ids.to(torch.int64)

    # Set LLM engine input shapes
    self.llm_engine.set_runtime_tensor_shape("inputs_embeds", inputs_embeds.shape)
    self.llm_engine.set_runtime_tensor_shape("attention_mask", attention_mask.shape)
    self.llm_engine.set_runtime_tensor_shape("position_ids", position_ids.shape)

    llm_kwargs = {}

    # Visual pos masks and deepstack features
    visual_pos_masks = prepared["visual_pos_masks"]
    deepstack_list = prepared["deepstack_list"]

    if visual_pos_masks is not None and deepstack_list:
        self.llm_engine.set_runtime_tensor_shape("visual_pos_masks", visual_pos_masks.shape)
        llm_kwargs["visual_pos_masks"] = visual_pos_masks

        for i, ds in enumerate(deepstack_list):
            name = f"deepstack_{i}"
            if ds.dtype != llm_float_dtype:
                ds = ds.to(llm_float_dtype)
            self.llm_engine.set_runtime_tensor_shape(name, ds.shape)
            llm_kwargs[name] = ds

    backbone_features = self.llm_engine(inputs_embeds, attention_mask, position_ids, **llm_kwargs)[
        "embeddings"
    ]

    # A retained PyTorch action head needs its parameter dtype at this boundary.
    # Full TRT keeps the LLM engine output dtype for downstream engine adapters.
    llm_out_dtype = getattr(self, "_trt_action_head_dtype", self.llm_engine.out_meta[0][2])
    if backbone_features.dtype != llm_out_dtype:
        backbone_features = backbone_features.to(llm_out_dtype)

    return BatchFeature(
        data={
            "backbone_features": backbone_features,
            "backbone_attention_mask": prepared["backbone_attention_mask"],
            "image_mask": prepared["image_mask_out"],
        }
    )


# ============================================================
# Action Head TRT Forward
# ============================================================


def action_head_tensorrt_forward(self, backbone_output, action_input, options=None):
    """Replace ActionHead.get_action() with TRT-accelerated inference.
    VLLN (LayerNorm) stays in PyTorch. State Encoder, Action Encoder,
    DiT, and Action Decoder are replaced with TRT engines.

    N1.7 change: state is reshaped from [B, state_history_length, max_state_dim]
    to [B, 1, state_history_length * max_state_dim] before the state encoder.

    Args:
        self: ActionHead instance (monkey-patched)
        backbone_output: BatchFeature with backbone_features, backbone_attention_mask, image_mask
        action_input: BatchFeature with state, embodiment_id
    """
    # --- VLLN (PyTorch) + vl_self_attention (TRT if available, else PyTorch) ---
    backbone_features = backbone_output.backbone_features
    # Keep dtype consistent with the PyTorch LayerNorm parameters to avoid
    # fp16/bf16 mismatch when TRT backbone output precision differs.
    vlln_param = next(self.vlln.parameters(), None)
    vlln_dtype = vlln_param.dtype if vlln_param is not None else backbone_features.dtype
    if backbone_features.dtype != vlln_dtype:
        backbone_features = backbone_features.to(vlln_dtype)
    backbone_features = self.vlln(backbone_features)
    if hasattr(self, "vl_sa_engine") and self.vl_sa_engine is not None:
        engine_dtype = self.vl_sa_engine.in_meta[0][2]
        if backbone_features.dtype != engine_dtype:
            backbone_features = backbone_features.to(engine_dtype)
        self.vl_sa_engine.set_runtime_tensor_shape("hidden_states", backbone_features.shape)
        backbone_features = self.vl_sa_engine(backbone_features)["output"]
    else:
        backbone_features = self.vl_self_attention(backbone_features)
    vl_embs = backbone_features

    embodiment_id = action_input.embodiment_id
    batch_size = vl_embs.shape[0]
    device = vl_embs.device

    # Derive expected dtypes from the loaded TRT engines
    dit_dtype = self.dit_engine.in_meta[0][2]  # sa_embs dtype
    aux_dtype = self.state_encoder_engine.in_meta[0][2]  # state dtype

    # Ensure consistent dtypes
    if vl_embs.dtype != dit_dtype:
        vl_embs = vl_embs.to(dit_dtype)
    if action_input.state.dtype != aux_dtype:
        action_input.state = action_input.state.to(aux_dtype)
    if embodiment_id.dtype != torch.int64:
        embodiment_id = embodiment_id.to(torch.int64)

    # --- State history reshape (N1.7) ---
    # N1.7: state comes as [B, state_history_length, max_state_dim]
    # Flatten to [B, 1, state_history_length * max_state_dim] for the encoder
    state = action_input.state
    if state.ndim == 3 and state.shape[1] > 1:
        state = state.view(state.shape[0], 1, -1)
    elif state.ndim == 3 and state.shape[1] == 1:
        # Already [B, 1, dim] — state_history_length=1
        pass
    else:
        # Unexpected shape, pass through
        logger.warning(f"Unexpected state shape: {state.shape}")

    # --- State Encoder TRT ---
    self.state_encoder_engine.set_runtime_tensor_shape("state", state.shape)
    self.state_encoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
    state_features = self.state_encoder_engine(state, embodiment_id)["output"]

    # --- Initialize actions as random noise ---
    if hasattr(self, "init_actions"):
        actions = self.init_actions.expand((batch_size, -1, -1))
    else:
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=aux_dtype,
            device=device,
        )

    num_steps = self.num_inference_timesteps
    dt = 1.0 / num_steps

    # --- Denoising loop ---
    for t in range(num_steps):
        t_cont = t / float(num_steps)
        t_discretized = int(t_cont * self.num_timestep_buckets)

        timesteps_tensor = torch.full(
            size=(batch_size,), fill_value=t_discretized, device=device, dtype=torch.int64
        )

        # Action Encoder TRT
        self.action_encoder_engine.set_runtime_tensor_shape("actions", actions.shape)
        self.action_encoder_engine.set_runtime_tensor_shape("timesteps", timesteps_tensor.shape)
        self.action_encoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
        action_features = self.action_encoder_engine(
            actions.to(aux_dtype), timesteps_tensor, embodiment_id
        )["output"]

        # Maybe add position embedding (stays in PyTorch)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0).to(aux_dtype)
            action_features = action_features + pos_embs

        # Concatenate state + action embeddings
        sa_embs = torch.cat((state_features, action_features), dim=1).to(dit_dtype)

        # DiT TRT
        self.dit_engine.set_runtime_tensor_shape("sa_embs", sa_embs.shape)
        self.dit_engine.set_runtime_tensor_shape("vl_embs", vl_embs.shape)
        self.dit_engine.set_runtime_tensor_shape("timestep", timesteps_tensor.shape)

        dit_kwargs = {}
        if hasattr(backbone_output, "image_mask") and backbone_output.image_mask is not None:
            image_mask = backbone_output.image_mask
            self.dit_engine.set_runtime_tensor_shape("image_mask", image_mask.shape)
            dit_kwargs["image_mask"] = image_mask

        if (
            hasattr(backbone_output, "backbone_attention_mask")
            and backbone_output.backbone_attention_mask is not None
        ):
            bb_mask = backbone_output.backbone_attention_mask
            self.dit_engine.set_runtime_tensor_shape("backbone_attention_mask", bb_mask.shape)
            dit_kwargs["backbone_attention_mask"] = bb_mask

        model_output = self.dit_engine(sa_embs, vl_embs, timesteps_tensor, **dit_kwargs)["output"]

        # Action Decoder TRT
        self.action_decoder_engine.set_runtime_tensor_shape("model_output", model_output.shape)
        self.action_decoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
        pred = self.action_decoder_engine(model_output, embodiment_id)["output"]
        pred_velocity = pred[:, -self.action_horizon :]

        # Euler integration
        actions = actions + dt * pred_velocity

    return BatchFeature(data={"action_pred": actions})


# ============================================================
# Engine Setup
# ============================================================


# Engines each mode swaps in with no PyTorch fallback: their absence must fail
# fast with a build hint rather than a bare FileNotFoundError from Engine.load.
_ACTION_HEAD_STATIC_ENGINES = (
    "state_encoder.engine",
    "action_encoder.engine",
    "action_decoder.engine",
)


def _required_action_head_engines(engine_dir: str) -> tuple[str, ...]:
    required = [
        *_ACTION_HEAD_STATIC_ENGINES,
        os.path.basename(_engine_path(engine_dir, "dit")),
    ]
    metadata = _load_engine_metadata(engine_dir)
    artifacts = OptimizationArtifacts.from_metadata(engine_dir, metadata, artifact_kind="engine")
    if artifacts is not None:
        required.extend(
            (os.path.basename(artifacts.cross_kv), os.path.basename(artifacts.constants))
        )
    return tuple(required)


def setup_tensorrt_engines(policy, trt_engine_path, mode=InferenceMode.n17_full_pipeline):
    """Load a validated TRT engine bundle and patch the selected inference path.

    Per-component precision and optional optimized-action-head artifacts are
    auto-detected from export_metadata.json. Legacy bundles without metadata
    fall back to the historical BF16 engine names.
    """
    mode = InferenceMode(mode)
    metadata = _load_engine_metadata(trt_engine_path)
    if metadata is not None:
        logger.info(
            "Auto-detected component precisions: vit=%s, llm=%s, dit=%s",
            metadata.get("vit_precision"),
            metadata.get("llm_precision"),
            metadata.get("dit_precision"),
        )
    else:
        logger.info(
            "No export_metadata.json found in %s; using legacy BF16 engine names.",
            trt_engine_path,
        )
    print(format_quantization_summary(metadata))

    if mode in (InferenceMode.n17_full_pipeline, InferenceMode.action_head):
        assert_engine_bundle_present(
            trt_engine_path,
            _required_action_head_engines(trt_engine_path),
            mode=mode.value,
            source=f"setup_tensorrt_engines({mode})",
        )

    checked_metadata = assert_engine_matches_policy(
        policy, trt_engine_path, source=f"setup_tensorrt_engines({mode})"
    )
    policy.model.backbone._vit_baked_grid_thw = (
        checked_metadata.get("vit_grid_thw") if checked_metadata else None
    )
    _INFERENCE_MODE_DISPATCH[mode](policy, trt_engine_path)


def close_tensorrt_engines(policy):
    """Release every TRT ``Engine`` attached to ``policy`` (best-effort).

    setup_tensorrt_engines stashes engines as plain attributes on the backbone
    and action head (which subset depends on the mode), so walk the model's
    submodules and close each. Needed because the sim-eval entrypoint hard-exits
    via os._exit, which skips ``Engine.__del__`` and would otherwise leak GPU
    memory into the next eval shard.
    """
    # The sim wrapper exposes the inner Gr00tPolicy as ``.policy``; a remote
    # PolicyClient has no model and no engines to close.
    inner_policy = getattr(policy, "policy", policy)
    model = getattr(inner_policy, "model", None)
    if model is None:
        return
    # Destroy the optimized runner's CUDA graph (and its engines) before the
    # remaining TensorRT execution contexts.
    close_optimized_action_head(inner_policy)
    closed = set()
    for module in model.modules():
        # Snapshot the attribute values: close() runs C++ destructors, so avoid
        # iterating a live __dict__ view.
        for value in list(vars(module).values()):
            if isinstance(value, Engine) and id(value) not in closed:
                closed.add(id(value))
                try:
                    value.close()
                except Exception as e:
                    print(f"Failed to close TRT engine: {e}")


@contextmanager
def closing_tensorrt_engines(policy):
    """``with`` form of :func:`close_tensorrt_engines`, scoped to ``policy``.

    Closes only the engines attached to ``policy`` on block exit (including the
    exception path), so it never touches engines owned by another live policy.
    """
    try:
        yield policy
    finally:
        close_tensorrt_engines(policy)


def _setup_n17_full_pipeline(policy, trt_engine_path):
    """Set up TRT engines for N1.7: ViT TRT + LLM TRT + Action Head TRT.

    The Qwen3-VL backbone's vision encoder and text model are both replaced
    with TRT engines. PyTorch ops kept: embed_tokens, masked_scatter,
    get_rope_index (lightweight, <1ms).

    Falls back to PyTorch LLM if the LLM engine is not found.
    """
    backbone = policy.model.backbone
    qwen_model = backbone.model  # Qwen3VLForConditionalGeneration
    action_head = policy.model.action_head

    # --- Backbone setup ---
    # Save references needed by the TRT forward
    backbone._embedding_layer = qwen_model.model.language_model.get_input_embeddings()
    backbone._image_token_id = qwen_model.config.image_token_id

    # Load ViT TRT engine (optional — PyTorch ViT used as fallback for accuracy)
    vit_engine_path = _engine_path(trt_engine_path, "vit")
    use_vit_trt = os.path.exists(vit_engine_path)
    if use_vit_trt:
        print(f"Loading ViT engine: {vit_engine_path}")
        backbone.vit_engine = Engine(vit_engine_path)
        del qwen_model.model.visual
        torch.cuda.empty_cache()
        print("  Deleted PyTorch ViT (replaced by TRT engine)")
    else:
        backbone.vit_engine = None
        print(f"  ViT engine not found at {vit_engine_path}, keeping PyTorch ViT")

    # Load LLM TRT engine (if available)
    llm_engine_path = _engine_path(trt_engine_path, "llm")
    use_llm_trt = os.path.exists(llm_engine_path)

    if use_llm_trt:
        print(f"Loading LLM engine: {llm_engine_path}")
        backbone.llm_engine = Engine(llm_engine_path)

        # Delete PyTorch LLM layers to free GPU memory
        # Keep embed_tokens (needed for token embedding before TRT)
        # Keep get_rope_index via inner_model (needed for position IDs)
        del qwen_model.model.language_model.layers
        del qwen_model.model.language_model.norm
        torch.cuda.empty_cache()
        print("  Deleted PyTorch LLM layers (replaced by TRT engine)")
    else:
        backbone.llm_engine = None
        print(f"  LLM engine not found at {llm_engine_path}, using PyTorch LLM")

    # Monkey-patch backbone forward
    if use_vit_trt and use_llm_trt:
        backbone.forward = partial(qwen3_backbone_full_trt_forward, backbone)
    elif use_vit_trt and not use_llm_trt:
        backbone.forward = partial(qwen3_backbone_tensorrt_forward, backbone)
    elif not use_vit_trt and use_llm_trt:
        # PyTorch ViT + LLM TRT (best accuracy when ViT TRT has issues)
        backbone.forward = partial(qwen3_backbone_llm_trt_forward, backbone)
    else:
        print("  No backbone TRT engines loaded, backbone remains in PyTorch")

    # --- Action head setup ---
    optimized_runner = OptimizedActionHeadRunner.maybe_create(
        action_head,
        trt_engine_path,
        _load_engine_metadata(trt_engine_path),
        dit_engine_path=_engine_path(trt_engine_path, "dit"),
    )
    if optimized_runner is not None:
        release_replaced_action_modules(action_head, optimized_runner)
        action_head._trt_optimization_runner = optimized_runner
        action_head.get_action = optimized_runner
        llm_status = "TRT" if use_llm_trt else "PyTorch"
        vit_status = "TRT" if backbone.vit_engine else "PyTorch"
        print("N1.7 optimized full-pipeline TRT engines loaded.")
        print(f"  ViT: {vit_status} | LLM: {llm_status} | Action Head: TRT CUDA graph")
        return
    action_head._trt_optimization_runner = None

    # Load vl_self_attention TRT engine (if available)
    vl_sa_engine_path = os.path.join(trt_engine_path, "vl_self_attention.engine")
    if os.path.exists(vl_sa_engine_path):
        print(f"Loading VL Self-Attention engine: {vl_sa_engine_path}")
        action_head.vl_sa_engine = Engine(vl_sa_engine_path)
        # Delete PyTorch module — TRT engine replaces it
        if hasattr(action_head, "vl_self_attention"):
            del action_head.vl_self_attention
            torch.cuda.empty_cache()
        print("  Deleted PyTorch vl_self_attention (replaced by TRT engine)")
    else:
        action_head.vl_sa_engine = None
        print(f"  VL Self-Attention engine not found at {vl_sa_engine_path}, using PyTorch")

    if hasattr(action_head, "model"):
        del action_head.model
    if hasattr(action_head, "state_encoder"):
        del action_head.state_encoder
    if hasattr(action_head, "action_encoder"):
        del action_head.action_encoder
    if hasattr(action_head, "action_decoder"):
        del action_head.action_decoder
    torch.cuda.empty_cache()

    assert action_head.action_dim == action_head.config.max_action_dim

    print(f"Loading action head engines from: {trt_engine_path}")
    action_head.state_encoder_engine = Engine(os.path.join(trt_engine_path, "state_encoder.engine"))
    action_head.action_encoder_engine = Engine(
        os.path.join(trt_engine_path, "action_encoder.engine")
    )
    dit_path = _engine_path(trt_engine_path, "dit")
    print(f"Loading DiT engine: {dit_path}")
    action_head.dit_engine = Engine(dit_path)
    action_head.action_decoder_engine = Engine(
        os.path.join(trt_engine_path, "action_decoder.engine")
    )

    action_head.get_action = partial(action_head_tensorrt_forward, action_head)

    llm_status = "TRT" if use_llm_trt else "PyTorch"
    vit_status = "TRT" if backbone.vit_engine else "PyTorch"
    print("N1.7 full-pipeline TRT engines loaded.")
    print(f"  ViT: {vit_status} | LLM: {llm_status} | Action Head: TRT")


def _setup_vit_llm_only(policy, trt_engine_path):
    """Set up TRT engines for ViT + LLM only; action head stays in PyTorch.

    Use this on platforms where DiT cannot be exported with dynamic vl_seq_len
    (e.g. DGX Spark / torch 2.10 dynamo exporter bakes seq_len as static).
    The backbone (ViT TRT + LLM TRT) still gets TRT acceleration; the PyTorch
    action head receives the LLM embeddings at the actual runtime seq_len
    without any shape constraint.
    """
    backbone = policy.model.backbone
    # This mode retains the PyTorch action head and must restore its dtype.
    backbone._trt_action_head_dtype = next(policy.model.action_head.parameters()).dtype
    qwen_model = backbone.model  # Qwen3VLForConditionalGeneration

    # Save references needed by the TRT forward
    backbone._embedding_layer = qwen_model.model.language_model.get_input_embeddings()
    backbone._image_token_id = qwen_model.config.image_token_id

    # Load ViT TRT engine
    vit_engine_path = _engine_path(trt_engine_path, "vit")
    if not os.path.exists(vit_engine_path):
        raise FileNotFoundError(
            f"ViT TRT engine not found: {vit_engine_path}\n"
            f"Run export_onnx_n1d7.py + build_tensorrt_engine.py first."
        )
    print(f"Loading ViT engine: {vit_engine_path}")
    backbone.vit_engine = Engine(vit_engine_path)
    del qwen_model.model.visual
    torch.cuda.empty_cache()
    print("  Deleted PyTorch ViT (replaced by TRT engine)")

    # Load LLM TRT engine
    llm_engine_path = _engine_path(trt_engine_path, "llm")
    if not os.path.exists(llm_engine_path):
        raise FileNotFoundError(
            f"LLM TRT engine not found: {llm_engine_path}\n"
            f"Run export_onnx_n1d7.py + build_tensorrt_engine.py first."
        )
    print(f"Loading LLM engine: {llm_engine_path}")
    backbone.llm_engine = Engine(llm_engine_path)
    del qwen_model.model.language_model.layers
    del qwen_model.model.language_model.norm
    torch.cuda.empty_cache()
    print("  Deleted PyTorch LLM layers (replaced by TRT engine)")

    # Patch backbone forward to use ViT TRT + LLM TRT
    backbone.forward = partial(qwen3_backbone_full_trt_forward, backbone)

    print("vit_llm_only TRT engines loaded.")
    print("  ViT: TRT | LLM: TRT | Action Head: PyTorch")


def _setup_action_head(policy, trt_engine_path):
    """Set up TRT engines for action head only (N1.7 mode).

    Backbone (Qwen3-VL) stays in PyTorch. Only the 4 action head components
    (State Encoder, Action Encoder, DiT, Action Decoder) are replaced with
    TRT engines.
    """
    action_head = policy.model.action_head
    optimized_runner = OptimizedActionHeadRunner.maybe_create(
        action_head,
        trt_engine_path,
        _load_engine_metadata(trt_engine_path),
        dit_engine_path=_engine_path(trt_engine_path, "dit"),
    )
    if optimized_runner is not None:
        release_replaced_action_modules(action_head, optimized_runner)
        action_head._trt_optimization_runner = optimized_runner
        action_head.get_action = optimized_runner
        print("Optimized action-head TRT engines loaded with CUDA graph replay.")
        print("  Backbone remains in PyTorch (Qwen3-VL).")
        return
    action_head._trt_optimization_runner = None

    # Delete PyTorch modules that are replaced by TRT
    if hasattr(action_head, "model"):
        del action_head.model
    if hasattr(action_head, "state_encoder"):
        del action_head.state_encoder
    if hasattr(action_head, "action_encoder"):
        del action_head.action_encoder
    if hasattr(action_head, "action_decoder"):
        del action_head.action_decoder
    torch.cuda.empty_cache()

    # Verify action_dim consistency
    assert action_head.action_dim == action_head.config.max_action_dim, (
        f"action_dim mismatch: action_head.action_dim={action_head.action_dim} "
        f"!= config.max_action_dim={action_head.config.max_action_dim}"
    )

    # Load action head TRT engines
    print(f"Loading action head engines from: {trt_engine_path}")
    action_head.state_encoder_engine = Engine(os.path.join(trt_engine_path, "state_encoder.engine"))
    action_head.action_encoder_engine = Engine(
        os.path.join(trt_engine_path, "action_encoder.engine")
    )
    dit_path = _engine_path(trt_engine_path, "dit")
    print(f"Loading DiT engine: {dit_path}")
    action_head.dit_engine = Engine(dit_path)
    action_head.action_decoder_engine = Engine(
        os.path.join(trt_engine_path, "action_decoder.engine")
    )

    # Monkey-patch: backbone.forward stays original, only action head is replaced
    action_head.get_action = partial(action_head_tensorrt_forward, action_head)

    print("Action head TRT engines loaded and forward method patched.")
    print("  Backbone remains in PyTorch (Qwen3-VL).")


def _setup_dit_only(policy, trt_engine_path):
    """Set up TRT engine for DiT-only acceleration (backward compatible).

    Only replaces the DiT model in the action head. The backbone and other
    action head components remain in PyTorch.
    """
    action_head = policy.model.action_head

    # Delete the PyTorch DiT model
    if hasattr(action_head, "model"):
        del action_head.model
    torch.cuda.empty_cache()

    # Load DiT TRT engine
    # Support both naming conventions (legacy bundles predate metadata).
    dit_path = (
        trt_engine_path if os.path.isfile(trt_engine_path) else _engine_path(trt_engine_path, "dit")
    )
    if not os.path.exists(dit_path):
        for legacy_name in ("dit_model_bf16.engine", "dit_model_bf16.trt"):
            legacy_path = os.path.join(trt_engine_path, legacy_name)
            if os.path.exists(legacy_path):
                dit_path = legacy_path
                break
    print(f"Loading DiT engine: {dit_path}")
    action_head.dit_engine = Engine(dit_path)

    # Monkey-patch only the get_action method
    # We need a simpler forward that only replaces the DiT call
    @torch.no_grad()
    def dit_only_get_action_with_features(
        backbone_features,
        state_features,
        embodiment_id,
        backbone_output,
        action_input=None,
        options=None,
    ):
        """get_action_with_features with DiT replaced by TRT.

        ``action_input``/``options`` are accepted (and unused) so the patched
        method matches the ``Gr00tN1d7.get_action_with_features`` contract that
        ``get_action`` calls with these keywords; the DiT-only path needs neither.
        """
        vl_embs = backbone_features
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        engine_dtype = action_head.dit_engine.in_meta[0][2]

        actions = torch.randn(
            size=(batch_size, action_head.config.action_horizon, action_head.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        dt = 1.0 / action_head.num_inference_timesteps

        for t in range(action_head.num_inference_timesteps):
            t_cont = t / float(action_head.num_inference_timesteps)
            t_discretized = int(t_cont * action_head.num_timestep_buckets)

            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = action_head.action_encoder(actions, timesteps_tensor, embodiment_id)

            if action_head.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            sa_embs = torch.cat((state_features, action_features), dim=1).to(engine_dtype)

            # Use TRT for DiT
            vl_embs_trt = vl_embs.to(engine_dtype)
            timesteps_trt = timesteps_tensor.to(torch.int64)

            action_head.dit_engine.set_runtime_tensor_shape("sa_embs", sa_embs.shape)
            action_head.dit_engine.set_runtime_tensor_shape("vl_embs", vl_embs_trt.shape)
            action_head.dit_engine.set_runtime_tensor_shape("timestep", timesteps_trt.shape)

            dit_kwargs = {}
            if hasattr(backbone_output, "image_mask") and backbone_output.image_mask is not None:
                image_mask = backbone_output.image_mask
                action_head.dit_engine.set_runtime_tensor_shape("image_mask", image_mask.shape)
                dit_kwargs["image_mask"] = image_mask

            if (
                hasattr(backbone_output, "backbone_attention_mask")
                and backbone_output.backbone_attention_mask is not None
            ):
                bb_mask = backbone_output.backbone_attention_mask
                action_head.dit_engine.set_runtime_tensor_shape(
                    "backbone_attention_mask", bb_mask.shape
                )
                dit_kwargs["backbone_attention_mask"] = bb_mask

            model_output = action_head.dit_engine(
                sa_embs, vl_embs_trt, timesteps_trt, **dit_kwargs
            )["output"]

            # Quantized DiT has FP16 I/O; the retained PyTorch decoder keeps
            # the model dtype (normally BF16).
            pred = action_head.action_decoder(model_output.to(action_features.dtype), embodiment_id)
            pred_velocity = pred[:, -action_head.action_horizon :]
            actions = actions + dt * pred_velocity

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embs,
                "state_features": state_features,
            }
        )

    action_head.get_action_with_features = dit_only_get_action_with_features
    print("DiT-only TRT engine loaded and forward method patched.")


# Maps each InferenceMode to its engine-swap routine. setup_tensorrt_engines
# dispatches through this; a deployment test asserts the keys equal InferenceMode
# exactly, so a new mode can't be added without a matching setup branch.
_INFERENCE_MODE_DISPATCH = {
    InferenceMode.n17_full_pipeline: _setup_n17_full_pipeline,
    InferenceMode.vit_llm_only: _setup_vit_llm_only,
    InferenceMode.action_head: _setup_action_head,
    InferenceMode.dit_only: _setup_dit_only,
}
