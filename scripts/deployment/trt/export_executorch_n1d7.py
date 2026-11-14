#!/usr/bin/env python3

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

"""
Export GR00T N1.7 components as ExecuTorch programs that run TensorRT engines.

This is the ExecuTorch counterpart of ``export_onnx_n1d7.py --export-mode
full_pipeline``. It builds the same component wrappers from the same captured
inputs, traces each with ``torch.export``, compiles it with Torch-TensorRT and
saves it as an ExecuTorch ``.pte`` program. Whatever TensorRT does not take would
run on ExecuTorch's CUDA backend, never on the CPU. Each program runs from C++ with
the ExecuTorch runtime and the TensorRT backend Torch-TensorRT provides for it,
without Python or libtorch, or from Python through
``executorch_model_forward.setup_executorch_programs``. The glue between the
components (token embedding, rope index, VLLN, the denoising loop) is not part
of the programs.

It covers the BF16 baseline profile (no FP8 or NVFP4, no optimized profile), and
for it one step replaces both ``export_onnx_n1d7.py`` and ``build_tensorrt_engine.py``.
TensorRT engines are GPU-architecture-specific, so export on the target device.

Requires PyTorch 2.15, ExecuTorch 1.6 and Torch-TensorRT 2.15 with the
``executorch`` extra, so it needs its own environment while this project pins
PyTorch 2.9 (the deployment README lists the full install):

    pip install "torch_tensorrt[executorch]"

The CUDA backend also needs ``nvcc`` on PATH, but only if TensorRT leaves part of a
component to it.

Usage:
    python -m scripts.deployment.trt.export_executorch_n1d7 \\
        --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \\
        --dataset-path demo_data/libero_demo \\
        --output-dir ./gr00t_executorch_deployment
"""

from dataclasses import dataclass
import inspect
import json
import logging
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Literal, Optional, get_args

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
import torch
import tyro

from .export_onnx_n1d7 import (
    DiTExportWrapper,
    VLAttentionWrapper,
    _patch_dit_attn_for_export,
    _restore_vision_attention,
    _unpatch_dit_attn,
    build_export_metadata,
    build_llm_for_export,
    build_vit_for_export,
    capture_component_inputs,
    dit_export_inputs,
    exports_vl_self_attention,
    llm_export_inputs,
)
from .n1d7_optimization_config import component_precisions, resolve_dtype


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXECUTORCH_IO_KEY = "executorch_io"


def _cuda_partitioner(name):
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner

    # Name the CUDA backend's weights file after the component, or the programs
    # written to one directory overwrite each other's file.
    return CudaPartitioner(
        [
            CudaBackend.generate_method_name_compile_spec("forward"),
            CudaBackend.generate_data_name_compile_spec(name),
        ]
    )


def _gpu_memory_config():
    from executorch.exir import ExecutorchBackendConfig
    from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
    from executorch.exir.passes.propagate_device_pass import PropagateDeviceConfig

    # Inputs and outputs stay in GPU memory, so the glue between components
    # reads and writes CUDA tensors in place, as with the TensorRT engines.
    return ExecutorchBackendConfig(
        propagate_device_config=PropagateDeviceConfig(
            skip_h2d_for_method_inputs=True, skip_d2h_for_method_outputs=True
        ),
        enable_non_cpu_memory_planning=True,
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False, alloc_graph_output=False),
    )


def _dynamic_dims(names, inputs, dynamic_axes, opt_len):
    """Map ONNX-style dynamic axes to torch.export dims and Torch-TensorRT ranges.

    Uses the profile ``build_tensorrt_engine.py`` derives for a named dynamic
    dim: optimum at the captured length, maximum ``max(2 * opt, opt + 64)``.
    """
    import torch_tensorrt

    max_len = max(2 * opt_len, opt_len + 64)
    dims = {}
    shapes, trt_inputs = [], []
    for name, tensor in zip(names, inputs):
        axes = (dynamic_axes or {}).get(name, {})
        spec = {}
        min_shape, max_shape = list(tensor.shape), list(tensor.shape)
        for axis, dim_name in axes.items():
            if dim_name not in dims:
                # A minimum of 2 keeps torch.export from specializing the
                # dimension to the 0/1 sizes it treats as constants.
                dims[dim_name] = torch.export.Dim(dim_name, min=2, max=max_len)
            spec[axis] = dims[dim_name]
            min_shape[axis], max_shape[axis] = 2, max_len
        shapes.append(spec or None)
        if spec:
            trt_inputs.append(
                torch_tensorrt.Input(
                    min_shape=min_shape,
                    opt_shape=list(tensor.shape),
                    max_shape=max_shape,
                    dtype=tensor.dtype,
                )
            )
        else:
            trt_inputs.append(torch_tensorrt.Input(shape=list(tensor.shape), dtype=tensor.dtype))
    return tuple(shapes), trt_inputs


def _nest_varargs_shapes(module, flat_shapes):
    """Group the shapes of the inputs bound to ``*args`` (the DiT masks) into one tuple.

    torch.export matches ``dynamic_shapes`` to the forward signature, where those
    inputs are one argument. With no such inputs the shapes stay flat.
    """
    params = inspect.signature(module.forward).parameters.values()
    n = sum(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params)
    if len(flat_shapes) > n and any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
        return (*flat_shapes[:n], tuple(flat_shapes[n:]))
    return flat_shapes


def export_component(
    name,
    module,
    inputs,
    input_names,
    output_names,
    output_dir,
    *,
    dynamic_axes=None,
    opt_len=None,
    workspace_size=None,
    disable_tf32=False,
):
    """Trace ``module``, compile it with Torch-TensorRT and save ``<name>.pte``.

    Returns the I/O description the runtime binds tensors by.
    """
    import torch_tensorrt

    logger.info(f"  Tracing {name}...")
    start = time.time()
    with torch.no_grad():
        if dynamic_axes:
            flat_shapes, arg_inputs = _dynamic_dims(input_names, inputs, dynamic_axes, opt_len)
            shapes = _nest_varargs_shapes(module, flat_shapes)
        else:
            flat_shapes = shapes = None
            arg_inputs = list(inputs)
        program = torch.export.export(
            module, tuple(inputs), dynamic_shapes=shapes, strict=False
        ).run_decompositions({})

    options = {
        "min_block_size": 1,
        # A TensorRT plugin that needs libtorch could not run in ExecuTorch.
        "fallback_pytorch_plugin_ops": True,
        "truncate_double": True,
        # Model weights wait in host memory while TensorRT builds.
        "offload_module_to_cpu": True,
        # The default leaves TF32 on for FP32 layers, as build_tensorrt_engine.py does.
        "disable_tf32": disable_tf32,
    }
    if workspace_size is not None:
        options["workspace_size"] = workspace_size
    logger.info(f"  Building TensorRT engines for {name}...")
    output_dtypes = [
        node.meta["val"].dtype for node in program.graph.find_nodes(op="output")[0].args[0]
    ]
    if len(output_dtypes) != len(output_names):
        raise RuntimeError(f"{name} returns {len(output_dtypes)} tensors, not {list(output_names)}")
    trt_module = torch_tensorrt.dynamo.compile(program, arg_inputs=arg_inputs, **options)
    # A file left by an earlier export of this component would be recorded as this run's.
    for f in os.listdir(output_dir):
        if f.startswith(f"{name}.") and f.endswith((".pte", ".ptd")):
            os.remove(os.path.join(output_dir, f))
    path = os.path.join(output_dir, f"{name}.pte")
    torch_tensorrt.save(
        trt_module,
        path,
        output_format="executorch",
        arg_inputs=list(inputs),
        # The compiled module takes its inputs flat.
        dynamic_shapes=flat_shapes,
        use_cuda_graphs=dynamic_axes is None,
        # What TensorRT leaves runs on the GPU too, never on a CPU kernel.
        partitioners=[_cuda_partitioner(name)],
        backend_config=_gpu_memory_config(),
    )
    elapsed = time.time() - start
    files = [f for f in os.listdir(output_dir) if f.startswith(f"{name}.")]
    size_mb = sum(os.path.getsize(os.path.join(output_dir, f)) for f in files) / 2**20
    logger.info(f"  {name}: {sorted(files)}, {size_mb:.1f} MB, {elapsed:.0f} s")
    # Weights the CUDA backend keeps outside the program, if TensorRT left it any work.
    data_files = [f for f in files if f.endswith(".ptd")]
    if len(data_files) > 1:
        raise RuntimeError(f"{name} wrote more than one data file: {data_files}")
    return {
        "file": f"{name}.pte",
        "data_file": data_files[0] if data_files else None,
        "inputs": [
            {"name": n, "shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch.")}
            for n, t in zip(input_names, inputs)
        ],
        "outputs": [
            {"name": n, "dtype": str(dtype).removeprefix("torch.")}
            for n, dtype in zip(output_names, output_dtypes)
        ],
        "dynamic_axes": dynamic_axes or {},
        "export_seconds": round(elapsed, 1),
    }


def _check_output_dir(output_dir, components):
    """Create ``output_dir``, refusing one whose files the loader could mix with this export."""
    os.makedirs(output_dir, exist_ok=True)
    if any(f.endswith((".engine", ".onnx")) for f in os.listdir(output_dir)):
        raise ValueError(
            f"{output_dir} holds TensorRT engines or ONNX files; use a fresh --output-dir."
        )
    metadata_path = os.path.join(output_dir, "export_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            if EXECUTORCH_IO_KEY not in json.load(f):
                raise ValueError(f"{output_dir} holds another export; use a fresh --output-dir.")
        if set(components) != set(get_args(Component)):
            raise ValueError(
                f"{output_dir} holds an export that a subset would replace; use a fresh --output-dir."
            )
    return metadata_path


def main(args):
    args.embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    precisions = component_precisions("none")
    vit_dtype = resolve_dtype(precisions["vit"])
    llm_dtype = resolve_dtype(precisions["llm"])
    dit_dtype = resolve_dtype(precisions["dit"])
    aux_dtype = resolve_dtype(precisions["aux"])
    bs = args.batch_size
    dynamic = args.seq_len == "dynamic"
    # Fail in the wrong environment before loading the policy.
    import executorch.backends.cuda.cuda_partitioner  # noqa: F401
    import executorch.exir  # noqa: F401
    import torch_tensorrt  # noqa: F401

    metadata_path = _check_output_dir(args.output_dir, args.components)

    logger.info("=" * 80)
    logger.info("GR00T N1.7 ExecuTorch + TensorRT Export")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Embodiment: {args.embodiment_tag}")
    logger.info(f"Sequence length: {args.seq_len}, batch size: {bs}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)

    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag, model_path=args.model_path, device="cuda"
    )
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path, modality_configs=policy.get_modality_config()
    )
    dit_capture, vit_capture, llm_capture = capture_component_inputs(
        policy, dataset, "full_pipeline"
    )
    if dit_capture is None:
        raise RuntimeError("Failed to capture component inputs")
    # build_export_metadata also describes ONNX exports; this one is always the BF16 baseline.
    metadata_args = SimpleNamespace(
        **vars(args),
        export_mode="full_pipeline",
        precision="bf16",
        quantization="none",
        execution_profile="baseline",
    )
    metadata = build_export_metadata(policy, metadata_args, dit_capture, vit_capture, llm_capture)
    vl_seq_len = metadata["vl_seq_len"]
    llm_seq_len = metadata["llm_seq_len"]
    action_head = policy.model.action_head
    config = action_head.config
    common = {
        "output_dir": args.output_dir,
        "workspace_size": None if args.workspace is None else args.workspace << 20,
        "disable_tf32": args.disable_tf32,
    }
    # The metadata is written again only after every requested component exports, so
    # a failed run never leaves programs from two exports that the loader accepts.
    if os.path.exists(metadata_path):
        os.remove(metadata_path)
    io = {}

    # ViT: pixel_values only, the grid is baked into the wrapper's buffers.
    if "vit" in args.components:
        wrapper, pixel_values, vision, originals = build_vit_for_export(
            policy, vit_capture, vit_dtype, bs
        )
        try:
            io["vit"] = export_component(
                "vit",
                wrapper,
                [pixel_values],
                ["pixel_values"],
                ["image_embeds", "deepstack_features"],
                **common,
            )
        finally:
            _restore_vision_attention(vision, originals)
        del wrapper
        torch.cuda.empty_cache()

    if "llm" in args.components:
        wrapper = build_llm_for_export(policy, llm_capture, llm_dtype)
        names, inputs, axes = llm_export_inputs(llm_capture, bs, llm_dtype, "cuda")
        axes.pop("embeddings", None)
        io["llm"] = export_component(
            f"llm_{precisions['llm']}",
            wrapper,
            inputs,
            names,
            ["embeddings"],
            dynamic_axes=axes if dynamic else None,
            opt_len=llm_seq_len,
            **common,
        )
        del wrapper
        torch.cuda.empty_cache()

    if "vl_self_attention" in args.components and exports_vl_self_attention(action_head):
        model = VLAttentionWrapper(action_head.vl_self_attention).to(aux_dtype).eval().cuda()
        hidden_states = torch.randn(
            bs, vl_seq_len, config.backbone_embedding_dim, dtype=aux_dtype, device="cuda"
        )
        names, inputs = ["hidden_states"], [hidden_states]
        axes = {"hidden_states": {1: "seq_len"}}
        handles = []
        if action_head.mask_vl_padding:
            names.append("attention_mask")
            inputs.append(torch.ones(bs, vl_seq_len, dtype=torch.bool, device="cuda"))
            axes["attention_mask"] = {1: "seq_len"}
            handles = _patch_dit_attn_for_export(action_head.vl_self_attention)
        try:
            io["vl_self_attention"] = export_component(
                "vl_self_attention",
                model,
                inputs,
                names,
                ["output"],
                dynamic_axes=axes if dynamic else None,
                opt_len=vl_seq_len,
                **common,
            )
        finally:
            _unpatch_dit_attn(handles)

    if "state_encoder" in args.components:
        model = action_head.state_encoder.to(aux_dtype).eval().cuda()
        state_dim = config.max_state_dim * config.state_history_length
        io["state_encoder"] = export_component(
            "state_encoder",
            model,
            [
                torch.randn(bs, 1, state_dim, dtype=aux_dtype, device="cuda"),
                torch.zeros(bs, dtype=torch.int64, device="cuda"),
            ],
            ["state", "embodiment_id"],
            ["output"],
            **common,
        )

    if "action_encoder" in args.components:
        model = action_head.action_encoder.to(aux_dtype).eval().cuda()
        io["action_encoder"] = export_component(
            "action_encoder",
            model,
            [
                torch.randn(
                    bs, config.action_horizon, config.max_action_dim, dtype=aux_dtype, device="cuda"
                ),
                torch.zeros(bs, dtype=torch.int64, device="cuda"),
                torch.zeros(bs, dtype=torch.int64, device="cuda"),
            ],
            ["actions", "timesteps", "embodiment_id"],
            ["output"],
            **common,
        )

    if "dit" in args.components:
        dit_model = action_head.model.to(dit_dtype).eval().cuda()
        names, inputs = dit_export_inputs(dit_capture, bs, dit_dtype, "cuda")
        wrapper = DiTExportWrapper(dit_model, names[3:]).to(dit_dtype).eval().cuda()
        axes = {name: {1: "vl_seq_len"} for name in names if name not in ("sa_embs", "timestep")}
        axes.pop("hidden_attention_mask", None)
        handles = _patch_dit_attn_for_export(dit_model)
        try:
            io["dit"] = export_component(
                f"dit_{precisions['dit']}",
                wrapper,
                inputs,
                names,
                ["output"],
                dynamic_axes=axes if dynamic else None,
                opt_len=vl_seq_len,
                **common,
            )
        finally:
            _unpatch_dit_attn(handles)

    if "action_decoder" in args.components:
        model = action_head.action_decoder.to(aux_dtype).eval().cuda()
        io["action_decoder"] = export_component(
            "action_decoder",
            model,
            [
                torch.randn(
                    bs,
                    1 + config.action_horizon,
                    config.hidden_size,
                    dtype=aux_dtype,
                    device="cuda",
                ),
                torch.zeros(bs, dtype=torch.int64, device="cuda"),
            ],
            ["model_output", "embodiment_id"],
            ["output"],
            **common,
        )

    metadata[EXECUTORCH_IO_KEY] = io
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved export metadata to {metadata_path}")
    for f in sorted(os.listdir(args.output_dir)):
        size_mb = os.path.getsize(os.path.join(args.output_dir, f)) / 2**20
        logger.info(f"  {f}: {size_mb:.2f} MB")


Component = Literal[
    "vit",
    "llm",
    "vl_self_attention",
    "state_encoder",
    "action_encoder",
    "dit",
    "action_decoder",
]


@dataclass
class ExecuTorchExportConfig:
    """Configuration for exporting GR00T N1.7 to ExecuTorch programs with TensorRT."""

    model_path: str
    """Path to the model checkpoint (required)."""

    dataset_path: str
    """Path to the dataset (required, used to capture input shapes)."""

    embodiment_tag: Optional[EmbodimentTag] = None
    """Embodiment tag. If not provided, auto-detected from model's processor_config.json."""

    output_dir: str = "./gr00t_executorch_deployment"
    """Output directory for the .pte programs."""

    seq_len: Literal["dynamic", "static"] = "dynamic"
    """'dynamic' accepts any prompt length in the TensorRT profile range, as the
    ONNX engines do. 'static' fixes the captured length."""

    components: tuple[Component, ...] = get_args(Component)
    """Components to export (default: all of the full_pipeline set). The runtime loads
    only an export of every component; a subset is for checking that those export."""

    workspace: Optional[int] = None
    """TensorRT builder workspace in MB, as for the engine builder (default: TensorRT decides)."""

    disable_tf32: bool = False
    """Build FP32 layers (the ViT) without TF32."""

    batch_size: int = 1
    """Batch size baked into the exported programs (default: 1)."""


if __name__ == "__main__":
    args = tyro.cli(ExecuTorchExportConfig)
    if args.embodiment_tag is None:
        config_file = Path(args.model_path) / "processor_config.json"
        modality_configs = {}
        if config_file.exists():
            with open(config_file) as f:
                processor_kwargs = json.load(f).get("processor_kwargs", {})
            modality_configs = processor_kwargs.get("modality_configs", {})
        if len(modality_configs) != 1:
            raise ValueError(
                f"Cannot auto-detect embodiment_tag from {config_file} "
                f"(embodiments: {sorted(modality_configs)}). "
                "Please provide --embodiment-tag explicitly."
            )
        args.embodiment_tag = EmbodimentTag.resolve(next(iter(modality_configs)))
    main(args)
