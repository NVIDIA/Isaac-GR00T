#!/usr/bin/env python3
# ruff: noqa: T201
"""Export a checkpoint-shaped N2 VAE encoder with BF16 I/O and FP32 normalization.

The contract is intentionally static and derived from checkpoint metadata:

    videos  [num_cameras, 3, conditioning_frames, height, width] BF16
    latents [num_cameras, C, conditioning_latent_frames, H, W]   BF16

The deterministic sample is consumed by the shared post-build parity gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
import time
from types import MethodType
from typing import Any

from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F

from groot.core.training.models.n2.components.wan22_video_vae import WanVideoVAE38
from groot.core.training.models.n2.components.wan_video_vae import (
    Resample,
    RMS_norm,
    WanVideoVAE,
)
from groot.deployment.n2_trt import (
    N2TensorRTContract,
    resolve_checkpoint_vae_type,
    resolve_checkpoint_vae_weights,
)


def _vae_shapes(
    num_cameras: int,
    conditioning_frames: int,
    image_size: tuple[int, int],
    latent_shape: tuple[int, int, int, int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not isinstance(num_cameras, int) or isinstance(num_cameras, bool) or num_cameras <= 0:
        raise ValueError(f"num_cameras must be a positive integer, got {num_cameras!r}")
    if conditioning_frames <= 0:
        raise ValueError(f"conditioning_frames must be positive, got {conditioning_frames}")
    return (
        (num_cameras, 3, conditioning_frames, *image_size),
        (num_cameras, *latent_shape),
    )


def _checkpoint_contract(checkpoint: Path, embodiment: str) -> N2TensorRTContract:
    """Resolve every fixed VAE dimension through the shared checkpoint contract."""
    return N2TensorRTContract.from_checkpoint(
        checkpoint,
        embodiment=embodiment,
        video_denoising_steps=2,
        action_denoising_steps=4,
        joint_denoise=True,
        conditional_kv_cache=True,
    )


def _build_vae(checkpoint: Path, weights: Path):
    """Construct the VAE this checkpoint actually trained with.

    The class and kwargs follow ``GrootN2.build_vae`` exactly. Building the
    wrong latent-space VAE fails with state-dict size mismatches.
    """
    config = json.loads((checkpoint / "config.json").read_text())
    vae_cfg = dict(config.get("vae_cfg") or {})
    vae_cfg["vae_pretrained_path"] = str(weights)
    vae_type = resolve_checkpoint_vae_type(config)
    vae_cls = WanVideoVAE38 if vae_type == "wan22" else WanVideoVAE
    return vae_cls(**vae_cfg)


INPUT_NAME = "videos"
OUTPUT_NAME = "latents"
EXPECTED_DTYPE = torch.bfloat16


def _downsample_full_sequence(layer: Resample, x: torch.Tensor) -> torch.Tensor:
    """Apply one causal 3D downsampler to a complete sequence."""
    batch, _channels, frames, _height, _width = x.shape
    x = rearrange(x, "b c t h w -> (b t) c h w")
    x = layer.resample(x)
    x = rearrange(x, "(b t) c h w -> b c t h w", b=batch, t=frames)
    if frames == 1:
        return x
    return torch.cat([x[:, :, :1], layer.time_conv(x)], dim=2)


def _encode_full_sequence(encoder: nn.Module, videos: torch.Tensor) -> torch.Tensor:
    """Exporter-local equivalent of the streaming T=5 encoder."""
    x = encoder.conv1(videos)
    for layer in encoder.downsamples:
        if isinstance(layer, Resample) and layer.mode == "downsample3d":
            x = _downsample_full_sequence(layer, x)
        else:
            x = layer(x)
    for layer in encoder.middle:
        x = layer(x)
    for layer in encoder.head:
        x = layer(x)
    return x


class YamN2VaeEncodeExportAdapter(nn.Module):
    """Tensor-only adapter around the equivalent full-sequence encoder."""

    def __init__(self, vae: WanVideoVAE):
        super().__init__()
        self.encoder = vae.model.encoder
        self.conv1 = vae.model.conv1
        self.z_dim = vae.z_dim
        self.register_buffer("mean", vae.mean.detach().to(dtype=torch.bfloat16))
        # Base N2's WanVideoVAE registers `std`; nano's Wan2.2 VAE38 registers the
        # reciprocal as `inv_std` and has no `std` at all.
        inv_std = vae.inv_std if hasattr(vae, "inv_std") else 1.0 / vae.std
        self.register_buffer("inv_std", inv_std.detach().to(dtype=torch.bfloat16))
        # Base N2 goes through _encode_full_sequence, a hand-unrolled equivalent of
        # its streaming encoder. That unrolling is specific to base's block layout:
        # Wan2.2 patchifies 2x2 into channels first and its residual blocks carry an
        # avg_shortcut with different temporal semantics, so the same walk desyncs
        # (T=5 against T=3). VideoVAE38_.encode already does patchify, the chunk
        # loop and the scale, and for a fixed 5-frame window that loop is a static
        # two-iteration Python loop that traces into ONNX as a flat graph -- so nano
        # takes the pipeline entry point instead of a second hand-written unrolling.
        # Bound without registering vae.model as a submodule: it owns the decoder
        # too, and the FP32 RMS_norm override walks adapter.modules(), so registering
        # it patches 52 norms while only the encoder's 44 reach the graph -- which
        # the graph validator then rejects. self.encoder is the same object encode()
        # runs, so the override still lands where it matters.
        object.__setattr__(self, "_pipeline_encode", vae.model.encode)
        self.use_pipeline_encode = isinstance(vae, WanVideoVAE38)

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        # The full-sequence graph is equivalent for complete 1+4k windows and
        # preserves the already-qualified path. Other fixed lengths use the
        # model's own static, traceable chunk loop so trailing frames have the
        # exact same semantics as eager inference.
        if self.use_pipeline_encode or (videos.shape[2] - 1) % 4:
            return self._pipeline_encode(videos, [self.mean, self.inv_std])
        encoded = _encode_full_sequence(self.encoder, videos)
        mean, _log_var = self.conv1(encoded).chunk(2, dim=1)
        return (mean - self.mean.view(1, self.z_dim, 1, 1, 1)) * self.inv_std.view(
            1, self.z_dim, 1, 1, 1
        )


def _rms_norm_fp32_accum(self: RMS_norm, x: torch.Tensor) -> torch.Tensor:
    """Export-only RMS normalization with a numerically safe FP32 reduction."""
    dim = 1 if self.channel_first else -1
    normalized = F.normalize(x.float(), dim=dim).to(dtype=x.dtype)
    return normalized * self.scale * self.gamma + self.bias


def _install_fp32_rms_norm_export(adapter: nn.Module) -> int:
    """Override only this export instance; eager source behavior is untouched.

    Matched by class name, not isinstance: base N2's VAE and nano's Wan2.2 VAE each
    define their own structurally identical RMS_norm, in
    groot/core/.../components/wan_video_vae.py and
    groot/vla/omni/model/dream_cosmos/wan_video_vae.py respectively. The latter is
    loaded through importlib under a synthetic module name, so an isinstance check
    against either import silently matches nothing for the other model.
    """
    count = 0
    for module in adapter.modules():
        if type(module).__name__ == RMS_norm.__name__:
            module.forward = MethodType(_rms_norm_fp32_accum, module)
            count += 1
    if count == 0:
        raise RuntimeError("VAE export adapter contains no RMS_norm modules")
    return count


def _cast_target(node: Any) -> int | None:
    for attribute in node.attribute:
        if attribute.name == "to":
            return int(attribute.i)
    return None


def _has_descendant_bf16_cast(
    start: Any,
    consumers: dict[str, list[Any]],
    *,
    max_depth: int = 8,
) -> bool:
    from onnx import TensorProto

    frontier = list(start.output)
    visited: set[str] = set()
    for _depth in range(max_depth):
        next_frontier: list[str] = []
        for value_name in frontier:
            if value_name in visited:
                continue
            visited.add(value_name)
            for node in consumers.get(value_name, []):
                if node.op_type == "Cast" and _cast_target(node) == TensorProto.BFLOAT16:
                    return True
                next_frontier.extend(node.output)
        frontier = next_frontier
    return False


def _validate_fp32_rms_norm_graph(path: Path, expected_count: int) -> dict[str, Any]:
    """Fail export unless every RMS ReduceL2 has FP32 input and BF16 output."""
    import onnx
    from onnx import TensorProto

    model = onnx.load(str(path), load_external_data=False)
    nodes = list(model.graph.node)
    producers = {output: node for node in nodes for output in node.output}
    consumers: dict[str, list[Any]] = {}
    for node in nodes:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    reductions = [node for node in nodes if node.op_type == "ReduceL2"]
    fp32_inputs = 0
    bf16_outputs = 0
    for reduction in reductions:
        producer = producers.get(reduction.input[0])
        if (
            producer is not None
            and producer.op_type == "Cast"
            and _cast_target(producer) == TensorProto.FLOAT
        ):
            fp32_inputs += 1
        if _has_descendant_bf16_cast(reduction, consumers):
            bf16_outputs += 1

    record = {
        "implementation": "export_instance_forward_override",
        "accumulation_dtype": "torch.float32",
        "affine_dtype": "torch.bfloat16",
        "patched_module_count": expected_count,
        "reduce_l2_count": len(reductions),
        "reduce_l2_fp32_input_count": fp32_inputs,
        "reduce_l2_with_bf16_output_cast_count": bf16_outputs,
    }
    # The guarantee is that *every* ReduceL2 in the graph accumulates in FP32 and
    # casts back to BF16. Base N2 traces each patched RMS_norm exactly once, so
    # equality with the module count held; nano's encode runs the encoder once per
    # temporal chunk (2 for the fixed 5-frame window), so each norm is traced once
    # per chunk and the count is a multiple. Requiring equality rejected a graph in
    # which all 44 reductions were correctly FP32.
    record["reduce_l2_per_patched_module"] = (
        len(reductions) / expected_count if expected_count else 0
    )
    if not (
        expected_count > 0
        and len(reductions) > 0
        and len(reductions) % expected_count == 0
        and fp32_inputs == len(reductions)
        and bf16_outputs == len(reductions)
    ):
        raise RuntimeError(f"FP32 RMS_norm ONNX graph validation failed: {record}")
    return record


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embodiment", required=True)
    parser.add_argument("--vae-weights", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-revision")
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
        "rms": float(value.square().mean().sqrt().item()),
    }


def _difference_record(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    diff = (actual.float() - expected.float()).abs()
    expected_rms = expected.float().square().mean().sqrt()
    rmse = diff.square().mean().sqrt()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rmse": float(rmse.item()),
        "relative_rmse": float((rmse / expected_rms).item()),
    }


def _save_eager_sample(
    path: Path,
    *,
    videos: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    from groot.deployment.n2_trt.manifest import VAE_EAGER_ORACLE

    torch.save(
        {
            "oracle": VAE_EAGER_ORACLE,
            INPUT_NAME: videos.detach().cpu(),
            OUTPUT_NAME: reference.detach().cpu(),
        },
        path,
    )


def _onnx_summary(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    op_counts = Counter(node.op_type for node in model.graph.node)
    return {
        "ir_version": model.ir_version,
        "opsets": [
            {"domain": opset.domain, "version": opset.version} for opset in model.opset_import
        ],
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "op_counts": dict(sorted(op_counts.items())),
        "inputs": [value.name for value in model.graph.input],
        "outputs": [value.name for value in model.graph.output],
    }


def _export(args: argparse.Namespace) -> int:
    import onnx

    from groot.core.training.models.n2.utils.cosmos_model_deps import ensure_cosmos_model_deps
    from groot.deployment.n2_trt.manifest import (
        VAE_EAGER_ORACLE,
        SourceRevision,
        sha256_file,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to export the GR00T2 VAE encoder")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    ensure_cosmos_model_deps(checkpoint)
    contract = _checkpoint_contract(checkpoint, args.embodiment)
    input_shape, output_shape = _vae_shapes(
        contract.camera_count,
        contract.conditioning_frames,
        contract.image_size,
        contract.latent_shape,
    )
    weights = resolve_checkpoint_vae_weights(checkpoint, args.vae_weights)
    output_dir = Path(args.output_dir)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "n2_vae_encoder.onnx"

    print(f"[n2-vae] checkpoint={checkpoint}")
    print(f"[n2-vae] weights={weights}")
    print(f"[n2-vae] output={onnx_path}")

    vae = _build_vae(checkpoint, weights)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    vae.model.load_state_dict(state, strict=True, assign=True)
    vae = vae.eval().requires_grad_(False).to(device="cuda", dtype=torch.bfloat16)
    adapter = YamN2VaeEncodeExportAdapter(vae).eval()

    generator = torch.Generator(device="cuda").manual_seed(42)
    videos = torch.rand(input_shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    videos = videos.mul(2).sub(1)

    # Capture the unchanged eager/chunked reference before overriding the
    # adapter's shared encoder RMS_norm instances for this export process.
    with torch.inference_mode():
        reference = vae.model.encode(videos, vae.scale).clone()
    rms_norm_count = _install_fp32_rms_norm_export(adapter)
    with torch.inference_mode():
        fused = adapter(videos).clone()
    torch.cuda.synchronize()
    if tuple(fused.shape) != output_shape:
        raise RuntimeError(f"unexpected fused output shape: {tuple(fused.shape)}")
    torch.testing.assert_close(fused, reference, rtol=0.05, atol=0.05)

    export_start = time.perf_counter()
    with torch.inference_mode():
        torch.onnx.export(
            adapter,
            (videos,),
            str(onnx_path),
            input_names=["videos"],
            output_names=["latents"],
            opset_version=20,
            dynamo=False,
            external_data=True,
            do_constant_folding=False,
        )
    torch.cuda.synchronize()
    export_elapsed_s = time.perf_counter() - export_start
    rms_norm_export = _validate_fp32_rms_norm_graph(onnx_path, rms_norm_count)

    _save_eager_sample(
        output_dir / "sample.pt",
        videos=videos,
        reference=reference,
    )

    manifest = {
        "source": asdict(
            SourceRevision.capture("exporter", __file__, revision=args.source_revision)
        ),
        "checkpoint": str(checkpoint),
        "num_cameras": contract.camera_count,
        "vae_weights": str(weights),
        "vae_weights_sha256": sha256_file(weights),
        "validation_sample_sha256": {"vae_eager": sha256_file(output_dir / "sample.pt")},
        "onnx": str(onnx_path),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "export_elapsed_s": export_elapsed_s,
        "input": _tensor_record(videos),
        "output": _tensor_record(reference),
        "export_adapter_output": _tensor_record(fused),
        "parity_oracle": VAE_EAGER_ORACLE,
        "fused_vs_chunked": _difference_record(fused, reference),
        "onnx_summary": _onnx_summary(onnx_path),
        "rms_norm_export": rms_norm_export,
        "runtime_contract": {
            "server_engine_dir_argument": "--engine-dir",
            "engine_filename": "n2_vae_encoder.engine",
            "input_name": "videos",
            "input_shape": list(input_shape),
            "input_dtype": "torch.bfloat16",
            "output_name": "latents",
            "output_shape": list(output_shape),
            "output_dtype": "torch.bfloat16",
        },
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "onnx": onnx.__version__,
        },
    }
    manifest_path = output_dir / "n2_vae_export_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"[n2-vae] export_elapsed_s={export_elapsed_s:.3f}")
    print(f"[n2-vae] manifest={manifest_path}")
    return 0


def main() -> int:
    return _export(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
