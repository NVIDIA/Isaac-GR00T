# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Fixed-shape TensorRT runtime for GR00T2 deployment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from groot.core.training.models.n2.components.kv_cache import CondKVCache
from groot.core.training.models.n2.components.sequence_packing import (
    AttentionMaskPattern,
    PackedSequence,
)
from groot.deployment.n2_trt.contract import N2TensorRTContract
from groot.deployment.n2_trt.manifest import (
    BUILD_FILENAME,
    MANIFEST_FILENAME,
    EngineManifest,
    RuntimeIdentity,
    required_serving_files,
    sha256_file,
)

_MEASUREMENT_ENGINE_ROLES = {
    "transformer_joint": "transformer_joint_capture",
    "transformer_action_cache": "transformer_action_cached",
    "vae_encoder": "vae_encode",
}
_MEASUREMENT_CHECKPOINT_FILES = {
    "config.json",
    "processor_config.json",
    "statistics.json",
    "model.safetensors.index.json",
}


def _torch_dtype_for_trt(trt_module: Any, dtype: Any) -> torch.dtype:
    if dtype == trt_module.float32:
        return torch.float32
    if dtype == trt_module.float16:
        return torch.float16
    if dtype == trt_module.bfloat16:
        return torch.bfloat16
    if dtype == trt_module.int32:
        return torch.int32
    if dtype == trt_module.int64:
        return torch.int64
    raise TypeError(f"Unsupported TensorRT dtype: {dtype}")


_SUPPORTED_CAMERA_COUNTS = (3, 4)
# Per-camera VAE latent (C, T, H, W) for one 5-frame 224x320 window. Base N2's VAE
# downsamples /8 into 16 channels; GR00T2-TURBO's Wan2.2 VAE downsamples /16 into 48,
# which is a different token count and a different engine shape, so callers that
# know their model pass its geometry instead of inheriting this default.
_BASE_LATENT_SHAPE = (16, 2, 28, 40)
_PATCH_SIZE = 2
_STATE_TOKENS = 1


def video_tokens_per_camera(latent_shape: tuple[int, int, int, int]) -> int:
    """Video tokens one camera contributes, from its per-camera latent (C, T, H, W).

    The DiT patchifies the latent 2x2 spatially and attends over both the
    conditioning and the target latent frames, i.e. 2T frames. Base N2's 28x40
    latents give 14*20*4 = 1120 tokens; GR00T2-TURBO's 14x20 give 7*10*4 = 280.
    """
    _, frames, height, width = latent_shape
    if height % _PATCH_SIZE or width % _PATCH_SIZE:
        raise ValueError(
            f"latent {height}x{width} is not divisible by the {_PATCH_SIZE}x{_PATCH_SIZE} patch"
        )
    return (height // _PATCH_SIZE) * (width // _PATCH_SIZE) * 2 * frames


def _require_supported_camera_count(num_cameras: int) -> None:
    if (
        not isinstance(num_cameras, int)
        or isinstance(num_cameras, bool)
        or num_cameras not in _SUPPORTED_CAMERA_COUNTS
    ):
        raise ValueError(
            f"fixed-shape N2 TensorRT supports camera counts {_SUPPORTED_CAMERA_COUNTS}; got {num_cameras}"
        )


def n2_segment_token_counts(
    num_cameras: int,
    action_horizon: int,
    latent_shape: tuple[int, int, int, int] = _BASE_LATENT_SHAPE,
) -> tuple[int, int, int]:
    """Fixed (video, state, action) layout for a supported GR00T2 export."""
    _require_supported_camera_count(num_cameras)
    if not isinstance(action_horizon, int) or isinstance(action_horizon, bool):
        raise TypeError(f"action_horizon must be an int, got {action_horizon!r}")
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")
    tokens = video_tokens_per_camera(latent_shape)
    return (tokens * num_cameras, _STATE_TOKENS, action_horizon)


def _vae_trt_shapes(
    num_cameras: int,
    latent_shape: tuple[int, int, int, int] = _BASE_LATENT_SHAPE,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    _require_supported_camera_count(num_cameras)
    return (num_cameras, 3, 5, 224, 320), (num_cameras, *latent_shape)


def n2_trt_runtime_contract(
    num_cameras: int,
    num_video_steps: int = 2,
    num_action_steps: int = 4,
) -> str:
    """Return the complete runtime contract string.

    The denoising schedule changes how many times each engine is called without
    changing engine shapes, so it is part of the runtime identity.
    """
    _require_supported_camera_count(num_cameras)
    return f"gr00t2_{num_cameras}cam_{num_video_steps}v{num_action_steps}a_tensorrt"


def _require_n2_runtime_contract(
    *,
    expected_num_cameras: int,
    expected_num_video_steps: int,
    expected_num_action_steps: int,
    batch_size: int,
    num_cameras: int,
    num_video_steps: int,
    num_action_steps: int,
    use_kv_cache: bool,
    cfg_on: bool,
    attention_pattern: AttentionMaskPattern,
    video_padded_mask: Any,
) -> None:
    """Fail closed unless inference matches the selected exported TRT workload."""
    _require_supported_camera_count(expected_num_cameras)
    reasons: list[str] = []
    if (batch_size, num_cameras) != (1, expected_num_cameras):
        reasons.append(f"batch_cameras:{(batch_size, num_cameras)}!=(1, {expected_num_cameras})")
    expected_schedule = (expected_num_video_steps, expected_num_action_steps)
    if (num_video_steps, num_action_steps) != expected_schedule:
        reasons.append(f"schedule:{(num_video_steps, num_action_steps)}!={expected_schedule}")
    if not use_kv_cache:
        reasons.append("kv_cache_disabled")
    if cfg_on:
        reasons.append("cfg_enabled")
    expected_pattern = AttentionMaskPattern.BIDIRECTIONAL_VIDEO_ACTION_FULL
    if attention_pattern != expected_pattern:
        reasons.append(f"attention_pattern:{attention_pattern.value}!={expected_pattern.value}")
    if (
        video_padded_mask is not None
        and _host_checks_allowed()
        and bool(torch.as_tensor(video_padded_mask).any().item())
    ):
        reasons.append("padded_camera")
    if reasons:
        raise RuntimeError(
            f"strict GR00T2 {expected_num_cameras}-camera TensorRT contract violation: "
            + ";".join(reasons)
        )


def _host_checks_allowed() -> bool:
    """Return false only while a CUDA graph is actively being captured."""
    if not torch.cuda.is_available():
        return True
    return not torch.cuda.is_current_stream_capturing()


def _require_segment_layout(
    segment_counts: tuple[int, int, int],
    *,
    num_cameras: int,
    action_horizon: int,
    latent_shape: tuple[int, int, int, int] = _BASE_LATENT_SHAPE,
) -> None:
    expected = n2_segment_token_counts(num_cameras, action_horizon, latent_shape)
    if segment_counts != expected:
        raise RuntimeError(
            f"fixed-shape N2 TensorRT segment layout {segment_counts} != {expected} "
            f"for {num_cameras} cameras and action horizon {action_horizon}"
        )


def _vae_trt_contract_skip_reason(
    runner: Any,
    *,
    num_cameras: int,
    latent_shape: tuple[int, int, int, int] = _BASE_LATENT_SHAPE,
) -> str | None:
    """Validate the fixed-shape VAE engine contract."""
    input_shape, output_shape = _vae_trt_shapes(num_cameras, latent_shape)
    reasons: list[str] = []
    if set(runner.inputs) != {"videos"}:
        reasons.append(f"input_names:{sorted(runner.inputs)}!=['videos']")
    else:
        input_meta = runner.inputs["videos"]
        if tuple(input_meta["shape"]) != input_shape:
            reasons.append(f"input_shape:{tuple(input_meta['shape'])}!={input_shape}")
        if input_meta["dtype"] != torch.bfloat16:
            reasons.append(f"input_dtype:{input_meta['dtype']}!=torch.bfloat16")

    if runner.output_names != ["latents"]:
        reasons.append(f"output_names:{runner.output_names}!=['latents']")
    elif "latents" not in runner.outputs:
        reasons.append("missing_output:latents")
    else:
        output = runner.outputs["latents"]
        if tuple(output.shape) != output_shape:
            reasons.append(f"output_shape:{tuple(output.shape)}!={output_shape}")
        if output.dtype != torch.bfloat16:
            reasons.append(f"output_dtype:{output.dtype}!=torch.bfloat16")
    return ";".join(reasons) if reasons else None


class N2TensorTrtRunner:
    """Runtime for a fixed-shape tensor-only engine."""

    def __init__(self, engine_path: str):
        import tensorrt as trt

        self.trt = trt
        self.engine_path = engine_path
        self.device = torch.device("cuda", torch.cuda.current_device())
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=self.device)
        self.inputs: dict[str, dict[str, Any]] = {}
        self.output_names: list[str] = []
        self.outputs: dict[str, torch.Tensor] = {}

        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            mode = self.engine.get_tensor_mode(name)
            dtype = _torch_dtype_for_trt(trt, self.engine.get_tensor_dtype(name))
            shape = tuple(int(dim) for dim in self.engine.get_tensor_shape(name))
            if mode == trt.TensorIOMode.INPUT:
                self.inputs[name] = {"shape": shape, "dtype": dtype}
            else:
                self.output_names.append(name)
                output = torch.empty(shape, device=self.device, dtype=dtype)
                self.outputs[name] = output
                self.context.set_tensor_address(name, output.data_ptr())

        if not self.outputs:
            raise RuntimeError(f"TensorRT engine has no output binding: {engine_path}")

    def skip_reason(self, tensors: dict[str, torch.Tensor]) -> str | None:
        """Return an input-contract error; callers treat any result as fatal."""
        reasons: list[str] = []
        for name, meta in self.inputs.items():
            if name not in tensors:
                reasons.append(f"missing_input:{name}")
                continue
            tensor = tensors[name]
            if tensor.device != self.device:
                reasons.append(f"device_mismatch:{name}:{tensor.device}!={self.device}")
                continue
            shape = tuple(tensor.shape)
            if shape != meta["shape"]:
                reasons.append(f"shape_mismatch:{name}:{shape}!={meta['shape']}")
            if tensor.dtype != meta["dtype"]:
                reasons.append(f"dtype_mismatch:{name}:{tensor.dtype}!={meta['dtype']}")
        return ";".join(reasons) if reasons else None

    def _execute(self, tensors: dict[str, torch.Tensor]) -> None:
        for name in self.inputs:
            tensor = tensors[name]
            if tensor.device != self.device:
                raise RuntimeError(
                    f"TensorRT input {name!r} is on {tensor.device}; runner is bound to {self.device}"
                )

        caller_stream = torch.cuda.current_stream(self.device)
        capturing = torch.cuda.is_current_stream_capturing()
        for name, meta in self.inputs.items():
            tensor = tensors[name].to(dtype=meta["dtype"]).contiguous()
            if not capturing:
                tensor.record_stream(self.stream)
            self.context.set_tensor_address(name, tensor.data_ptr())
            tensors[name] = tensor

        if capturing:
            # The private side stream is not part of a whole-predict graph.
            # Warmup validates these bindings before capture, then capture must
            # enqueue directly on the stream owned by the graph.
            ok = self.context.execute_async_v3(caller_stream.cuda_stream)
        else:
            self.stream.wait_stream(caller_stream)
            with torch.cuda.stream(self.stream):
                ok = self.context.execute_async_v3(self.stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned false")
        if not capturing:
            caller_stream.wait_stream(self.stream)

    def __call__(self, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        self._execute(tensors)
        return self.outputs[self.output_names[0]].clone()


# Private exporter/runtime ABI. TensorRT provides named I/O bindings, but does
# not define these N2-specific names. ``export_n2_transformer.py`` emits this
# schema and the runtime requires it before serving; a mismatch never selects
# the eager implementation.
_TRANSFORMER_TRT_COMMON_INPUT_NAMES = {
    "latents",
    "timesteps",
    "rope_cos",
    "rope_sin",
    "context",
}
_TRANSFORMER_TRT_INPUT_NAMES = {
    "joint": _TRANSFORMER_TRT_COMMON_INPUT_NAMES | {"modality_ids"},
    "action": _TRANSFORMER_TRT_COMMON_INPUT_NAMES
    | {"modality_ids", "kv_cache_key", "kv_cache_value"},
}
_TRANSFORMER_TRT_OUTPUT_NAMES = {
    # joint: denoised tokens, conditioning-cache keys, conditioning-cache values
    "joint": {"output", "output2", "output3"},
    # action: denoised tokens
    "action": {"output1"},
}
_TOKEN_MERGE_CACHE_NAMES = {
    "kv_cache_key_fine",
    "kv_cache_value_fine",
    "kv_cache_key_coarse",
    "kv_cache_value_coarse",
}


def _stack_token_merge_cache(
    cache: CondKVCache,
    token_merge_layers: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack heterogeneous per-layer K/V into fine and coarse tensors."""
    start, end = token_merge_layers
    if not (0 < start < end < len(cache)):
        raise ValueError(
            "N2 TensorRT token merge requires fine layers on both sides of "
            f"the coarse range; got layers={token_merge_layers}, "
            f"transformer_layers={len(cache)}"
        )
    fine_keys = cache.key_cache[:start] + cache.key_cache[end:]
    fine_values = cache.value_cache[:start] + cache.value_cache[end:]
    return (
        torch.stack(fine_keys, dim=0),
        torch.stack(fine_values, dim=0),
        torch.stack(cache.key_cache[start:end], dim=0),
        torch.stack(cache.value_cache[start:end], dim=0),
    )


def _restore_token_merge_cache(
    key_fine: torch.Tensor,
    value_fine: torch.Tensor,
    key_coarse: torch.Tensor,
    value_coarse: torch.Tensor,
    token_merge_layers: tuple[int, int],
) -> CondKVCache:
    """Restore transformer layer order from fixed fine/coarse tensors."""
    start, end = token_merge_layers
    fine_keys = list(key_fine.unbind(0))
    fine_values = list(value_fine.unbind(0))
    coarse_keys = list(key_coarse.unbind(0))
    coarse_values = list(value_coarse.unbind(0))
    if len(coarse_keys) != end - start or len(fine_keys) < start:
        raise RuntimeError(
            "token-merge TensorRT cache outputs do not match layers "
            f"{token_merge_layers}: fine={len(fine_keys)}, coarse={len(coarse_keys)}"
        )
    return CondKVCache(
        key_cache=fine_keys[:start] + coarse_keys + fine_keys[start:],
        value_cache=fine_values[:start] + coarse_values + fine_values[start:],
    )


def _require_transformer_bindings(
    mode: str,
    input_names: set[str],
    output_names: list[str],
    *,
    token_merge_layers: tuple[int, int] | None = None,
) -> None:
    """Require the exact private ABI, with one documented ONNX optimization.

    Static-routing exports bake routing indices into the graph, so ONNX may
    remove the otherwise unused ``modality_ids`` input. That is the only
    accepted omission. Every other missing, renamed, or additional binding is
    fatal; this function does not implement an eager fallback.
    """
    expected_inputs = set(_TRANSFORMER_TRT_INPUT_NAMES[mode])
    expected_outputs = set(_TRANSFORMER_TRT_OUTPUT_NAMES[mode])
    if token_merge_layers is not None:
        if mode == "action":
            expected_inputs -= {"kv_cache_key", "kv_cache_value"}
            expected_inputs |= _TOKEN_MERGE_CACHE_NAMES
        else:
            expected_outputs = {"output"} | _TOKEN_MERGE_CACHE_NAMES
    actual_inputs = set(input_names)
    actual_outputs = set(output_names)
    reasons = []
    # ONNX/TensorRT removes modality_ids when the checkpoint-shaped routing
    # resolves to constants; the 1v1 graph still consumes it at runtime.
    accepted_inputs = (expected_inputs, expected_inputs - {"modality_ids"})
    if actual_inputs not in accepted_inputs:
        accepted = [sorted(names) for names in accepted_inputs]
        reasons.append(f"inputs:{sorted(actual_inputs)} not in {accepted}")
    if actual_outputs != expected_outputs:
        reasons.append(f"outputs:{sorted(actual_outputs)}!={sorted(expected_outputs)}")
    if reasons:
        raise RuntimeError(f"invalid {mode} transformer bindings: " + ";".join(reasons))


class _N2TransformerTrtRunner(N2TensorTrtRunner):
    """Runtime for an exported N2 transformer TensorRT engine."""

    def __init__(
        self,
        engine_path: str,
        *,
        mode: str,
        token_merge_layers: tuple[int, int] | None = None,
    ):
        super().__init__(engine_path)
        self.mode = mode
        self.token_merge_layers = token_merge_layers
        _require_transformer_bindings(
            mode,
            set(self.inputs),
            self.output_names,
            token_merge_layers=token_merge_layers,
        )

    @staticmethod
    def _candidate_tensors(
        *,
        packed: PackedSequence,
        prompt_embs: torch.Tensor,
        kv_cache: CondKVCache | None,
        token_merge_layers: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        tensors = {
            "latents": packed.tokens,
            "timesteps": packed.timesteps,
            "rope_cos": packed.rope_freqs[0],
            "rope_sin": packed.rope_freqs[1],
            "context": prompt_embs,
            "modality_ids": packed.modality_ids,
        }
        if packed.extra_pos_emb is not None:
            tensors["extra_pos_emb"] = packed.extra_pos_emb
        if kv_cache is not None and len(kv_cache) > 0:
            if token_merge_layers is None:
                tensors["kv_cache_key"] = torch.stack(kv_cache.key_cache, dim=0)
                tensors["kv_cache_value"] = torch.stack(kv_cache.value_cache, dim=0)
            else:
                key_fine, value_fine, key_coarse, value_coarse = _stack_token_merge_cache(
                    kv_cache,
                    token_merge_layers,
                )
                tensors.update(
                    {
                        "kv_cache_key_fine": key_fine,
                        "kv_cache_value_fine": value_fine,
                        "kv_cache_key_coarse": key_coarse,
                        "kv_cache_value_coarse": value_coarse,
                    }
                )
        return tensors

    def skip_reason(
        self,
        *,
        packed: PackedSequence,
        prompt_embs: torch.Tensor,
        kv_cache: CondKVCache | None,
    ) -> str | None:
        if packed.extra_pos_emb is not None and "extra_pos_emb" not in self.inputs:
            return "extra_pos_emb_present"
        return super().skip_reason(
            self._candidate_tensors(
                packed=packed,
                prompt_embs=prompt_embs,
                kv_cache=kv_cache,
                token_merge_layers=self.token_merge_layers,
            )
        )

    def __call__(
        self,
        *,
        packed: PackedSequence,
        prompt_embs: torch.Tensor,
        kv_cache: CondKVCache | None,
    ) -> torch.Tensor:
        tensors = self._candidate_tensors(
            packed=packed,
            prompt_embs=prompt_embs,
            kv_cache=kv_cache,
            token_merge_layers=self.token_merge_layers,
        )
        self._execute(tensors)
        if self.mode == "joint":
            _store_joint_kv_outputs(
                kv_cache,
                self.outputs,
                token_merge_layers=self.token_merge_layers,
            )
            return self.outputs["output"]
        return self.outputs["output1"]


def _store_joint_kv_outputs(
    kv_cache: CondKVCache | None,
    outputs: dict[str, torch.Tensor],
    *,
    token_merge_layers: tuple[int, int] | None = None,
) -> None:
    """Retain joint-call KV outputs only when a later action call needs them."""
    if kv_cache is None:
        return
    # Views into reusable engine output buffers. A caller that needs them
    # across another invocation must slice/clone them first.
    if token_merge_layers is None:
        kv_cache.key_cache = list(outputs["output2"].unbind(0))
        kv_cache.value_cache = list(outputs["output3"].unbind(0))
        return
    restored = _restore_token_merge_cache(
        outputs["kv_cache_key_fine"],
        outputs["kv_cache_value_fine"],
        outputs["kv_cache_key_coarse"],
        outputs["kv_cache_value_coarse"],
        token_merge_layers,
    )
    kv_cache.key_cache = restored.key_cache
    kv_cache.value_cache = restored.value_cache


class N2TensorRTRuntime:
    """Complete fixed-shape engine bundle with no eager fallback path."""

    def __init__(
        self,
        engine_paths: dict[str, str],
        *,
        contract: N2TensorRTContract,
        manifest: EngineManifest | None = None,
        artifact_root: str | Path | None = None,
    ):
        contract.validate()
        self.contract = contract
        self.manifest = manifest
        self.artifact_root = (
            Path(artifact_root).expanduser().resolve(strict=True)
            if artifact_root is not None
            else None
        )
        self.num_cameras = contract.camera_count
        self.action_horizon = contract.action_horizon
        self.latent_shape = contract.latent_shape
        self.num_video_steps = contract.video_denoising_steps
        self.num_action_steps = contract.action_denoising_steps
        self.token_merge_layers = contract.token_merge_layers
        self.segment_token_counts = n2_segment_token_counts(
            self.num_cameras,
            self.action_horizon,
            self.latent_shape,
        )
        # With the conditioning KV cache on, the action steps are split across the
        # joint calls, so the schedule must divide evenly.
        if self.num_video_steps <= 0 or self.num_action_steps % self.num_video_steps != 0:
            raise ValueError(
                f"num_action_steps ({self.num_action_steps}) must be a positive multiple of "
                f"num_video_steps ({self.num_video_steps})"
            )
        self.transformer_runners = {
            "joint": _N2TransformerTrtRunner(
                engine_paths["joint"],
                mode="joint",
                token_merge_layers=self.token_merge_layers,
            ),
            "action": _N2TransformerTrtRunner(
                engine_paths["action"],
                mode="action",
                token_merge_layers=self.token_merge_layers,
            ),
        }
        self.vae_runner = N2TensorTrtRunner(engine_paths["vae"])
        contract_reason = _vae_trt_contract_skip_reason(
            self.vae_runner,
            num_cameras=self.num_cameras,
            latent_shape=self.latent_shape,
        )
        if contract_reason is not None:
            raise RuntimeError(f"invalid VAE TensorRT engine contract: {contract_reason}")

    @classmethod
    def from_directory(
        cls,
        engine_dir: str | Path,
        *,
        contract: N2TensorRTContract,
        runtime_identity: RuntimeIdentity | None = None,
    ) -> N2TensorRTRuntime:
        """Validate the complete artifact bundle before deserializing plans."""
        root = Path(engine_dir).expanduser().resolve(strict=True)
        manifest = EngineManifest.read(root / MANIFEST_FILENAME)
        manifest.validate(
            artifact_root=root,
            contract=contract,
            runtime=runtime_identity or RuntimeIdentity.capture(),
        )
        missing = [name for name in required_serving_files(manifest) if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete N2 TensorRT serving bundle; missing {missing}")
        role_to_name = {
            "transformer_joint": "joint",
            "transformer_action_cache": "action",
            "vae_encoder": "vae",
        }
        engine_paths = {
            role_to_name[record.role]: str((root / record.relative_path).resolve(strict=True))
            for record in manifest.engines
        }
        return cls(
            engine_paths,
            contract=contract,
            manifest=manifest,
            artifact_root=root,
        )

    @property
    def runtime_contract(self) -> str:
        return n2_trt_runtime_contract(
            self.num_cameras, self.num_video_steps, self.num_action_steps
        )

    def measurement_artifacts(self, *, backend: str, precision: str) -> dict[str, Any]:
        """Describe the admitted live bundle using the shared measurement schema."""
        if self.manifest is None or self.artifact_root is None:
            raise RuntimeError("measurement provenance requires a manifest-backed runtime")
        root = self.artifact_root
        build_path = root / BUILD_FILENAME
        if not build_path.is_file():
            raise FileNotFoundError(f"TensorRT build receipt is missing: {build_path}")

        def identity(path: Path, *, digest: str | None = None) -> dict[str, Any]:
            resolved = path.resolve(strict=True)
            return {
                "path": str(resolved),
                "size_bytes": resolved.stat().st_size,
                "sha256": digest or sha256_file(resolved),
            }

        checkpoint_root = Path(self.contract.checkpoint).expanduser().resolve(strict=True)
        checkpoint_files: dict[str, dict[str, Any]] = {}
        weight_shards: dict[str, dict[str, Any]] = {}
        for relative_name, digest in self.manifest.checkpoint_files.items():
            path = checkpoint_root / relative_name
            record = {
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
            if relative_name in _MEASUREMENT_CHECKPOINT_FILES:
                checkpoint_files[relative_name] = record
            elif path.suffix in {".safetensors", ".bin"}:
                weight_shards[relative_name] = record
        if set(checkpoint_files) != _MEASUREMENT_CHECKPOINT_FILES or not weight_shards:
            raise RuntimeError(
                "N2 TensorRT manifest cannot satisfy the formal checkpoint identity: "
                f"files={sorted(checkpoint_files)}, shards={sorted(weight_shards)}"
            )

        engines = {
            _MEASUREMENT_ENGINE_ROLES[record.role]: identity(
                root / record.relative_path,
                digest=record.sha256,
            )
            for record in self.manifest.engines
        }
        if set(engines) != set(_MEASUREMENT_ENGINE_ROLES.values()):
            raise RuntimeError(f"N2 TensorRT runtime loaded an incomplete engine bundle: {engines}")

        return {
            "schema_version": "n2-runtime-artifacts-v1",
            "backend": backend,
            "precision": precision,
            "quantize_mode": "none",
            "tensorrt_version": self.manifest.runtime.tensorrt,
            "checkpoint": {
                "path": str(checkpoint_root),
                "files": checkpoint_files,
                "weight_shards": weight_shards,
            },
            "target_gpu": {
                "torch_name": self.manifest.runtime.gpu_name,
                "uuid": self.manifest.runtime.gpu_uuid,
                "compute_capability": list(self.manifest.runtime.compute_capability),
            },
            "build_manifest": identity(build_path),
            "loaded_engines": engines,
        }

    def transformer_input_dtypes(self, mode: str) -> dict[str, torch.dtype]:
        """Return the exact external input ABI for one admitted transformer engine."""
        if mode not in self.transformer_runners:
            raise RuntimeError(
                f"fixed-shape N2 TensorRT only supports explicit joint/action calls; got {mode!r}"
            )
        return {
            name: metadata["dtype"]
            for name, metadata in self.transformer_runners[mode].inputs.items()
        }

    def run_transformer(
        self,
        mode: str | None,
        *,
        packed: PackedSequence,
        prompt_embs: torch.Tensor,
        kv_cache: CondKVCache | None,
    ) -> torch.Tensor:
        if mode not in self.transformer_runners:
            raise RuntimeError(
                f"fixed-shape N2 TensorRT only supports explicit joint/action calls; got {mode!r}"
            )
        runner = self.transformer_runners[mode]
        skip_reason = runner.skip_reason(
            packed=packed,
            prompt_embs=prompt_embs,
            kv_cache=kv_cache,
        )
        if skip_reason is not None:
            raise RuntimeError(f"fixed-shape N2 TensorRT {mode} call rejected: {skip_reason}")
        if mode == "joint" and kv_cache is None:
            raise RuntimeError("joint TensorRT call requires a KV-cache destination")
        return runner(packed=packed, prompt_embs=prompt_embs, kv_cache=kv_cache)

    def encode_video(self, videos: torch.Tensor, *, tiled: bool) -> torch.Tensor:
        if self.vae_runner is None:
            raise RuntimeError("fixed-shape N2 TensorRT VAE runner is missing")
        if tiled:
            raise RuntimeError("fixed-shape N2 TensorRT VAE does not support tiling")
        tensors = {"videos": videos.contiguous()}
        skip_reason = self.vae_runner.skip_reason(tensors)
        if skip_reason is not None:
            raise RuntimeError(f"fixed-shape N2 TensorRT VAE call rejected: {skip_reason}")
        latents = self.vae_runner(tensors)
        if _host_checks_allowed() and not bool(torch.isfinite(latents).all().item()):
            raise RuntimeError("fixed-shape N2 TensorRT VAE produced non-finite output")
        return latents

    def validate_runtime(self, **kwargs: Any) -> None:
        _require_n2_runtime_contract(
            expected_num_cameras=self.num_cameras,
            expected_num_video_steps=self.num_video_steps,
            expected_num_action_steps=self.num_action_steps,
            **kwargs,
        )

    def validate_segment_layout(self, segment_counts: tuple[int, int, int]) -> None:
        _require_segment_layout(
            segment_counts,
            num_cameras=self.num_cameras,
            action_horizon=self.action_horizon,
            latent_shape=self.latent_shape,
        )
