# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""N2 Policy: joint video generation + action prediction.

Subclasses Gr00tPolicy for N2 refactor models. Calls model.generate() which
jointly denoises video and action, then decodes actions via the processor.
Also supports saving generated videos.
"""

# Module-wide exemptions for pre-existing baseline code. The per-step timing/config prints
# are the eval-run instrumentation; the broad except keeps a video-save failure from killing
# a rollout; `options` and `video_texts` are kept for base-class / public-signature parity.
# ruff: noqa: ARG002, BLE001, T201

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import os
import time
from typing import Any

import numpy as np
import torch
import transformers

from groot.core.data import schema, types
from groot.core.policy.benchmark_contract import resolve_policy_acceleration_state
from groot.core.policy.env_utils import env_bool as _env_bool
from groot.core.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype
from groot.core.policy.n2_acceleration import N2AccelerationOptions
from groot.core.policy.n2_codec import resolve_n2_codec_roundtrip

N2_VIEW_NAMES = ("top", "left", "right", "bottom")

# Codec used for the observation-video roundtrip at inference (see
# Gr00tN2Policy._codec_roundtrip_views). "h264" matches purchased/h264-converted
# datasets, "mp4v" matches in-house cv2.VideoWriter('mp4v') datasets, "none"
# disables the roundtrip (feed raw frames).
_MAX_CODEC_WORKERS = 4


def _decode_video(source: Any) -> np.ndarray:
    import av

    with av.open(source) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    return np.stack(frames, axis=0)


def _mp4v_roundtrip(video: np.ndarray) -> np.ndarray:
    import tempfile

    import cv2

    frame_count, height, width, _ = video.shape
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir="/dev/shm") as tmp:
        tmp_path = tmp.name
    try:
        writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
        try:
            for frame_index in range(frame_count):
                writer.write(cv2.cvtColor(video[frame_index], cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        return _decode_video(tmp_path)
    finally:
        os.unlink(tmp_path)


def _h264_roundtrip(video: np.ndarray) -> np.ndarray:
    import io

    import av

    frame_count, height, width, _ = video.shape
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("h264", rate=20)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "22"}
        for frame_index in range(frame_count):
            frame = av.VideoFrame.from_ndarray(video[frame_index], format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    buffer.seek(0)
    return _decode_video(buffer)


def _codec_roundtrip_clip(video: np.ndarray, *, codec: str) -> np.ndarray:
    if codec == "mp4v":
        return _mp4v_roundtrip(video)
    if codec == "h264":
        return _h264_roundtrip(video)
    raise ValueError(f"Unsupported codec roundtrip: {codec!r}")


def _codec_roundtrip_videos(vids: dict[str, Any], codec: str) -> float:
    """Roundtrip clips concurrently and return elapsed seconds."""
    if codec == "none" or not vids:
        return 0.0

    start = time.perf_counter()
    arrays = {key: np.asarray(value) for key, value in vids.items()}
    clips = [
        (key, batch_index, array[batch_index])
        for key, array in arrays.items()
        for batch_index in range(array.shape[0])
    ]
    if not clips:
        return 0.0
    roundtrip = partial(_codec_roundtrip_clip, codec=codec)
    with ThreadPoolExecutor(max_workers=min(_MAX_CODEC_WORKERS, len(clips))) as executor:
        decoded_clips = executor.map(roundtrip, (clip[2] for clip in clips))

    decoded_by_key: dict[str, list[np.ndarray]] = {key: [] for key in arrays}
    for (key, _, _), decoded in zip(clips, decoded_clips, strict=True):
        decoded_by_key[key].append(decoded)
    for key in arrays:
        vids[key] = np.stack(decoded_by_key[key], axis=0).astype(np.uint8, copy=False)
    return time.perf_counter() - start


def _env_int(name: str, *, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw)


def _resolve_constructor_env_int(name: str, *, constructor_value: int | None, default: int) -> int:
    """Resolve an integer as constructor argument, environment, then default."""
    if constructor_value is not None:
        return int(constructor_value)
    return _env_int(name, default=default)


def _env_float(name: str, *, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return float(raw)


_resolve_codec_roundtrip = resolve_n2_codec_roundtrip


def _normalize_n2rl_config_for_inference(config: Any) -> Any:
    """Map legacy actor-only BC modes onto the loadable inference wrapper."""
    if getattr(config, "model_type", None) == "GrootN2RL" and getattr(
        config, "loss_mode", None
    ) in {"BC", "BC_NORMAL_ONLY", "BC_CONTINUOUS_WEIGHTED"}:
        config.loss_mode = "BC_QUALITY_WEIGHTED"
    return config


def apply_torch_compile(model, mode: str = "reduce-overhead") -> None:
    """Compile the predict()-path entry points of a GrootN2 model.

    ``torch.compile(module)`` only intercepts the ``module(...)``/``forward``
    entry, but ``predict()`` never enters through forward: the DiT is called
    via ``transformer._forward_with_routing_indices`` (GrootN2._call_transformer)
    and the VAE via ``vae.model.encode`` (WanVideoVAE.single_encode). Plain
    methods on an ``OptimizedModule`` are delegated back to the original
    uncompiled module, so whole-module compilation was a silent no-op here —
    compile the bound methods instead.
    """
    model.vae.model.encode = torch.compile(model.vae.model.encode, mode=mode, dynamic=False)
    # NOTE(FY): dynamic=True is needed for using kv cache — the action-only
    # forward after the first joint step has a different sequence length.
    model.transformer._forward_with_routing_indices = torch.compile(  # noqa: SLF001
        model.transformer._forward_with_routing_indices,  # noqa: SLF001
        mode=mode,
    )


def _install_inference_acceleration(
    policy: "Gr00tN2Policy",
    *,
    options: N2AccelerationOptions,
) -> None:
    """The policy-side acceleration seam, called once at model wrap time.

    Twin of the trainer's ``_install_train_acceleration``
    (``models/n2/setup.py``): activation comes from the policy's ctor/env
    knobs, never from the checkpoint's saved config, and the
    ``_accel_installed`` stamp keeps one process from layering both installers
    onto the same model. Order discipline lives here too: module swaps
    (compile now, TensorRT engines later) must precede graph capture, so the
    runner is only armed here and captures lazily on the first predict.
    Compile and manual graph capture need mode="default" together —
    reduce-overhead's cudagraph trees cannot nest inside a manual capture.
    """
    model = policy.model
    installed = getattr(model, "_accel_installed", None)
    if installed is not None:
        raise RuntimeError(f"model already carries {installed!r} acceleration")
    trt_installed = getattr(model, "n2_trt_runtime", None) is not None
    options.validate()
    if trt_installed != options.use_tensorrt:
        raise RuntimeError(
            "loaded TensorRT runtime does not match the resolved acceleration options"
        )
    if trt_installed and options.use_torch_compile:
        raise RuntimeError("TensorRT and torch.compile cannot be installed together")
    if model.config.token_merge_kernel is not None:
        if options.use_torch_compile:
            raise RuntimeError("token merge does not support torch.compile")
        if options.use_cuda_graph and getattr(model, "n2_trt_runtime", None) is None:
            raise RuntimeError("token-merge CUDA graph requires an admitted TensorRT runtime")
    if options.use_torch_compile:
        print(f"[Gr00tN2Policy] Compiling model with mode: {options.torch_compile_mode}")
        apply_torch_compile(model, mode=options.torch_compile_mode)
    policy._use_predict_graph = options.use_cuda_graph  # noqa: SLF001 - installer owns attrs.
    policy._graph_runner = None  # noqa: SLF001
    policy._graph_capture_error = None  # noqa: SLF001
    policy._graph_capture_count = 0  # noqa: SLF001
    policy._graph_recapture_count = 0  # noqa: SLF001
    policy._graph_replay_count = 0  # noqa: SLF001
    policy.inference_acceleration_state = options.runtime_state()
    if trt_installed or options.use_torch_compile or options.use_cuda_graph:
        model._accel_installed = "inference"  # noqa: SLF001


class Gr00tN2Policy(Gr00tPolicy):
    """Policy for GrootN2: joint video + action inference.

    Calls model.generate() which produces both video latents and action predictions.
    Actions are decoded through the processor's StateActionProcessor.
    Video latents can be decoded and saved via save_video().
    """

    def __init__(
        self,
        embodiment_tag: schema.EmbodimentTag,
        model_path: str,
        *,
        num_inference_timesteps: int | None = None,
        video_guidance_scale: float = 1.0,
        action_guidance_scale: float = 1.0,
        shift: float = 5.0,
        save_video_dir: str | None = None,
        enable_n2_pred_comparison: bool = False,
        seed: int = 42,
        device: int | str = "cuda:0",
        strict: bool = True,
        use_torch_compile: bool = False,
        torch_compile_mode: str = "reduce-overhead",
        trt_engine_dir: str | None = None,
        use_cuda_graph: bool = False,
        codec_roundtrip: str = "mp4v",
    ):
        # Import to register the N2 model (pipeline setup runs AutoModel
        # registration as a side effect).
        import groot.core.training.models.n2

        # Internal builds include this package for GrootN2RL AutoModel registration,
        # while the curated OSS package intentionally excludes RL models. Suppress
        # only the missing OSS package; missing RL dependencies must still fail.
        # TODO(ryhalabi): Remove this conditional import once internal and OSS model  # noqa: TD003
        # use the same dependency-safe registration entry point.
        try:
            import groot.core.training.models.rl  # noqa: F401
        except ModuleNotFoundError as exc:
            if exc.name != "groot.core.training.models.rl":
                raise

        # Skip Gr00tPolicy.__init__ (it assumes action_head exists) — do the
        # essential setup directly.
        super(Gr00tPolicy, self).__init__(strict=strict)

        self.num_inference_steps = _resolve_constructor_env_int(
            "GROOT_N2_NUM_ACTION_STEPS",
            constructor_value=num_inference_timesteps,
            default=4,
        )
        self.video_guidance_scale = video_guidance_scale
        self.action_guidance_scale = action_guidance_scale
        self.shift = _env_float("GROOT_N2_SHIFT", default=shift)
        self.seed = seed
        self.kv_cache_cond = _env_bool("GROOT_N2_KV_CACHE_COND", default=True)
        self.joint_denoise = _env_bool("GROOT_N2_JOINT_DENOISE", default=True)
        self.num_video_steps = _env_int("GROOT_N2_NUM_VIDEO_STEPS", default=2)
        self.codec_roundtrip = _resolve_codec_roundtrip(codec_roundtrip)
        image_workers = _env_int("GROOT_N2_IMAGE_PREPROCESS_WORKERS", default=1)
        if not 1 <= image_workers <= 4:
            raise ValueError("GROOT_N2_IMAGE_PREPROCESS_WORKERS must be between 1 and 4")
        acceleration = N2AccelerationOptions.resolve(
            trt_engine_dir=trt_engine_dir,
            use_cuda_graph=use_cuda_graph,
            use_torch_compile=use_torch_compile,
            torch_compile_mode=torch_compile_mode,
        )
        if acceleration.use_cuda_graph and (
            video_guidance_scale != 1.0 or action_guidance_scale != 1.0
        ):
            raise ValueError(
                "predict CUDA graph requires video_guidance_scale=1.0 and "
                "action_guidance_scale=1.0 because negative text conditioning is not "
                "part of the fixed replay input"
            )

        from groot.core.training.models.n2.utils.cosmos_model_deps import ensure_cosmos_model_deps

        model_dir = ensure_cosmos_model_deps(model_path)
        # RL subclasses lazily reload the actor for preprocessing/critic identity.
        # Those loads must use the same immutable-source runtime view as this one.
        self._model_path = model_dir
        model_config = transformers.AutoConfig.from_pretrained(model_dir)
        # Inference immediately unwraps the base actor. Normalize only the in-memory load config;
        # the checkpoint remains an immutable record of the training recipe that produced it.
        model_config = _normalize_n2rl_config_for_inference(model_config)
        model = transformers.AutoModel.from_pretrained(model_dir, config=model_config)
        # GrootN2RL actor checkpoints only wrap the base GrootN2 actor; inference
        # uses the inner model's predict/vae/decode_latents APIs directly.
        if (
            getattr(model, "wrapped_model", None) is not None
            and type(model).__name__ == "GrootN2RL"
        ):
            model = model.wrapped_model
        model.eval()
        if acceleration.use_tensorrt:
            # The eager transformer/VAE are replaced by TRT. Keep the text
            # tower on CPU until its serving residency policy is installed.
            model.to(dtype=torch.bfloat16)
            for name, child in model.named_children():
                if name not in {"text_encoder", "transformer", "vae"}:
                    child.to(device=device)
            if list(model.parameters(recurse=False)) or list(model.buffers(recurse=False)):
                raise RuntimeError("N2 TRT placement requires module-owned parameters/buffers")
        else:
            model.to(device=device, dtype=torch.bfloat16)
        self.model = model
        trt_runtime = None
        if acceleration.use_tensorrt:
            from groot.deployment.n2_trt.contract import N2TensorRTContract
            from groot.deployment.n2_trt.installer import install_n2_tensorrt

            contract = N2TensorRTContract.from_checkpoint(
                model_dir,
                embodiment=embodiment_tag.value,
                video_denoising_steps=self.num_video_steps,
                action_denoising_steps=self.num_inference_steps,
                joint_denoise=self.joint_denoise,
                conditional_kv_cache=self.kv_cache_cond,
            )
            trt_runtime = install_n2_tensorrt(
                model,
                acceleration.trt_engine_dir,
                contract=contract,
                expected_codec=self.codec_roundtrip,
            )
        _install_inference_acceleration(
            self,
            options=acceleration,
        )
        print(
            f"[Gr00tN2Policy] acceleration: "
            f"trt_engines={'on' if acceleration.use_tensorrt else 'off'}, "
            f"predict_cuda_graph={'on' if acceleration.use_cuda_graph else 'off'} "
            f"(typed constructor option), "
            f"torch_compile={'on' if acceleration.use_torch_compile else 'off'}"
        )

        self.processor: types.BaseProcessor = transformers.AutoProcessor.from_pretrained(model_dir)
        self.processor.eval()
        if image_workers != 1:
            self.processor.set_inference_image_workers(image_workers)

        self.embodiment_tag = embodiment_tag
        self.modality_configs = self._load_modality_configs()
        self.collate_fn = self.processor.collator

        language_keys = self.modality_configs["language"].modality_keys
        self.language_key = language_keys[0]

        print(
            f"[Gr00tN2Policy] inference config: "
            f"num_inference_timesteps={self.num_inference_steps}, "
            f"num_video_steps={self.num_video_steps}, "
            f"shift={self.shift}, "
            f"joint_denoise={self.joint_denoise}, "
            f"kv_cache_cond={self.kv_cache_cond}, "
            f"codec_roundtrip={self.codec_roundtrip}, "
            f"image_preprocess_workers={image_workers}"
        )
        acceleration_state = resolve_policy_acceleration_state(
            self,
            use_torch_compile=use_torch_compile,
        )
        self.benchmark_runtime_contract = {
            **acceleration_state,
            "precision": "bf16",
            "policy_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "codec": self.codec_roundtrip,
            "image_preprocess_workers": image_workers,
            "num_video_steps": self.num_video_steps,
            "num_action_steps": self.num_inference_steps,
            "shift": self.shift,
            "joint_denoise": self.joint_denoise,
            "kv_cache_cond": self.kv_cache_cond,
            "video_guidance_scale": self.video_guidance_scale,
            "action_guidance_scale": self.action_guidance_scale,
            "seed": self.seed,
        }
        if trt_runtime is not None:
            self.benchmark_runtime_contract["text_encoder_offload"] = dict(
                model.n2_text_encoder_diagnostics
            )
            self.benchmark_runtime_contract["runtime_artifacts"] = (
                trt_runtime.measurement_artifacts(
                    backend=acceleration.backend,
                    precision="bf16",
                )
            )
        self._publish_cuda_graph_lifecycle()

        # Video buffer for save_video
        self.video_latents_buffer: list[torch.Tensor] = []
        self.save_video_dir = save_video_dir
        self.enable_n2_pred_comparison = enable_n2_pred_comparison
        self._step_count = 0

        # Cached text embeddings keyed by the batch language strings
        # (json-dumped list). Reused when the incoming languages match;
        # invalidated on language change and on reset().
        self._cached_prompt_embs: torch.Tensor | None = None
        self._cached_language_dict: str | None = None

    def _run_predict(self, collated_inputs: Any, prompt_embs: Any) -> dict[str, Any]:
        """Run the admitted prediction strategy behind one stable policy seam.

        Eager/TensorRT execution uses ``model.predict``; CUDA-graph execution
        uses ``_run_predict_graph``. Keeping the dispatch here preserves one
        logical model-computation boundary while the codec roundtrip,
        preprocessing, prompt cache, and action decode remain unchanged.
        """
        # In-process RL adapters can construct the policy from already-loaded
        # components without running the deployment acceleration installer.
        # That legacy path is explicitly eager unless it installs graph state.
        if getattr(self, "_use_predict_graph", False):
            return self._run_predict_graph(collated_inputs, prompt_embs)
        with torch.inference_mode():
            return self.model.predict(
                collated_inputs,
                num_action_steps=self.num_inference_steps,
                num_video_steps=self.num_video_steps,
                video_guidance_scale=self.video_guidance_scale,
                action_guidance_scale=self.action_guidance_scale,
                shift=self.shift,
                seed=self.seed,
                predict_video=self.save_video_dir is not None or self.enable_n2_pred_comparison,
                joint_denoise=self.joint_denoise,
                kv_cache_cond=self.kv_cache_cond,
                prompt_embs=prompt_embs,
                step_callback=getattr(self, "step_callback", None),
            )

    def _run_predict_graph(self, collated_inputs: Any, prompt_embs: Any = None) -> dict[str, Any]:
        """Replay a captured CUDA graph of the denoise, capturing on first use.

        Prompt encoding happens outside capture. Replay copies embeddings into
        an owned static buffer while preserving the captured key, shape, dtype,
        device, and padding contract.
        """
        if self.save_video_dir is not None or self.enable_n2_pred_comparison:
            raise RuntimeError(
                "the predict CUDA graph bakes predict_video=False; disable "
                "use_cuda_graph to save videos or compare predictions"
            )
        if self._graph_capture_error is not None:
            raise RuntimeError(
                "A previous CUDA graph operation failed and left this policy process unsafe "
                "to reuse. Terminate it completely and start a fresh process. "
                f"First graph failure: {self._graph_capture_error}"
            )
        from groot.core.policy.n2_predict_graph import (
            PredictGraphRunner,
            make_steady_state_inputs,
        )

        payload = collated_inputs if isinstance(collated_inputs, dict) else collated_inputs.data
        payload = payload.get("inputs", payload)
        # The collator leaves tensors on the host -- _rec_to_dtype casts dtype but
        # never moves device, and model.predict does the transfer itself. Capturing
        # from host tensors makes the model do a CPU->GPU copy inside the capture,
        # which is illegal ("Cannot copy between CPU and CUDA tensors during CUDA
        # graph capture"). Move once here so both the capture and the per-call
        # copy_ into the static buffers are device-to-device.
        device = self.model.device
        payload = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in payload.items()
        }

        if prompt_embs is None:
            captured, prompt_embs = make_steady_state_inputs(self.model, payload, device)
        else:
            captured = {k: v for k, v in payload.items() if k != "text_negative"}

        if self._graph_runner is None:
            try:
                self._graph_runner = PredictGraphRunner(
                    self.model,
                    captured,
                    prompt_embs,
                    {
                        "num_action_steps": self.num_inference_steps,
                        "num_video_steps": self.num_video_steps,
                        "video_guidance_scale": self.video_guidance_scale,
                        "action_guidance_scale": self.action_guidance_scale,
                        "shift": self.shift,
                        "seed": self.seed,
                        "predict_video": False,
                        "joint_denoise": self.joint_denoise,
                        "kv_cache_cond": self.kv_cache_cond,
                    },
                )
                previous_captures = getattr(self, "_graph_capture_count", 0)
                if previous_captures:
                    self._graph_recapture_count = getattr(self, "_graph_recapture_count", 0) + 1
                self._graph_capture_count = previous_captures + 1
                Gr00tN2Policy._publish_cuda_graph_lifecycle(self)
            except Exception as exc:
                self._graph_capture_error = f"{type(exc).__name__}: {exc}"
                Gr00tN2Policy._publish_cuda_graph_lifecycle(self)
                raise

        try:
            # Replay exactly the contract captured above.  The processor may
            # leave ``text_negative`` in its payload, but steady-state graph
            # execution consumes the already-materialized prompt embeddings
            # and deliberately excludes that source tensor.
            result = self._graph_runner(captured, prompt_embs=prompt_embs)
            self._graph_replay_count = getattr(self, "_graph_replay_count", 0) + 1
            Gr00tN2Policy._publish_cuda_graph_lifecycle(self)
        except Exception as exc:
            self._graph_capture_error = f"{type(exc).__name__}: {exc}"
            Gr00tN2Policy._publish_cuda_graph_lifecycle(self)
            raise
        # The graph's outputs are only the captured tensors; hand the prompt
        # embeddings back so the caller's prompt cache keeps working as before.
        return {**result, "prompt_embs": prompt_embs}

    def _publish_cuda_graph_lifecycle(self) -> None:
        """Publish live graph counters consumed by the formal measurement gate."""
        if not getattr(self, "_use_predict_graph", False):
            return
        contract = getattr(self, "benchmark_runtime_contract", None)
        if not isinstance(contract, dict):
            return
        contract["cuda_graph_lifecycle"] = {
            "schema_version": "n2-cuda-graph-lifecycle-v1",
            "capture_count": getattr(self, "_graph_capture_count", 0),
            "recapture_count": getattr(self, "_graph_recapture_count", 0),
            "replay_count": getattr(self, "_graph_replay_count", 0),
        }

    def server_readiness(self) -> dict[str, Any]:
        """Report whether this process is safe to arm for action serving.

        CUDA-graph capture needs a real, processor-shaped observation and task
        prompt, so the server cannot complete it at socket startup. Operators
        must send one explicit warmup request and wait for a successful replay.
        A capture/replay failure or recapture marks the process unhealthy until
        it is restarted.
        """
        encoder = getattr(getattr(self, "model", None), "n2_prompt_encoder", None)
        if getattr(encoder, "failure", None) is not None:
            return {
                "schema_version": "n2-server-readiness-v1",
                "status": "failed",
                "healthy": False,
                "ready": False,
                "message": "text encoding failed; restart the policy server",
            }
        if not getattr(self, "_use_predict_graph", False):
            return {
                "schema_version": "n2-server-readiness-v1",
                "status": "ready",
                "healthy": True,
                "ready": True,
                "message": "policy is ready",
            }

        captures = getattr(self, "_graph_capture_count", 0)
        recaptures = getattr(self, "_graph_recapture_count", 0)
        replays = getattr(self, "_graph_replay_count", 0)
        failure = getattr(self, "_graph_capture_error", None)
        lifecycle = {
            "capture_count": captures,
            "recapture_count": recaptures,
            "replay_count": replays,
        }
        if failure is not None or recaptures != 0:
            reason = failure or "CUDA graph recaptured after its initial fixed-shape capture"
            return {
                "schema_version": "n2-server-readiness-v1",
                "status": "failed",
                "healthy": False,
                "ready": False,
                "message": f"restart required: {reason}",
                "cuda_graph_lifecycle": lifecycle,
            }

        ready = captures == 1 and replays >= 1
        return {
            "schema_version": "n2-server-readiness-v1",
            "status": "ready" if ready else "warmup_required",
            "healthy": True,
            "ready": ready,
            "message": (
                "CUDA graph captured and replayed"
                if ready
                else "submit one real-observation warmup request before arming"
            ),
            "cuda_graph_lifecycle": lifecycle,
        }

    def _codec_roundtrip_views(self, vids: dict[str, Any]) -> float:
        """Encode→decode each observation video through the configured codec.

        N2 ckpts are trained on codec-encoded LeRobot datasets, and the visual
        encoder overfits to codec artifacts (DCT block quantization, chroma
        subsampling). Feeding raw camera frames at deployment causes predictions
        to collapse to near-constant action targets even though the weights are
        fine. ``self.codec_roundtrip`` selects which codec to mimic:

        - ``"mp4v"``: ``cv2.VideoWriter('mp4v')`` @ 30fps with RGB->BGR — the exact
          encoder the in-house YAM recorder uses (in-house / cv2-mp4v datasets).
        - ``"h264"``: libx264 at crf=22 (purchased / h264-converted datasets).
        - ``"none"``: skip the roundtrip and feed raw frames.

        Mutates ``vids`` in place (values are (B, T, H, W, C) uint8 arrays).
        """
        return _codec_roundtrip_videos(vids, self.codec_roundtrip)

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Generate video + action jointly, return decoded actions."""
        t0 = time.perf_counter()
        # Codec roundtrip on observation videos (no-op when codec_roundtrip="none").
        codec_time = self._codec_roundtrip_views(observation.get("video", {}))
        t_codec = time.perf_counter()

        observation = self._post_process_observation(observation)
        unbatched_observations = self._unbatch_observation(observation)
        processed_inputs = []

        states = []
        languages = []
        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            states.append(vla_step_data.states)
            languages.append(vla_step_data.languages)
            messages = [{"type": types.MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))

        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)
        t_preprocess = time.perf_counter()

        language_dict = json.dumps(languages)
        prompt_embs = (
            self._cached_prompt_embs if language_dict == self._cached_language_dict else None
        )
        if prompt_embs is None:
            print(f"[Gr00tN2Policy] Prompt cache MISS, encoding new text from {language_dict}")

        result = self._run_predict(collated_inputs, prompt_embs)
        t_predict = time.perf_counter()
        self._cached_prompt_embs = result.get("prompt_embs")
        self._cached_language_dict = language_dict

        video_latents = result.get("video_latents")
        normalized_action = result.get("action_pred")
        if normalized_action is not None:
            normalized_action = normalized_action.float()

        # Buffer video latents for save_video (not available in action_only mode)
        if video_latents is not None:
            self.video_latents_buffer.append(video_latents.detach())

        self._step_count += 1
        predicted_image = None
        if self.enable_n2_pred_comparison and self.video_latents_buffer:
            # Per-inference decoded frames for env-side pred vs real comparison images.
            predicted_image = self._decode_latest_video_frames()

        if normalized_action is None:
            return {}, {}

        # Decode actions
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )

        casted_action = {
            key: value.astype(np.float32) for key, value in unnormalized_action.items()
        }
        if predicted_image is not None:
            casted_action["predicted_image"] = predicted_image
        t_end = time.perf_counter()
        action_shapes = {k: v.shape for k, v in casted_action.items() if isinstance(v, np.ndarray)}
        if predicted_image is not None:
            action_shapes["predicted_image"] = {
                view: arr.shape for view, arr in predicted_image.items()
            }
        print(
            f"[Gr00tN2Policy] codec={codec_time:.3f}s  "
            f"preprocess={t_preprocess - t_codec:.3f}s  "
            f"predict={t_predict - t_preprocess:.3f}s  "
            f"postprocess={t_end - t_predict:.3f}s  "
            f"total={t_end - t0:.3f}s  "
            f"action_shapes={action_shapes}"
        )
        # Optional heads travel in info, never in the physical action vector.
        # Keep the batch dimension and use CPU arrays for the policy RPC codec.
        info = {}
        for key in (
            "completion_cdf",
            "completion_hazard_probability",
            "completion_tau",
            "completion_query_denoising_step",
            "completion_action_edge",
        ):
            if key in result:
                value = result[key]
                info[key] = (
                    value.detach().float().cpu().numpy() if torch.is_tensor(value) else value
                )
        return casted_action, info

    def _decode_latest_video_frames(self) -> dict[str, np.ndarray] | None:
        """Decode latest latent chunk to per-view pixel frames."""
        if not self.video_latents_buffer:
            return None
        step_latents = self.video_latents_buffer[-1]
        decoded_views = self.model.decode_latents(step_latents)
        if not decoded_views:
            return None
        result: dict[str, np.ndarray] = {}
        for v_idx, view_tensor in enumerate(decoded_views):
            if v_idx >= len(N2_VIEW_NAMES):
                break
            result[N2_VIEW_NAMES[v_idx]] = view_tensor[0].cpu().numpy()
        return result

    def _flush_video(self):
        """Decode buffered video latents and save to disk."""
        if not self.video_latents_buffer or not self.save_video_dir:
            self.video_latents_buffer.clear()
            return
        os.makedirs(self.save_video_dir, exist_ok=True)
        traj_id = len([f for f in os.listdir(self.save_video_dir) if f.endswith(".mp4")])
        save_path = os.path.join(self.save_video_dir, f"traj_{traj_id}.mp4")
        try:
            self.save_video(save_path, traj_id=traj_id)
            print(
                f"[Gr00tN2Policy] Saved generated video to {save_path} "
                f"({len(self.video_latents_buffer)} chunks)"
            )
        except Exception as e:
            print(f"[Gr00tN2Policy] Failed to save video: {e}")
        self.video_latents_buffer.clear()

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        self._flush_video()
        # The graph owns its prompt buffer; the bounded encoder LRU survives
        # rollout resets independently of this single-request fast cache.
        self._cached_prompt_embs = None
        self._cached_language_dict = None
        return super().reset(options)

    def save_video(
        self,
        save_path: str,
        traj_id: int,
        video_latents: list | None = None,
        video_texts: list[str] | None = None,
        fps: int = 8,
    ) -> None:
        """Decode video latents and save as a tiled multi-view mp4.

        For multiview models, all views are tiled horizontally into one frame.
        For single-view, saves the video directly.
        """
        from torchvision.io import write_video

        latents = video_latents if video_latents is not None else self.video_latents_buffer
        if not latents:
            return

        # Decode each step's latents separately (views are interleaved within each step),
        # then concatenate decoded frames per view across steps.
        n_views = self.model.max_n_cameras
        per_view_frames: list[list[np.ndarray]] = [[] for _ in range(n_views)]

        for step_latents in latents:
            decoded_views = self.model.decode_latents(step_latents)
            if not decoded_views:
                continue
            for v_idx, view_tensor in enumerate(decoded_views):
                per_view_frames[v_idx].append(view_tensor[0].cpu().numpy())

        if not per_view_frames[0]:
            return

        # Concatenate frames per view across steps: list of (T_total, H, W, C)
        view_frames = [np.concatenate(frames, axis=0) for frames in per_view_frames]

        if len(view_frames) == 1:
            # Single view: save directly
            tiled = view_frames[0]
        else:
            # Multiview: tile all views horizontally → (T, H, V*W, C)
            tiled = np.concatenate(view_frames, axis=2)

        if save_path.endswith(".mp4"):
            full_path = save_path
        else:
            full_path = f"{save_path.split('.mp4')[0]}_{traj_id}.mp4"

        os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
        write_video(full_path, torch.from_numpy(tiled), fps=fps)

    def get_modality_config(self) -> dict[str, Any]:
        return self.modality_configs
