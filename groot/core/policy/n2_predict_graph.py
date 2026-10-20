# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# ruff: noqa: SLF001 - capture-time freezing rebinds the model's private collaborators by design.
"""Whole-predict CUDA graph runner for GrootN2 steady-state inference.

Captures video preprocess + VAE encode + the full joint denoise loop
(schedulers, packing, kv-cache, DiT calls) into one manual
``torch.cuda.CUDAGraph``. Replay executes no Python, so the eager glue
between compiled regions disappears (GB300, G1 batch: 117ms -> 62ms per
step). Text encoding stays outside: pre-encoded ``prompt_embs`` are copied
into a graph-owned buffer (see ``make_steady_state_inputs``). Prompt contents
may change without recapture; their shape, dtype, and device may not.

Contract: the runner owns the model instance and the process. Capture-time
preparation rebinds a few collaborators to frozen, graph-safe equivalents
(schedulers, attention-mask builder, kv-cache slicing), keyed to the example
batch — the instance must not be reused for training or variable-shape
inference, and one process hosts one runner. Later calls must keep the
example batch's shapes, dtypes, devices, and padding layout; only tensor
contents may change. Compile the DiT/VAE with mode="default":
reduce-overhead's cudagraph trees cannot nest inside a manual capture.
"""

from __future__ import annotations

import copy
import logging
import time

import torch

logger = logging.getLogger(__name__)


def _capture_rng_states(device) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Snapshot ambient CPU and selected-device CUDA RNG state."""
    cpu_state = torch.get_rng_state().clone()
    cuda_state = None
    if torch.device(device).type == "cuda":
        cuda_state = torch.cuda.get_rng_state(device).clone()
    return cpu_state, cuda_state


def _restore_rng_states(
    device,
    states: tuple[torch.Tensor, torch.Tensor | None],
) -> None:
    """Restore ambient RNG state after capture preparation, including failures."""
    cpu_state, cuda_state = states
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


def _graph_safe_full(
    real_full,
    size,
    fill_value,
    *,
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
    pin_memory=False,
):
    """Implement ``torch.full`` with a device scalar without a host sync."""
    if (
        torch.is_tensor(fill_value)
        and fill_value.numel() == 1
        and out is None
        and layout is torch.strided
        and not pin_memory
    ):
        target_device = torch.device(device) if device is not None else fill_value.device
        target_dtype = dtype if dtype is not None else fill_value.dtype
        scalar = fill_value.to(device=target_device, dtype=target_dtype)
        result = scalar.reshape(()).expand(size).clone()
        result.requires_grad_(requires_grad)
        return result
    return real_full(
        size,
        fill_value,
        out=out,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
    )


def make_steady_state_inputs(model, inputs: dict, device) -> tuple[dict, torch.Tensor]:
    """Deployment steady state for the runner's inputs.

    prompt_embs pre-encoded once, and the text_negative encode dropped
    (unused while guidance scales are 1.0).
    """
    with torch.inference_mode():
        prompt_embs = model.encode_text(inputs["text"].to(device))
    stripped = {k: v for k, v in inputs.items() if k != "text_negative"}
    return stripped, prompt_embs


class PredictGraphRunner:
    def __init__(
        self,
        model,
        example_inputs: dict,
        prompt_embs: torch.Tensor,
        predict_kwargs: dict,
        warmup: int = 3,
    ):
        if warmup < 3:
            raise ValueError(f"predict CUDA graph requires at least 3 warmups, got {warmup}")
        self.model = model
        self._require_camera_padding_mask(example_inputs)
        self.device = prompt_embs.device
        self.prompt_embs = prompt_embs.clone()
        self.kwargs = dict(predict_kwargs)
        if (
            self.kwargs.get("video_guidance_scale") != 1.0
            or self.kwargs.get("action_guidance_scale") != 1.0
        ):
            raise ValueError(
                "predict CUDA graph requires unit video and action guidance scales; "
                "negative text conditioning is excluded from fixed replay inputs"
            )
        # Flat input dict: tensors become static graph buffers, everything
        # else (frame-count ints, id lists) is baked into the capture.
        self.static = {
            k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in example_inputs.items()
        }
        self._freeze_schedulers()
        self._freeze_attention_mask_builder()
        self._freeze_cache_slicing()
        self._freeze_conditioning_index()
        self._prepare_capture()
        self._capture(warmup)

    def __del__(self):
        # The graph replays kernels owned by the model's TensorRT execution
        # contexts. If the interpreter frees those first the graph's teardown
        # touches freed memory and the process dies with SIGSEGV at exit, after
        # the results are computed but before they are reported. Dropping the
        # graph here keeps it ahead of anything it points at.
        self.close()

    def close(self) -> None:
        """Release graph-owned TensorRT references before model teardown."""
        graph = getattr(self, "graph", None)
        if graph is not None:
            reset = getattr(graph, "reset", None)
            if reset is not None:
                reset()
            self.graph = None
        # Release external graph inputs only after destroying the graph.
        self._frozen_noise = {}

    def __call__(self, inputs: dict, *, prompt_embs: torch.Tensor, seed: int | None = None) -> dict:
        """Replay on ``inputs``; ``seed`` redraws the noise as ``model.predict(seed=seed)`` would."""
        self._validate_replay_contract(inputs, prompt_embs)
        with torch.cuda.device(self.device):
            # inference_mode: replay_prologue does inplace updates on the
            # generator's graph-safe state tensors, which are inference
            # tensors whenever inference-mode code created them (see
            # _capture). Only the copy+replay needs it.
            with torch.inference_mode():
                for key, dst in self.static.items():
                    if isinstance(dst, torch.Tensor):
                        dst.copy_(inputs[key])
                self.prompt_embs.copy_(prompt_embs)
                if seed is not None and seed != self._noise_seed:
                    self._redraw_noise(seed)
                conditioner = getattr(
                    getattr(getattr(self, "model", None), "n2_trt_runtime", None),
                    "conditioner",
                    None,
                )
                if conditioner is not None and not conditioner.is_valid:
                    # The prompt changed since capture: recondition the static prompt
                    # buffer outside the graph so the replayed action engine reads
                    # the current prompt's projection from the same stable buffer.
                    conditioner.refresh(self.prompt_embs)
                self.graph.replay()
            # Clone: the static buffers are overwritten by the next replay.
            # Outside inference_mode so callers get normal tensors, exactly
            # as before the replay wrap above.
            outputs = {
                k: v.clone() for k, v in self.static_out.items() if isinstance(v, torch.Tensor)
            }
        non_finite = [
            name
            for name, value in outputs.items()
            if value.is_floating_point() and not bool(torch.isfinite(value).all().item())
        ]
        if non_finite:
            raise RuntimeError(f"predict CUDA graph returned non-finite outputs: {non_finite}")
        return outputs

    def _redraw_noise(self, seed: int) -> None:
        """Refill the captured noise buffers in place with ``seed``'s draws, outside the graph.

        The capture froze predict()'s seeded draws as constant buffers. Drawing them again in
        predict()'s order from a Generator seeded with ``seed`` gives exactly the noise
        ``model.predict(seed=seed)`` would use, and the replay reads it from the same addresses.
        """
        generator = torch.Generator(device=self.device).manual_seed(seed)
        for key, shape, draw in self._noise_draws:
            self._frozen_noise[key].copy_(torch.randn(*shape, generator=generator, **draw))
        self._noise_seed = seed

    def _require_camera_padding_mask(self, inputs: dict) -> None:
        """Require explicit camera validity for padded TRT capture and replay."""
        model = getattr(self, "model", None)
        trt_runtime = getattr(model, "n2_trt_runtime", None)
        contract = getattr(trt_runtime, "contract", None)
        if (
            contract is not None
            and contract.padded_camera_slot_count > 0
            and inputs.get("video_padded_mask") is None
        ):
            raise RuntimeError(
                "predict CUDA graph requires video_padded_mask for a TRT contract "
                "with padded camera slots"
            )

    def _validate_replay_contract(
        self,
        inputs: dict,
        prompt_embs: torch.Tensor,
    ) -> None:
        """Reject any replay input that can alter the captured execution layout."""
        self._require_camera_padding_mask(inputs)
        expected_keys = set(self.static)
        actual_keys = set(inputs)
        if actual_keys != expected_keys:
            raise RuntimeError(
                "predict CUDA graph input keys changed: "
                f"missing={sorted(expected_keys - actual_keys)}, "
                f"extra={sorted(actual_keys - expected_keys)}"
            )
        for key, expected in self.static.items():
            actual = inputs[key]
            if isinstance(expected, torch.Tensor):
                self._validate_tensor_contract(key, actual, expected)
            elif type(actual) is not type(expected) or actual != expected:
                raise RuntimeError(
                    f"predict CUDA graph non-tensor input {key!r} changed: "
                    f"{actual!r} != {expected!r}"
                )
        self._validate_tensor_contract("prompt_embs", prompt_embs, self.prompt_embs)
        if not bool(torch.isfinite(prompt_embs).all().item()):
            raise RuntimeError("predict CUDA graph prompt embeddings must be finite")

        padded_mask = inputs.get("video_padded_mask")
        if padded_mask is not None and not torch.equal(
            padded_mask, self.static["video_padded_mask"]
        ):
            raise RuntimeError("predict CUDA graph padded camera layout changed")

    @staticmethod
    def _validate_tensor_contract(name: str, actual, expected: torch.Tensor) -> None:
        if not isinstance(actual, torch.Tensor):
            raise TypeError(
                f"predict CUDA graph input {name!r} must remain a tensor, "
                f"got {type(actual).__name__}"
            )
        mismatches = {}
        if tuple(actual.shape) != tuple(expected.shape):
            mismatches["shape"] = {"actual": tuple(actual.shape), "expected": tuple(expected.shape)}
        if actual.dtype != expected.dtype:
            mismatches["dtype"] = {"actual": str(actual.dtype), "expected": str(expected.dtype)}
        if actual.device != expected.device:
            mismatches["device"] = {
                "actual": str(actual.device),
                "expected": str(expected.device),
            }
        if mismatches:
            raise RuntimeError(f"predict CUDA graph input {name!r} changed: {mismatches}")

    # ── capture-time freezing ──────────────────────────────────────────────
    # Each helper replaces one per-call construction whose CPU->GPU copies or
    # data-dependent host syncs are illegal during capture with a constant
    # prepared here, outside the graph.

    def _prepare_capture(self) -> None:
        """Finish eager work without evicting an active CUDA memory pool.

        PyTorch 2.7 uses the allocator's capture bookkeeping for CUDA graphs
        and ``torch.cuda.MemPool``. Synchronization is sufficient here; the
        context-manager capture helper's unconditional cache eviction is not.
        """
        torch.cuda.synchronize(self.device)

    def _freeze_schedulers(self) -> None:
        """Freeze scheduler construction to graph-safe prototypes.

        Schedulers are rebuilt per predict() and set_timesteps copies CPU
        schedules to GPU. The schedules are constants for fixed (num_steps,
        shift): prebuild one prototype per scheduler and hand out frozen
        copies — fresh python state, shared on-device tensors, no-op
        set_timesteps.
        """
        import groot.core.training.models.n2.model as n2_model

        def freeze(proto):
            for name, value in list(vars(proto).items()):
                if torch.is_tensor(value):
                    setattr(proto, name, value.to(self.device))
            tensor_state = {k: v for k, v in vars(proto).items() if torch.is_tensor(v)}
            other_state = {k: v for k, v in vars(proto).items() if not torch.is_tensor(v)}
            cls = type(proto)

            def factory(*_args, **_kwargs):
                sched = object.__new__(cls)
                sched.__dict__.update(copy.deepcopy(other_state))
                sched.__dict__.update(tensor_state)
                sched.set_timesteps = lambda *_a, **_k: None
                return sched

            return factory

        vid_proto = self.model._make_video_scheduler(self.model.config)
        vid_proto.set_timesteps(
            self.kwargs["num_video_steps"], device=self.device, shift=self.kwargs["shift"]
        )
        self.model._make_video_scheduler = freeze(vid_proto)

        act_proto = n2_model.BetaFlowMatchEulerScheduler(self.model.action_scheduler.config)
        act_proto.set_timesteps(self.kwargs["num_action_steps"], device=self.device)
        # Module-global rebind — part of the one-runner-per-process contract.
        n2_model.BetaFlowMatchEulerScheduler = freeze(act_proto)

    def _freeze_attention_mask_builder(self) -> None:
        """Memoize the BlockMask builder.

        The packer rebuilds the BlockMask from a CPU pattern table per call.
        It is constant per argument signature under the fixed-layout
        contract: warmup populates the memo, capture and replay hit it.
        """
        import groot.core.training.models.n2.components.sequence_packing as packing

        real_build = packing._build_attention_mask
        cache: dict = {}

        def describe(x):
            return tuple(x.shape) if torch.is_tensor(x) else repr(x)

        def cached_build(*args, **kwargs):
            key = (
                tuple(describe(x) for x in args),
                tuple((name, describe(x)) for name, x in sorted(kwargs.items())),
            )
            if key not in cache:
                cache[key] = real_build(*args, **kwargs)
            return cache[key]

        packing._build_attention_mask = cached_build

    def _freeze_cache_slicing(self) -> None:
        """Replace the kv-cache slice's host syncs with a cached index.

        _slice_cache_to_static_prefix derives its keep-index with a tensor
        assert + nonzero() — both host syncs. The index is constant under the
        fixed-layout contract: derive it during warmup, then index_select
        with the cached index (identical ascending order). Padded TRT layouts
        gather each shared fine/coarse engine slab once, retaining layer views
        so preparing the cached-action bindings needs no additional stack.
        """
        import groot.core.training.models.n2.model as n2_model

        model = self.model
        idx_cache: dict = {}
        # TRT's joint output slabs outlive the immediately following cached
        # action call. Keep prefix views or gather a shared slab once. Other
        # graph backends retain independent per-layer copies.
        trt_cache = getattr(model, "n2_trt_runtime", None) is not None

        def graph_safe_slice(
            cache, modality_ids, token_valid_mask=None, token_merge=None, extra_keep_modalities=()
        ):
            if extra_keep_modalities:
                raise ValueError(
                    "predict CUDA graph does not support cacheable AdditionalTokenGroups"
                )
            key = (tuple(modality_ids[0].shape), token_merge is not None)
            if key not in idx_cache:
                table = model._resolve_attention_mask_pattern().to_table()
                action_row = table[3]
                keep_modalities = []
                if action_row[0]:
                    keep_modalities.append(n2_model.MODALITY_VIDEO_COND)
                if action_row[1]:
                    keep_modalities.append(n2_model.MODALITY_VIDEO)
                if action_row[2]:
                    keep_modalities.append(n2_model.MODALITY_STATE)
                ids = modality_ids[0]
                keep = torch.zeros_like(ids, dtype=torch.bool)
                for m in keep_modalities:
                    keep |= ids == m
                if token_valid_mask is not None:
                    assert torch.all(token_valid_mask == token_valid_mask[:1])
                    keep &= token_valid_mask[0, : keep.shape[0]]
                fine_idx = keep.nonzero(as_tuple=True)[0]
                coarse_idx = None
                if token_merge is not None:
                    fine_idx, coarse_idx = token_merge.keep_indices_per_segment(keep)
                prefix_flags = tuple(
                    index is None
                    or torch.equal(index, torch.arange(index.numel(), device=index.device))
                    for index in (fine_idx, coarse_idx)
                )
                idx_cache[key] = (fine_idx, coarse_idx, prefix_flags)
            fine_idx, coarse_idx, prefix_flags = idx_cache[key]
            sliced = n2_model.CondKVCache()
            gathered_layers: dict = {}
            if trt_cache:
                from groot.deployment.n2_trt.runtime import _gather_cache_layers

                # Token-merge fine layers straddle the coarse range, but their
                # engine outputs are consecutive views of the same fine slab.
                for reduced, keep_idx in ((False, fine_idx), (True, coarse_idx)):
                    if keep_idx is None or prefix_flags[1 if reduced else 0]:
                        continue
                    layer_indices = [
                        index
                        for index in range(len(cache))
                        if (token_merge is not None and token_merge.is_reduced_layer(index))
                        == reduced
                    ]
                    keys = _gather_cache_layers(
                        [cache.key_cache[index] for index in layer_indices], keep_idx
                    )
                    values = _gather_cache_layers(
                        [cache.value_cache[index] for index in layer_indices], keep_idx
                    )
                    gathered_layers.update(
                        zip(layer_indices, zip(keys, values, strict=True), strict=True)
                    )
            for layer_idx, (k_layer, v_layer) in enumerate(
                zip(cache.key_cache, cache.value_cache, strict=False)
            ):
                keep_idx = (
                    coarse_idx
                    if token_merge is not None and token_merge.is_reduced_layer(layer_idx)
                    else fine_idx
                )
                reduced = token_merge is not None and token_merge.is_reduced_layer(layer_idx)
                can_borrow = trt_cache and prefix_flags[1 if reduced else 0]
                if layer_idx in gathered_layers:
                    selected_key, selected_value = gathered_layers[layer_idx]
                    sliced.key_cache.append(selected_key)
                    sliced.value_cache.append(selected_value)
                elif can_borrow:
                    # Only metadata operations during capture. The runner
                    # materializes dense bindings before the next joint call
                    # can overwrite this engine-owned storage.
                    sliced.key_cache.append(k_layer[:, :, : keep_idx.numel(), :])
                    sliced.value_cache.append(v_layer[:, :, : keep_idx.numel(), :])
                else:
                    sliced.key_cache.append(k_layer.index_select(2, keep_idx).clone())
                    sliced.value_cache.append(v_layer.index_select(2, keep_idx).clone())
            return sliced

        model._slice_cache_to_static_prefix = graph_safe_slice

    def _freeze_conditioning_index(self) -> None:
        """Memoize the conditioning-token index the frozen-prefix kv-cache is built from.

        ``conditioning_token_index`` derives it with ``nonzero()`` — a host sync. The
        index is constant under the fixed-layout contract: derive it during warmup,
        then serve the cached tensor during capture and replay.
        """
        import groot.core.training.models.n2.model as n2_model

        real_index = n2_model.conditioning_token_index
        idx_cache: dict = {}

        def cached_index(modality_ids):
            key = (tuple(modality_ids[0].shape), modality_ids.dtype)
            if key not in idx_cache:
                idx_cache[key] = real_index(modality_ids)
            return idx_cache[key]

        # Module-global rebind — part of the one-runner-per-process contract.
        n2_model.conditioning_token_index = cached_index

    # ── capture ────────────────────────────────────────────────────────────

    def _capture_predict(self) -> dict:
        """Capture one predict call without the cache-evicting graph wrapper."""
        self.graph.capture_begin()
        try:
            return self.model.predict(
                self.static,
                prompt_embs=self.prompt_embs,
                **self.kwargs,
            )
        finally:
            # End capture even when predict raises so allocator pool routing is
            # not deliberately left active. The policy failure latch still
            # requires a fresh process after any capture error.
            self.graph.capture_end()

    def _capture(self, warmup: int) -> None:
        t_start = time.perf_counter()
        real_manual_seed = torch.manual_seed
        real_generator = torch.Generator
        real_randn = torch.randn
        real_full = torch.full
        rng_states = _capture_rng_states(self.device)
        # These are allocated during warmup, outside the graph-private pool.
        # CUDA captures pointers, not Python tensor ownership. Keep them alive
        # across prompt encoding, allocator reuse, and empty_cache calls.
        self._frozen_noise: dict = {}
        frozen = self._frozen_noise
        # The seeded draws, in predict()'s order, so _redraw_noise can refill them per request.
        self._noise_draws: list[tuple[tuple, tuple, dict]] = []

        # predict() uses a local Generator so open-loop evaluation does not
        # reset the process-global RNG. CUDA forbids constructing that
        # generator while stream capture is active. Create and seed it once
        # before warmup/capture; later model calls receive this lightweight
        # proxy, while frozen_randn unwraps the real generator only for the
        # first eager draw of each fixed tensor.
        noise_seed = self.kwargs["seed"]
        noise_generator = real_generator(device=self.device).manual_seed(noise_seed)
        self._noise_seed = noise_seed

        class FrozenGenerator:
            def manual_seed(self, seed):
                if seed != noise_seed:
                    raise RuntimeError(
                        "predict CUDA graph seed changed during fixed-noise capture: "
                        f"{seed} != {noise_seed}"
                    )
                return self

        frozen_generator = FrozenGenerator()

        def frozen_generator_factory(*args, **kwargs):
            if len(args) > 1 or set(kwargs) - {"device"}:
                raise TypeError(
                    "unsupported torch.Generator call during predict CUDA graph capture"
                )
            requested_device = kwargs.get("device", args[0] if args else self.device)
            if torch.device(requested_device) != torch.device(self.device):
                raise RuntimeError(
                    "predict CUDA graph generator device changed: "
                    f"{requested_device} != {self.device}"
                )
            return frozen_generator

        def frozen_randn(*shape, **kwargs):
            # predict() reseeds and draws the same noise every call; captured
            # RNG would re-randomize per replay. Draw once with the real RNG
            # (first warmup call, original order) and freeze as constants.
            if len(shape) == 1 and isinstance(shape[0], tuple | list):
                shape = tuple(shape[0])
            key = (shape, str(kwargs.get("dtype")), str(kwargs.get("device")))
            if key not in frozen:
                if kwargs.get("generator") is frozen_generator:
                    draw = {k: v for k, v in kwargs.items() if k != "generator"}
                    self._noise_draws.append((key, shape, draw))
                    kwargs["generator"] = noise_generator
                frozen[key] = real_randn(*shape, **kwargs)
            return frozen[key]

        torch.manual_seed = lambda *_: None
        torch.Generator = frozen_generator_factory
        torch.randn = frozen_randn
        torch.full = lambda size, fill_value, **kwargs: _graph_safe_full(
            real_full,
            size,
            fill_value,
            **kwargs,
        )
        try:
            # Capture APIs bind to the current device; warmup runs on a side
            # stream so its allocations are reusable during capture.
            with torch.cuda.device(self.device):
                side = torch.cuda.Stream(self.device)
                side.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(side), torch.inference_mode():
                    for _ in range(warmup):
                        self.model.predict(self.static, prompt_embs=self.prompt_embs, **self.kwargs)
                torch.cuda.current_stream(self.device).wait_stream(side)
                torch.cuda.synchronize(self.device)

                self.graph = torch.cuda.CUDAGraph()
                # The context manager unconditionally empty_cache()s, which can
                # assert while TensorRT owns a MemPool in PyTorch 2.7. Drive the
                # CUDAGraph lifecycle explicitly on the warmed side stream.
                side.wait_stream(torch.cuda.current_stream(self.device))
                # A replay executes the captured kernels and dispatches no Python, so
                # the TensorRT runtime's call counters never see it. The counter delta
                # around the capture is the per-request engine schedule the graph bakes.
                trt_runtime = getattr(self.model, "n2_trt_runtime", None)
                calls_before = dict(getattr(trt_runtime, "engine_calls", None) or {})
                with torch.cuda.stream(side), torch.inference_mode():
                    self.static_out = self._capture_predict()
                self.captured_engine_calls = (
                    {
                        mode: count - calls_before.get(mode, 0)
                        for mode, count in trt_runtime.engine_calls.items()
                    }
                    if trt_runtime is not None
                    else None
                )
                torch.cuda.current_stream(self.device).wait_stream(side)
        finally:
            torch.manual_seed = real_manual_seed
            torch.Generator = real_generator
            torch.randn = real_randn
            torch.full = real_full
            _restore_rng_states(self.device, rng_states)

        # The only evidence the graph is engaged: a caller that installs it logs
        # whatever backend it runs on and nothing about the graph, so a silent
        # fallback to eager reads the same. Shapes ride along as the replay
        # contract -- a later call that differs is a bug, not a slow path.
        logger.info(
            "Captured predict CUDA graph in %.1fs (%d warmup): %s",
            time.perf_counter() - t_start,
            warmup,
            ", ".join(
                f"{k}{tuple(v.shape)}"
                for k, v in self.static.items()
                if isinstance(v, torch.Tensor)
            ),
        )
