# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Live (in-process) policy adapter for the open-loop eval callback.

Composition-only wrapper around an in-memory ``Gr00tPolicy`` / ``Gr00tN2Policy``
that we instantiate without calling ``from_pretrained`` (the model is already
loaded in the training process). Auto-dispatches on the live model's surface.

The adapter delegates every ``BasePolicy`` method to the wrapped concrete
policy, which means any future change to ``Gr00tPolicy._get_action`` (or any
subclass's override) flows through automatically. The only things this module
owns are:

* construction-without-disk-load (``__new__`` + explicit attribute assign),
* model classification (``_classify_model``: predict+has_action_head -> N2),
* ``num_inference_timesteps`` plumbing per flavour,
* ``reset()`` semantics that don't drop the prompt cache or call save_video,
* ``close()`` lifecycle (restore mutated model state).

NOTE: ``gr00t_policy.py`` and ``n2_policy.py`` are *untouched* by this design.
The N2 inference defaults are inlined in ``_make_n2_concrete`` below, with a
KEEP-IN-SYNC comment.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Protocol, cast, runtime_checkable

import torch

from groot.core.data import schema, types
from groot.core.policy.base import BasePolicy
from groot.core.policy.gr00t_policy import Gr00tPolicy
from groot.core.policy.n2_policy import (
    Gr00tN2Policy,
    _env_bool,
    _env_float,
    _env_int,
    _resolve_codec_roundtrip,
    _resolve_constructor_env_int,
)

# ^ _env_bool / _env_int / _env_float / _resolve_codec_roundtrip are module-private
#   but shared across both the disk-load path (n2_policy.Gr00tN2Policy.__init__) and
#   the live path (_make_n2_concrete below). Importing instead of re-implementing
#   eliminates semantic-drift risk on env-var parsing — a future fix to
#   the parser semantics takes effect on both paths simultaneously.
#   This is the one approved exception to the "live policy doesn't depend
#   on n2_policy internals" rule: these helpers parse strings, not policy state.
# Module-class registration side effects — these are what
# Gr00tPolicy.__init__ / Gr00tN2Policy.__init__ would have imported.
# We do them at module scope so the __new__ bypass is safe even if a future
# entry point reaches the live path before any other model-loading code.
import groot.core.training.models
import groot.core.training.models.n2  # noqa: F401  (registers N2 with HF AutoModel)

_Restorer = Callable[[], None]
_MAX_UNWRAP_DEPTH = 4


@runtime_checkable
class _LivePolicyProtocol(Protocol):
    """Narrow surface the adapter delegates to the wrapped policy.

    Note: ``_get_action`` is intentionally NOT on the protocol — the adapter
    reaches into the wrapped policy's private method directly via ``cast``
    (see ``LivePolicyAdapter._get_action``).
    """

    def get_action(self, observation: dict, options: dict | None = None) -> tuple[dict, dict]: ...
    def get_modality_config(self) -> dict: ...
    def check_observation(self, observation: dict) -> None: ...
    def check_action(self, action: dict) -> None: ...


# ----------------------------------------------------------------------
# Adapter
# ----------------------------------------------------------------------


class LivePolicyAdapter(BasePolicy):
    """Composition wrapper around an in-memory ``Gr00tPolicy`` subclass.

    The wrapped policy is built by ``build_live_policy(...)`` against a *live*
    training model + processor. The adapter forwards every ``BasePolicy``
    method to the wrapped policy and owns lifecycle: ``close()`` runs the
    restorers (e.g. put back the action head's pre-eval timestep count) and
    restores the processor's training mode.

    NOT a subclass of ``Gr00tPolicy``. Future ``Gr00tPolicy`` changes flow
    through delegation, not inheritance.
    """

    def __init__(
        self,
        *,
        policy: _LivePolicyProtocol,
        restorers: list[_Restorer],
        prev_processor_training: bool,
        strict: bool = True,
        ttt_model: torch.nn.Module | None = None,
    ) -> None:
        super().__init__(strict=strict)
        self._policy = policy
        self._restorers = restorers
        self._prev_processor_training = prev_processor_training
        self._ttt_model = ttt_model
        self._ttt_states = None

    @contextmanager
    def _use_ttt_cache(self) -> Iterator[None]:
        """Lend this adapter's cache to N2-TTT only for the prediction.

        The policy allocates inference tensors. Leaving those on the live model
        breaks the next training backward; restoring only on close also mixes
        caches between embodiment adapters. Preserve the original cache by
        identity and discard partial eval state if prediction raises.
        """
        model = self._ttt_model
        if model is None:
            yield
            return
        original = model._ttt_states  # noqa: SLF001 — manage the live model's inference cache
        model._ttt_states = self._ttt_states  # noqa: SLF001
        self._ttt_states = None
        try:
            yield
            self._ttt_states = model._ttt_states  # noqa: SLF001
        finally:
            model._ttt_states = original  # noqa: SLF001

    # ---- BasePolicy delegation ---------------------------------------
    def check_observation(self, observation: dict[str, Any]) -> None:
        self._policy.check_observation(observation)

    def check_action(self, action: dict[str, Any]) -> None:
        self._policy.check_action(action)

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # Reach into the wrapped policy's private _get_action: the adapter's
        # public get_action already ran the strict-mode validation, so calling
        # the wrapped policy's get_action would re-validate. Cast for type
        # clarity instead of a # type: ignore.
        wrapped = cast(Gr00tPolicy, self._policy)
        # Wrap in bf16 autocast: the live training model's params may be fp32
        # (e.g. when started from a fp32-loaded `--model.no-load_bf16` run),
        # but the inference path goes through FlashAttention which requires
        # fp16/bf16 Q/K/V. HF Trainer's training_step autocast does NOT cover
        # `on_step_end` callbacks, so we wrap explicitly here.
        with self._use_ttt_cache(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            return wrapped._get_action(observation, options)  # noqa: SLF001 — see above

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:  # noqa: ARG002
        """Per-trajectory reset that does NOT delegate blindly.

        ``Gr00tN2Policy.reset`` invalidates the prompt cache and calls
        ``save_video()``. Both are wrong for in-process eval: the prompt
        is shared across trajectories (re-encoding wastes work), and
        ``save_video_dir`` is forced to ``None`` so the auto-save path
        would crash on ``os.path.join(None, ...)``. We re-implement reset
        to do only the safe parts: clear the latent buffer and this adapter's
        TTT cache between trajectories, return ``{}``.
        """
        self._ttt_states = None
        latent_buf = getattr(self._policy, "video_latents_buffer", None)
        if isinstance(latent_buf, list):
            latent_buf.clear()
        return {}

    def get_modality_config(self) -> dict[str, types.ModalityConfig]:
        return self._policy.get_modality_config()

    # ---- Lifecycle ---------------------------------------------------
    def close(self) -> None:
        """Restore mutated state. Let-it-crash: failure aborts training."""
        self._ttt_states = None
        for r in self._restorers:
            r()
        self._restorers = []
        if self._prev_processor_training:
            cast(Any, self._policy).processor.train()


# ----------------------------------------------------------------------
# Dispatch + factory
# ----------------------------------------------------------------------


def _unwrap_to_inner_module(model: torch.nn.Module) -> torch.nn.Module:
    """Strip DDP/DeepSpeed wrappers (``.module``).

    Stops at the first object without a ``.module`` attribute. Does NOT peel RL
    wrappers — those expose a different inner via ``wrapped_model`` and N1's
    ``_resolve_n1_action_head`` knows how to walk both.

    Asserts the unwrap depth bound. 4 is a generous upper bound for
    DDP+DeepSpeed+ZeRO+anything. If the bound is hit we have a pathological
    wrapper stack and the right answer is to crash with diagnostics.
    """
    inner = model
    for _ in range(_MAX_UNWRAP_DEPTH):
        sub = getattr(inner, "module", None)
        if isinstance(sub, torch.nn.Module):
            inner = sub
            continue
        return inner
    raise RuntimeError(
        f"Could not unwrap to inner module after {_MAX_UNWRAP_DEPTH} hops; "
        f"current type is {type(inner).__name__}. "
        f"This usually means a wrapper class has a `.module` attribute that "
        f"itself wraps another `.module`-bearing wrapper beyond the expected "
        f"DDP/DeepSpeed depth. Increase _MAX_UNWRAP_DEPTH if intentional."
    )


def _classify_model(inner: torch.nn.Module) -> str:
    """Return 'n1' or 'n2' from the unwrapped live model's surface.

    N2 is identified by both ``predict`` (the call site) AND ``has_action_head``
    (the attribute set in ``groot/core/training/models/n2/model.py:95``,
    ``self.has_action_head = config.action_cfg is not None``). Requiring both
    avoids false positives on any N1 model that grows a ``predict()`` shim.
    """
    if hasattr(inner, "predict") and getattr(inner, "has_action_head", False):
        return "n2"
    if hasattr(inner, "action_head") or hasattr(inner, "wrapped_model"):
        return "n1"
    raise TypeError(
        f"Cannot dispatch LivePolicyAdapter: model "
        f"{type(inner).__name__} has neither N2 ('predict' + "
        f"'has_action_head') nor N1 ('action_head' / 'wrapped_model') surface."
    )


def _resolve_n1_action_head(inner: torch.nn.Module) -> torch.nn.Module | None:
    """Find the action_head for the N1 family (incl. RL-wrapped GrootN1d6RL)."""
    if hasattr(inner, "action_head"):
        return inner.action_head
    if hasattr(inner, "wrapped_model"):
        return getattr(inner.wrapped_model, "action_head", None)
    return None


def _override_n1_inference_timesteps(action_head: torch.nn.Module, value: int) -> _Restorer:
    """Mutate the action head's inference timestep count.

    Returns a closure that restores the previous value.
    """
    if hasattr(action_head, "num_inference_timesteps"):
        attr = "num_inference_timesteps"
    elif hasattr(action_head, "num_inference_steps"):
        attr = "num_inference_steps"
    else:
        raise RuntimeError(
            f"action_head {type(action_head).__name__} has neither "
            f"num_inference_timesteps nor num_inference_steps attribute"
        )
    prev = int(getattr(action_head, attr))
    setattr(action_head, attr, int(value))

    def _restore() -> None:
        setattr(action_head, attr, prev)

    return _restore


# ----------------------------------------------------------------------
# Per-flavour constructor closures
# ----------------------------------------------------------------------


def _make_n1_concrete(
    *,
    model: torch.nn.Module,
    inner_module: torch.nn.Module,
    processor: types.BaseProcessor,
    embodiment_tag: schema.EmbodimentTag,
    num_inference_timesteps: int | None,
    strict: bool,
) -> tuple[Gr00tPolicy, list[_Restorer]]:
    """Construct a Gr00tPolicy *without* calling its from_pretrained ctor.

    Bypasses ``Gr00tPolicy.__init__`` (which loads from disk and moves to
    device — both already done in training) via ``__new__`` + explicit
    attribute assignment of every field ``Gr00tPolicy.__init__`` sets that
    the inference path actually reads.

    KEEP IN SYNC WITH ``Gr00tPolicy.__init__``. Whenever the upstream ctor
    adds a new attribute that ``_get_action`` reads, add the corresponding
    line here. The e2e test in
    ``tests/groot/core/training/open_loop/test_live_policy_adapter.py``
    will trip ``AttributeError`` on drift.
    """
    policy = Gr00tPolicy.__new__(Gr00tPolicy)
    BasePolicy.__init__(policy, strict=strict)
    policy.model = model
    policy.processor = processor
    policy.embodiment_tag = embodiment_tag
    policy.modality_configs = policy._load_modality_configs()  # noqa: SLF001
    policy.collate_fn = processor.collator
    language_keys = policy.modality_configs["language"].modality_keys
    language_delta = policy.modality_configs["language"].delta_indices
    assert len(language_delta) == 1, "Only one language delta index is supported"
    policy.language_key = language_keys[0]

    restorers: list[_Restorer] = []
    if num_inference_timesteps is not None:
        action_head = _resolve_n1_action_head(inner_module)
        if action_head is None:
            raise RuntimeError(
                f"num_inference_timesteps={num_inference_timesteps} requested "
                f"but model {type(inner_module).__name__} has no action_head."
            )
        restorers.append(_override_n1_inference_timesteps(action_head, num_inference_timesteps))
    return policy, restorers


def _make_n2_concrete(
    *,
    model: torch.nn.Module,
    inner_module: torch.nn.Module,
    processor: types.BaseProcessor,
    embodiment_tag: schema.EmbodimentTag,
    num_inference_timesteps: int | None,
    strict: bool,
) -> tuple[Gr00tN2Policy, list[_Restorer]]:
    """Same idea for N2.

    The override site is policy.num_inference_steps, not
    action_head.num_inference_timesteps. No model-side mutation, so no restorer
    needed.

    KEEP IN SYNC WITH ``Gr00tN2Policy.__init__``. The defaults below
    (kv_cache_cond, joint_denoise, num_video_steps from env; the others as
    constants) mirror what ``Gr00tN2Policy.__init__`` sets given default
    constructor kwargs. If ``Gr00tN2Policy.__init__`` adds a new env-driven
    knob or a new constructor default, mirror it here. The unit test
    ``test_make_n2_concrete_sets_documented_attributes`` is the regression
    guard.
    """
    policy = Gr00tN2Policy.__new__(Gr00tN2Policy)
    BasePolicy.__init__(policy, strict=strict)
    policy.model = model
    policy.processor = processor
    policy.embodiment_tag = embodiment_tag
    policy.modality_configs = policy._load_modality_configs()  # noqa: SLF001
    policy.collate_fn = processor.collator
    if hasattr(inner_module, "_ttt_states"):
        # TTT is not exported by the OSS N2 package. Import it only for a TTT
        # model, using the same cache marker as build_live_policy below.
        from groot.core.training.models.n2_ttt.processor import GrootN2TTTDataCollator

        if isinstance(policy.collate_fn, GrootN2TTTDataCollator):
            # TTT predict consumes one timestep per observation. Training may pad
            # trajectories to a fixed length; never reuse or mutate that collator.
            policy.collate_fn = GrootN2TTTDataCollator(
                tokenizer=policy.collate_fn.tokenizer,
                seq_len_handle_strategy="fixed",
                fixed_seq_len=1,
            )
    policy.language_key = policy.modality_configs["language"].modality_keys[0]

    # KEEP IN SYNC WITH Gr00tN2Policy.__init__:
    #   constructor > environment > default action-step precedence
    policy.num_inference_steps = _resolve_constructor_env_int(
        "GROOT_N2_NUM_ACTION_STEPS",
        constructor_value=num_inference_timesteps,
        default=4,
    )
    policy.video_guidance_scale = 1.0
    policy.action_guidance_scale = 1.0
    policy.seed = 42

    # KEEP IN SYNC WITH Gr00tN2Policy.__init__:
    #   env-driven knobs — re-read here so a training run that flips
    #   GROOT_N2_* on the command line has the same effect on the live
    #   eval path as on a disk-loaded policy. The _env_bool / _env_int
    #   helpers are *imported from n2_policy* (see top-of-file comment),
    #   not re-implemented locally — that way a future fix to the parser
    #   takes effect on both paths simultaneously.
    policy.kv_cache_cond = _env_bool("GROOT_N2_KV_CACHE_COND", default=True)
    policy.joint_denoise = _env_bool("GROOT_N2_JOINT_DENOISE", default=True)
    policy.num_video_steps = _env_int("GROOT_N2_NUM_VIDEO_STEPS", default=2)
    policy.codec_roundtrip = _resolve_codec_roundtrip()
    policy.shift = _env_float("GROOT_N2_SHIFT", default=5.0)

    # Per-trajectory state — adapter clears the buffer in `reset()`.
    policy.video_latents_buffer = []
    policy.save_video_dir = None  # don't dump training-time video
    policy.enable_n2_pred_comparison = False
    # `_step_count` is incremented unconditionally by every `_get_action` call;
    # only consulted when `save_video_dir is not None`, but read either way so
    # it must exist. KEEP IN SYNC WITH ``Gr00tN2Policy.__init__``.
    policy._step_count = 0  # noqa: SLF001
    policy._cached_prompt_embs = None  # noqa: SLF001
    policy._cached_language_dict = None  # noqa: SLF001
    return policy, []


_CONCRETE_POLICY_FACTORIES: dict[
    str,
    Callable[..., tuple[_LivePolicyProtocol, list[_Restorer]]],
] = {
    "n1": _make_n1_concrete,
    "n2": _make_n2_concrete,
}


# ----------------------------------------------------------------------
# Public factory — the only thing the callback imports
# ----------------------------------------------------------------------


def build_live_policy(
    *,
    model: torch.nn.Module,
    processor: types.BaseProcessor,
    embodiment_tag: schema.EmbodimentTag,
    num_inference_timesteps: int | None = None,
    strict: bool = True,
) -> LivePolicyAdapter:
    """Build a LivePolicyAdapter against a live training model + processor.

    Dispatches on ``_classify_model(unwrapped(model))``: N2 -> Gr00tN2Policy,
    N1 -> Gr00tPolicy. RL-wrapped N1 (GrootN1d6RL) classifies as N1 and the
    N1 path knows how to reach into ``wrapped_model.action_head``.

    This is the *only* function the in-process eval callback should import.
    """
    inner = _unwrap_to_inner_module(model)
    factory = _CONCRETE_POLICY_FACTORIES[_classify_model(inner)]
    policy, restorers = factory(
        model=model,  # bind wrapped model so DDP/DS hooks fire
        inner_module=inner,  # used only for action_head introspection
        processor=processor,
        embodiment_tag=embodiment_tag,
        num_inference_timesteps=num_inference_timesteps,
        strict=strict,
    )
    prev_processor_training = bool(getattr(processor, "training", False))
    processor.eval()
    return LivePolicyAdapter(
        policy=policy,
        restorers=restorers,
        prev_processor_training=prev_processor_training,
        strict=strict,
        ttt_model=inner if hasattr(inner, "_ttt_states") else None,
    )
