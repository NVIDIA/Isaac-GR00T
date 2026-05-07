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

import os


def _hf_env_repr() -> str:
    """Human-readable HF cache env (for log lines)."""
    hf_home = os.environ.get("HF_HOME")
    hf_hub = os.environ.get("HUGGINGFACE_HUB_CACHE")
    return f"HF_HOME={hf_home} HUGGINGFACE_HUB_CACHE={hf_hub}"


def _hf_local_first_call(orig_func, klass, pretrained_model_name_or_path, *args, **kwargs):
    """Invoke ``orig_func`` (an unwrapped from_pretrained) preferring the local cache.

    Strategy:

    1. Local filesystem path → call ``orig_func`` unchanged.
    2. Caller already passed ``local_files_only=True`` → honor it.
    3. Otherwise probe the cache by calling ``orig_func`` with
       ``local_files_only=True`` first.  If it succeeds, the cache had every
       file ``from_pretrained`` needed — no network traffic at all.  If it
       raises (cache empty, partial snapshot, missing shard), fall through
       to a normal call which lets HF Hub download the missing pieces.

    This mirrors exactly what ``transformers.from_pretrained`` itself uses
    to decide cache hit vs miss, so we never get a false miss from a
    stricter probe (the original implementation called
    ``snapshot_download(local_files_only=True)``, which requires a fully
    populated ``refs/main`` + ``snapshots/<commit>/`` tree that
    ``from_pretrained`` does not always create — observed to cause 100%
    false-miss rate in CI Job 308778931).
    """
    name_str = str(pretrained_model_name_or_path)
    if os.path.isdir(name_str):
        print(f"[groot/hf] local path: {name_str} | {_hf_env_repr()}", flush=True)
        return orig_func(klass, pretrained_model_name_or_path, *args, **kwargs)
    if kwargs.get("local_files_only", False):
        return orig_func(klass, pretrained_model_name_or_path, *args, **kwargs)
    try:
        result = orig_func(
            klass,
            pretrained_model_name_or_path,
            *args,
            **{**kwargs, "local_files_only": True},
        )
        print(f"[groot/hf] cache hit: {name_str} | {_hf_env_repr()}", flush=True)
        return result
    except Exception:
        print(
            f"[groot/hf] cache miss (will download): {name_str} | {_hf_env_repr()}",
            flush=True,
        )
        return orig_func(klass, pretrained_model_name_or_path, *args, **kwargs)


def _patch_hf_local_first() -> None:
    """Patch from_pretrained to prefer the local HF cache over network calls.

    When a HF repo ID is passed we first invoke ``from_pretrained`` with
    ``local_files_only=True``; on any error we fall through to a normal
    download.  This avoids the per-file etag roundtrip that
    ``transformers.from_pretrained`` does on every load and the 429
    rate-limit storm when many CI jobs run concurrently.

    Covers: PreTrainedModel, PretrainedConfig, ProcessorMixin, AutoConfig,
    AutoProcessor — every transformers from_pretrained entrypoint.

    Triggered by GROOT_HF_LOCAL_FIRST (set by conftest.py, survives uv run) or
    PYTEST_CURRENT_TEST (set automatically by pytest).
    """

    def _wrap(cls: type) -> None:
        if "from_pretrained" not in cls.__dict__:
            return
        original = cls.from_pretrained
        if getattr(original, "_groot_hf_local_patched", False):
            return

        orig_func = original.__func__

        @classmethod  # type: ignore[misc]
        def patched(klass, pretrained_model_name_or_path, *args, **kwargs):
            return _hf_local_first_call(
                orig_func, klass, pretrained_model_name_or_path, *args, **kwargs
            )

        patched._groot_hf_local_patched = True  # type: ignore[attr-defined]
        cls.from_pretrained = patched

    try:
        import transformers as _transformers

        for _attr in (
            "PreTrainedModel",
            "PretrainedConfig",
            "ProcessorMixin",
            "AutoConfig",
            "AutoProcessor",
        ):
            _cls = getattr(_transformers, _attr, None)
            if _cls is not None:
                _wrap(_cls)
    except Exception:
        pass


def _patch_mistral() -> None:
    """Suppress 429 / connection errors / hangs from the HuggingFace Hub in mistral regex patching.

    transformers calls model_info() inside a nested is_base_mistral() function
    unconditionally even when loading from a fully local checkpoint. Qwen3VL /
    Cosmos is never Mistral, so returning the tokenizer unchanged on any network
    failure is correct.

    NOTE: is_base_mistral is a *nested* function inside _patch_mistral_regex, so
    it is not accessible as a module-level attribute — we must wrap the classmethod.

    The wrapper short-circuits before _orig for two cases that should never need
    a network roundtrip:

    1. Local filesystem path — can't be a Hub repo ID at all.
    2. Repo ID that doesn't even mention "mistral" — the underlying check exists
       solely to recognize Mistral-derived tokenizers; on any other ID the
       network call is wasted at best.

    The short-circuit also closes a hang the surrounding ``except Exception``
    cannot catch: when a TCP connection to huggingface.co succeeds but the
    server stops sending bytes, ``socket.recv_into()`` blocks in the kernel
    with no Python-level exception, eventually breaching pytest-timeout (e.g.
    job 309584593: ``test_trt_full_pipeline[1]`` hung 600 s here while loading
    the tokenizer for ``nvidia/Cosmos-Reason2-2B``).

    Triggered by GROOT_PATCH_MISTRAL (set by conftest.py, survives uv run) or
    PYTEST_CURRENT_TEST (set automatically by pytest, belt-and-suspenders).
    """
    try:
        import transformers.tokenization_utils_base as _tub

        _cls = _tub.PreTrainedTokenizerBase
        _orig = _cls._patch_mistral_regex.__func__
        if getattr(_orig, "_groot_patched", False):
            return

        def _safe(cls, tokenizer, pretrained_model_name_or_path, **kwargs):
            name_str = str(pretrained_model_name_or_path)
            if os.path.isdir(name_str) or "mistral" not in name_str.lower():
                return tokenizer
            try:
                return _orig(cls, tokenizer, pretrained_model_name_or_path, **kwargs)
            except Exception:
                return tokenizer

        _safe._groot_patched = True  # type: ignore[attr-defined]
        _cls._patch_mistral_regex = classmethod(_safe)
    except Exception:
        pass


if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("GROOT_HF_LOCAL_FIRST"):
    _patch_hf_local_first()

if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("GROOT_PATCH_MISTRAL"):
    _patch_mistral()
