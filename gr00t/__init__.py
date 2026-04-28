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


def _patch_hf_local_first() -> None:
    """Patch from_pretrained to prefer the local HF snapshot cache over network calls.

    When a HF repo ID is passed we try snapshot_download(local_files_only=True)
    first; if the model is not cached we fall through to the normal download path.
    This avoids 429 rate-limit errors when many CI jobs run concurrently.

    Covers: PreTrainedModel, PretrainedConfig, ProcessorMixin, AutoConfig,
    AutoProcessor — every transformers from_pretrained entrypoint.

    Triggered by GROOT_HF_LOCAL_FIRST (set by conftest.py, survives uv run) or
    PYTEST_CURRENT_TEST (set automatically by pytest).
    """

    def _resolve(name_or_path: str) -> str:
        hf_home = os.environ.get("HF_HOME")
        hf_hub = os.environ.get("HUGGINGFACE_HUB_CACHE")
        hf_cache_info = f"HF_HOME={hf_home} HUGGINGFACE_HUB_CACHE={hf_hub}"
        if os.path.isdir(name_or_path):
            print(f"[groot/hf] local path: {name_or_path} | {hf_cache_info}", flush=True)
            return name_or_path
        try:
            from huggingface_hub import snapshot_download

            resolved = snapshot_download(name_or_path, local_files_only=True)
            print(
                f"[groot/hf] cache hit: {name_or_path} -> {resolved} | {hf_cache_info}", flush=True
            )
            return resolved
        except Exception:
            print(
                f"[groot/hf] cache miss (will download): {name_or_path} | {hf_cache_info}",
                flush=True,
            )
            return name_or_path

    def _wrap(cls: type) -> None:
        if "from_pretrained" not in cls.__dict__:
            return
        original = cls.from_pretrained
        if getattr(original, "_groot_hf_local_patched", False):
            return

        def _make_patched(orig):
            @classmethod  # type: ignore[misc]
            def patched(klass, pretrained_model_name_or_path, *args, **kwargs):
                resolved = _resolve(str(pretrained_model_name_or_path))

                return orig.__func__(klass, resolved, *args, **kwargs)

            patched._groot_hf_local_patched = True  # type: ignore[attr-defined]
            return patched

        cls.from_pretrained = _make_patched(original)

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
    """Suppress 429 / connection errors from the HuggingFace Hub in mistral regex patching.

    transformers calls model_info() inside a nested is_base_mistral() function
    unconditionally even when loading from a fully local checkpoint. Qwen3VL /
    Cosmos is never Mistral, so returning the tokenizer unchanged on any network
    failure is correct.

    NOTE: is_base_mistral is a *nested* function inside _patch_mistral_regex, so
    it is not accessible as a module-level attribute — we must wrap the classmethod.

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
