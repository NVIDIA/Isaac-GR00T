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

import logging

import torch
from transformers.feature_extraction_utils import BatchFeature


logger = logging.getLogger(__name__)


try:
    from transformers import Qwen3VLForConditionalGeneration

    _QWEN3VL_AVAILABLE = True
except ImportError:
    _QWEN3VL_AVAILABLE = False


def _real_inference_device(module: torch.nn.Module) -> torch.device:
    """Best-effort real (non-meta) device backing a module, defaulting to CPU.

    RoPE buffers must be rebuilt on the device holding the loaded weights, never
    on the ``meta`` device that may be the active default during nested load.
    """
    for tensor in module.parameters():
        if tensor.device.type != "meta":
            return tensor.device
    for tensor in module.buffers():
        if tensor.device.type != "meta":
            return tensor.device
    return torch.device("cpu")


def recompute_vision_rotary_inv_freq(
    rotary: torch.nn.Module, head_dim_half: int, device: torch.device
) -> torch.Tensor:
    """Re-derive Qwen3-VL's vision RoPE ``inv_freq`` via the module's own class.

    Reusing ``type(rotary)(...)`` keeps the analytic formula owned by Transformers;
    the real-device context makes it correct even under a ``meta`` default device.
    """
    with torch.device(device):
        fresh = type(rotary)(head_dim_half)
    return fresh.inv_freq.detach().to(device=device, dtype=torch.float32)


def recompute_text_rotary_inv_freq(
    rotary: torch.nn.Module, config, device: torch.device
) -> tuple[torch.Tensor, float]:
    """Re-derive Qwen3-VL's text RoPE ``inv_freq`` via the module's own class.

    Delegates to the module's constructor so the configured ``rope_init_fn``
    (default / scaled / dynamic) stays the single source of truth; returns the
    FP32 ``inv_freq`` and its ``attention_scaling``.
    """
    with torch.device(device):
        fresh = type(rotary)(config=config, device=device)
    inv_freq = fresh.inv_freq.detach().to(device=device, dtype=torch.float32)
    attention_scaling = float(getattr(fresh, "attention_scaling", 1.0))
    return inv_freq, attention_scaling


def _assign_inv_freq(
    rotary: torch.nn.Module, name: str, value: torch.Tensor, *, persistent: bool
) -> bool:
    """Write ``value`` onto ``rotary.<name>`` if it differs; return whether it changed.

    Returns ``True`` when the buffer/attribute was (re)written, ``False`` when the
    existing value already matched (idempotent no-op). This boolean lets tests
    assert that a corrupted buffer is actually repaired rather than silently
    skipped.
    """
    current = getattr(rotary, name, None)
    if (
        isinstance(current, torch.Tensor)
        and current.device.type != "meta"
        and current.device == value.device
        and current.shape == value.shape
        and current.dtype == value.dtype
        and torch.equal(current, value)
    ):
        return False
    if name in rotary._buffers:
        rotary.register_buffer(name, value, persistent=persistent)
    else:
        setattr(rotary, name, value)
    return True


class Qwen3Backbone(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason2-2B",
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = True,
        use_flash_attention: bool = False,
        projector_dim: int = -1,
        load_bf16: bool = False,
        tune_top_llm_layers: int = 0,
        trainable_params_fp32: bool = False,
        transformers_loading_kwargs: dict = {},
    ):
        """
        Qwen3Backbone is to generate n_queries to represent the future action hidden states.
        Args:
            model_name: nvidia/Cosmos-Reason2-2B
            tune_llm: whether to tune the LLM model (default: False)
            tune_visual: whether to tune the visual model (default: False)
        """
        if not _QWEN3VL_AVAILABLE:
            raise ImportError(
                "Qwen3VLForConditionalGeneration is not available. "
                "Please upgrade transformers to a version that supports Qwen3-VL: "
                "pip install transformers>=4.57.0"
            )

        super().__init__()

        # Add attention kwargs
        extra_kwargs = {}
        if use_flash_attention:
            try:
                import flash_attn  # noqa: F401

                extra_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                logger.warning(
                    "flash_attn is not installed. Falling back to sdpa attention. "
                    "Install flash-attn for better performance: pip install flash-attn"
                )
                extra_kwargs["attn_implementation"] = "sdpa"
        if load_bf16:
            extra_kwargs["torch_dtype"] = torch.bfloat16

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            **extra_kwargs,
            **transformers_loading_kwargs,
        ).eval()

        # needed since we don't use these layers. Also saves compute
        while len(self.model.language_model.layers) > select_layer:
            self.model.language_model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual, tune_top_llm_layers)
        if load_bf16 and trainable_params_fp32:
            # cast trainable parameters to fp32
            for n, p in self.named_parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)
                    logger.debug(f"Casting trainable parameter {n} to fp32")

        # Repair Qwen3-VL's non-persistent RoPE buffers once, after weights are
        # loaded and trainable dtypes are finalized. See _reset_rotary_inv_freq.
        self._reset_rotary_inv_freq()

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool, tune_top_llm_layers: int):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.model.language_model.requires_grad_(False)
        if not tune_visual:
            self.model.visual.requires_grad_(False)

        if tune_top_llm_layers > 0:
            for layer in self.model.language_model.layers[-tune_top_llm_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        logger.debug(f"Tune backbone llm: {self.tune_llm}")
        logger.debug(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, log a warning.
        for name, p in self.named_parameters():
            if p.requires_grad:
                logger.debug(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.model.language_model and not self.tune_llm:
                self.model.language_model.eval()
            if self.model.visual and not self.tune_visual:
                self.model.visual.eval()

    def _reset_rotary_inv_freq(self) -> None:
        """Re-derive Qwen3-VL's non-persistent RoPE ``inv_freq`` buffers once at load.

        These ``persistent=False`` buffers are not restored from a checkpoint and
        can be left uninitialized under the nested ``no_init_weights`` load, so we
        rebuild the analytic FP32 frequencies here (load time only, no per-forward
        cost). See the MR description for the FA2-vs-SDPA divergence this fixed.
        """
        config = getattr(self.model, "config", None)
        vision_changed = self._reset_vision_rotary_inv_freq(config)
        text_changed = self._reset_language_rotary_inv_freq()
        logger.debug(
            "Qwen3-VL RoPE inv_freq reset (vision_rewritten=%s, text_rewritten=%s).",
            vision_changed,
            text_changed,
        )

    def _reset_vision_rotary_inv_freq(self, config) -> bool:
        visual = getattr(self.model, "visual", None)
        rotary = getattr(visual, "rotary_pos_emb", None)
        if rotary is None or not hasattr(rotary, "inv_freq"):
            logger.warning(
                "Qwen3-VL visual rotary embedding not found; skipping vision RoPE "
                "inv_freq reset. The transformers Qwen3-VL layout may have changed."
            )
            return False

        vision_config = getattr(config, "vision_config", None)
        if vision_config is None or not all(
            hasattr(vision_config, attr) for attr in ("hidden_size", "num_heads")
        ):
            logger.warning(
                "Qwen3-VL vision_config missing hidden_size/num_heads; skipping "
                "vision RoPE inv_freq reset."
            )
            return False

        head_dim = vision_config.hidden_size // vision_config.num_heads
        device = _real_inference_device(visual)
        inv_freq = recompute_vision_rotary_inv_freq(rotary, head_dim // 2, device)
        return _assign_inv_freq(rotary, "inv_freq", inv_freq, persistent=False)

    def _reset_language_rotary_inv_freq(self) -> bool:
        language_model = getattr(self.model, "language_model", None)
        rotary = getattr(language_model, "rotary_emb", None)
        text_config = getattr(rotary, "config", None) or getattr(language_model, "config", None)
        if rotary is None or not hasattr(rotary, "inv_freq") or text_config is None:
            logger.warning(
                "Qwen3-VL language rotary embedding/config not found; skipping text "
                "RoPE inv_freq reset. The transformers Qwen3-VL layout may have changed."
            )
            return False

        device = _real_inference_device(language_model)
        inv_freq, _attention_scaling = recompute_text_rotary_inv_freq(rotary, text_config, device)
        changed = _assign_inv_freq(rotary, "inv_freq", inv_freq, persistent=False)
        # ``original_inv_freq`` is a plain attribute (not a buffer) that dynamic
        # RoPE updates restore from; keep it consistent with inv_freq.
        if hasattr(rotary, "original_inv_freq"):
            changed = (
                _assign_inv_freq(rotary, "original_inv_freq", inv_freq.clone(), persistent=False)
                or changed
            )
        return changed

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        # 0. Set frozen module to eval
        keys_to_use = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
        vl_input = {k: vl_input[k] for k in keys_to_use}
        outputs = self.model(**vl_input, output_hidden_states=True)
        outputs = outputs.hidden_states[-1]
        image_mask = vl_input["input_ids"] == self.model.config.image_token_id
        attention_mask = vl_input["attention_mask"] == 1
        return BatchFeature(
            data={
                "backbone_features": outputs,
                "backbone_attention_mask": attention_mask,
                "image_mask": image_mask,
            }
        )  # [B, T2, hidden_size]
