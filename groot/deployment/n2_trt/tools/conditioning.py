"""Export-only conditioning tables for the observed fixed BF16 schedule."""

from contextlib import contextmanager

import torch
from torch import nn

from groot.core.training.models.n2.components.transformer_cosmos import (
    CosmosAdaLayerNorm,
    CosmosAdaLayerNormZero,
)


class _SigmaIndices(nn.Module):
    def __init__(self, sigmas):
        super().__init__()
        self.register_buffer("sigmas", sigmas)
        self.register_buffer(
            "row_numbers", torch.arange(1, sigmas.numel() + 1, device=sigmas.device)
        )

    def forward(self, _hidden_states, timestep):
        # Match the production input cast. No match (including NaN/Inf) selects
        # row zero, whose gathered modulation is NaN. Integer carriers survive
        # the core's reshape, coarse-token gather and fine-token restoration.
        matches = timestep.to(torch.bfloat16).reshape(-1, 1) == self.sigmas.reshape(1, -1)
        indices = (matches.to(torch.int64) * self.row_numbers).sum(dim=-1, keepdim=True)
        return indices, indices


class _ConditionedNorm(nn.Module):
    def __init__(self, original, embedded, temb):
        super().__init__()
        self.norm = original.norm
        self.parts = 2 if isinstance(original, CosmosAdaLayerNorm) else 3
        base = original.linear_2(original.linear_1(original.activation(embedded)))
        conditioned = base + (temb[..., : base.shape[-1]] if self.parts == 2 else temb)
        poison = torch.full_like(base[:1], float("nan"))
        self.register_buffer("base", torch.cat((poison, base)).detach())
        self.register_buffer("conditioned", torch.cat((poison, conditioned)).detach())

    def forward(self, hidden_states, embedded_timestep, temb=None, segments=None):
        if segments is not None:
            raise ValueError(
                "TensorRT conditioning export does not support grouped timestep segments"
            )
        table = self.base if temb is None else self.conditioned
        et = table[embedded_timestep[..., 0]]
        values = et.chunk(self.parts, dim=-1)
        if et.ndim == 2:
            values = tuple(value.unsqueeze(1) for value in values)
        shift, scale = values[:2]
        output = self.norm(hidden_states) * (1 + scale) + shift
        return output if self.parts == 2 else (output, values[2])


@contextmanager
def export_conditioning(transformer, supported_sigmas):
    """Temporarily replace timestep-only math; restore all modules on exit."""
    originals = []
    try:
        with torch.no_grad():
            parameter = next(transformer.time_embed.parameters())
            sigmas = torch.unique(
                supported_sigmas.detach().to(device=parameter.device, dtype=torch.bfloat16)
            )
            if not sigmas.numel() or not torch.isfinite(sigmas).all():
                raise ValueError("conditioning requires nonempty finite BF16 sigmas")
            if parameter.dtype != torch.bfloat16:
                raise ValueError("conditioning export requires BF16 weights")
            temb, embedded = transformer.time_embed(parameter, sigmas)
            replacements = []
            for parent in list(transformer.modules()):
                for name, child in list(parent.named_children()):
                    if isinstance(child, CosmosAdaLayerNorm | CosmosAdaLayerNormZero):
                        replacements.append(
                            (parent, name, child, _ConditionedNorm(child, embedded, temb))
                        )
            replacements.append(
                (transformer, "time_embed", transformer.time_embed, _SigmaIndices(sigmas))
            )
            for parent, name, original, replacement in replacements:
                originals.append((parent, name, original))
                setattr(parent, name, replacement)
        yield {
            "kind": "fixed_schedule_bf16_lookup",
            "supported_sigmas": sigmas.float().cpu().tolist(),
            "unsupported_sigma": "nan_modulation_row",
            "normalization_modules": len(replacements) - 1,
        }
    finally:
        for parent, name, original in reversed(originals):
            setattr(parent, name, original)
