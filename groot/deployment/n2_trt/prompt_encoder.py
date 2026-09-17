"""Instance-local prompt caching with GPU computation and optional CPU weight storage."""

from __future__ import annotations

from collections import OrderedDict
import threading

import torch


class CachedPromptEncoder:
    """Keep a bounded CPU LRU; never run the text model inside a CUDA graph.

    Each instance owns one frozen encoder, so entries cannot cross checkpoint
    identities. Keys contain the effective token IDs, not the source string.
    Accelerate's hooks execute modules on CUDA without moving the whole tower
    there, including when an unseen prompt arrives after graph capture.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        device: torch.device,
        *,
        offload: bool,
        max_cache_bytes: int = 512 << 20,
    ) -> None:
        if max_cache_bytes < 0:
            raise ValueError("prompt cache byte limit must be nonnegative")
        self.encoder = encoder.eval()
        self.device = torch.device(device)
        self.offload = offload
        self.max_cache_bytes = max_cache_bytes
        self.cache_bytes = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.failure: str | None = None
        self._cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()
        self._lock = threading.Lock()
        if offload:
            if self.device.type != "cuda":
                raise ValueError("CPU weight offload requires CUDA execution")
            from accelerate import cpu_offload

            cpu_offload(self.encoder, execution_device=self.device, offload_buffers=True)
        else:
            self.encoder.to(self.device)

    @torch.inference_mode()
    def __call__(self, text_ids: torch.Tensor) -> torch.Tensor:
        with self._lock:
            if self.failure is not None:
                raise RuntimeError("text encoding previously failed; restart the policy server")
            try:
                ids = text_ids.detach().to("cpu")
                key = (str(ids.dtype), tuple(ids.shape), tuple(ids.reshape(-1).tolist()))
                cached = self._cache.pop(key, None)
                if cached is not None:
                    self.cache_hits += 1
                    self._cache[key] = cached
                    return cached.to(self.device, copy=True)

                self.cache_misses += 1
                # Do not call GrootN2.encode_text: its whole-tower .to(cuda)
                # would defeat Accelerate's per-module residency hooks.
                embeddings = self.encoder(text_ids.to(self.device)).to(
                    device=self.device, dtype=torch.bfloat16
                )
                if not bool(torch.isfinite(embeddings).all().item()):
                    raise RuntimeError("text encoder returned non-finite embeddings")
                size = embeddings.numel() * embeddings.element_size()
                if size <= self.max_cache_bytes:
                    while self.cache_bytes + size > self.max_cache_bytes and self._cache:
                        _, evicted = self._cache.popitem(last=False)
                        self.cache_bytes -= evicted.numel() * evicted.element_size()
                    self._cache[key] = embeddings.to("cpu", copy=True)
                    self.cache_bytes += size
                return embeddings
            except Exception as exc:
                # A failed forward can leave offload hooks or CUDA state unsafe.
                # Latch even cache misses as unhealthy instead of serving old actions.
                self.failure = type(exc).__name__
                raise
