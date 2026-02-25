from __future__ import annotations
from typing import Optional, Tuple
import torch

def get_gpu_compute_capability() -> Optional[Tuple[int, int]]:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability()

def check_flash_attn_compatibility() -> Tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "No CUDA GPU detected."
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    gpu_name = torch.cuda.get_device_name()
    try:
        import flash_attn
        raw_version = flash_attn.__version__.split("+")[0]
        version = tuple(int(x) for x in raw_version.split(".")[:3] if x.isdigit())
        if sm >= 120 and version < (2, 8, 2):
            return False, (
                f"GPU '{gpu_name}' (SM{sm}) requires flash-attn >= 2.8.2, "
                f"but found {flash_attn.__version__}. "
                f"Upgrade: pip install 'flash-attn>=2.8.2' --no-build-isolation\n"
                f"See https://github.com/NVIDIA/Isaac-GR00T/issues/309"
            )
        return True, f"flash-attn {flash_attn.__version__} is compatible with '{gpu_name}' (SM{sm})."
    except ImportError:
        return False, "flash-attn not installed. Run: pip install 'flash-attn>=2.8.2' --no-build-isolation"

def require_flash_attn_compatible() -> None:
    is_compatible, message = check_flash_attn_compatibility()
    if not is_compatible:
        raise RuntimeError(f"Flash attention compatibility check failed:\n{message}")
