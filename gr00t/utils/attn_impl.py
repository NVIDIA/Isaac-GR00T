import torch
import importlib.util

def _flash2_available() -> bool:
    """
    Return True if:
      - CUDA is available,
      - flash_attn package is installed & importable,
      - GPU compute capability >= 8.0 (Ampere/Ada/Hopper).
    """
    if not torch.cuda.is_available():
        return False  # No GPU :contentReference[oaicite:5]{index=5}

    # 1) Check module presence
    spec = importlib.util.find_spec("flash_attn")
    if spec is None:
        return False  # Package missing :contentReference[oaicite:6]{index=6}

    try:
        __import__("flash_attn")
    except ImportError:
        return False  # Broken install :contentReference[oaicite:7]{index=7}

    # 2) Check compute capability
    major, minor = torch.cuda.get_device_capability()
    cc = major + minor / 10.0
    # FlashAttn-2 supports CC >= 8.0 (Ampere/Ada) or >= 9.0 (Hopper) :contentReference[oaicite:8]{index=8}
    return cc >= 8.0

def select_attn_impl(user_choice: str = "auto") -> str:
    """
    Determine attention implementation priority:
      - flash_attention_2 if _flash2_available()
      - sdpa if GPU exists but no flash2
      - eager otherwise
    """
    flash2 = _flash2_available()
    is_gpu = torch.cuda.is_available()

    if user_choice == "auto":
        if flash2:
            final = "flash_attention_2"
        elif is_gpu:
            final = "sdpa"
        else:
            final = "eager"
    elif user_choice == "flash_attention_2":
        if flash2:
            final = "flash_attention_2"
        else:
            print("[Warning] flash_attention_2 requested but not compatible; falling back to 'eager'.")
            final = "eager"
    elif user_choice == "sdpa":
        if is_gpu:
            final = "sdpa"
        else:
            print("[Warning] sdpa requested but no GPU detected; falling back to 'eager'.")
            final = "eager"
    elif user_choice == "eager":
        final = "eager"
    else:
        print(f"[Warning] Unknown attn_implementation '{user_choice}'; defaulting to 'eager'.")
        final = "eager"

    print(f"Using attention implementation: {final}")
    return final
