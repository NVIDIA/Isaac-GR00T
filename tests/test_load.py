import pytest
import torch

from gr00t.model.gr00t_n1 import GR00T_N1_5

# Skip test if no GPU is available
if not torch.cuda.is_available():
    pytest.skip("Skipping test: No GPU detected", allow_module_level=True)

model_path = "nvidia/GR00T-N1.5-3B"
# NOTE: you can provide a local path to the checkpoint to check it's validity
# model_path = "/home/youliang/checkpoints/gn1-clothes-40k/"

# Load the model
model = GR00T_N1_5.from_pretrained(model_path, torch_dtype=torch.bfloat16)


# Check if future_tokens key exists in safetensors files
def check_future_tokens(model_path):
    from pathlib import Path

    from safetensors import safe_open

    print(f"Checking safetensors files in: {model_path}")
    safetensors_files = list(Path(model_path).glob("*.safetensors"))

    if not safetensors_files:
        print("\033[91mNo safetensors files found!\033[0m")
        raise ValueError("No safetensors files found!")
    else:
        print(f"\033[92mFound {len(safetensors_files)} safetensors files:\033[0m")
        found_future_tokens = False
        for file_path in safetensors_files:
            print(f"  - {file_path.name}")
            try:
                with safe_open(str(file_path), framework="pt", device="cpu") as f:
                    keys = f.keys()
                    # print(f"    Keys: {keys}")
                    future_tokens_found = "action_head.future_tokens.weight" in keys
                    print(f"    Contains future_tokens: {future_tokens_found}")
                    if future_tokens_found:
                        tensor = f.get_tensor("action_head.future_tokens.weight")
                        print(f"    Shape: {tensor.shape}")
                        print(f"    Dtype: {tensor.dtype}")
                    found_future_tokens = found_future_tokens or future_tokens_found
            except Exception as e:
                print(f"    Error reading file: {e}")

    if not found_future_tokens:
        raise ValueError("No future_tokens found!")


# This is to check if the checkpoint contains future_tokens.weight
# Related to this PR to fix GR00T-N1.5-3B: https://github.com/NVIDIA/Isaac-GR00T/pull/246
check_future_tokens(model.local_model_path)
