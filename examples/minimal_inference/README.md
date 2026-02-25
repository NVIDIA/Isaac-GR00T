# Minimal GR00T Inference Example

Simple, standalone example of running GR00T policy inference **without GPU**.

Perfect starting point for integrating GR00T into your project!

## Overview

This example shows the **core concepts** of using GR00T:
- Loading a pre-trained policy
- Preparing observations (images + state)
- Getting action predictions
- Interpreting output

**No GPU needed** - runs on CPU for testing and development.

## Quick Start

### 1. Install Dependencies

```bash
# If in GR00T environment
cd ../..
uv pip install -e .

# OR just the essentials
pip install torch transformers
```

### 2. Run the Example

```bash
python inference_minimal.py
```

### Expected Output

```
============================================================
GR00T Minimal Inference Example
============================================================

📥 Step 1: Loading pre-trained GR00T policy...
   ✅ Policy loaded successfully!

📸 Step 2: Preparing observations...
   Image shape: torch.Size([1, 1, 3, 256, 256])
   State shape: torch.Size([1, 23])
   Instruction: 'pick up the cube'
   ✅ Observations ready!

🤖 Step 3: Getting action prediction...
   Predicted action shape: torch.Size([1, 23])
   Action values (first 6 DOF): [0.123 -0.456 0.789 ...]
   ✅ Action prediction successful!

✅ Example completed successfully!
============================================================
```

## What Each Step Does

### Step 1: Load Policy

```python
policy = Gr00tPolicy.from_pretrained(
    model_name="nvidia/GR00T-N1.6-3B",
    embodiment_tag="GR1",  # UNITREE G1 humanoid
    device="cpu"           # Use CPU instead of GPU
)
```

**Available embodiment tags:**
- `GR1` - UNITREE G1 (default)
- `OXE_GOOGLE` - Google robot
- `OXE_WIDOWX` - WidowX manipulator
- `BEHAVIOR_R1_PRO` - R1-Pro humanoid
- `UNITREE_G1` - Alternative G1 tag

### Step 2: Prepare Observations

```python
observations = {
    "images": torch.randn(B, T, C, H, W),      # Images from cameras
    "proprioception": torch.randn(B, state_dim),  # Joint positions/velocities
    "instruction": "pick up the cube"          # Natural language command
}
```

**Observation formats:**
- `B` = batch size (1 for single inference)
- `T` = sequence length (1 for single frame, >1 for temporal)
- `C=3` = RGB channels
- `H=W=256` = standard resolution
- `state_dim` varies by robot (23 for GR1)

### Step 3: Get Action

```python
with torch.no_grad():
    action = policy.get_action(observations)

# Output: (batch_size, action_dim)
# For GR1: (1, 23) - 23 DOF command
```

## Extending the Example

### Use Real Camera Input

```python
import cv2

# Capture from camera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
frame = cv2.resize(frame, (256, 256))

observations = {
    "images": torch.from_numpy(frame).unsqueeze(0).unsqueeze(0),
    "proprioception": get_robot_state(),
    "instruction": "your command"
}

action = policy.get_action(observations)
```

### Custom Embodiment

```python
# Use your own robot
policy = Gr00tPolicy.from_pretrained(
    model_name="nvidia/GR00T-N1.6-3B",
    embodiment_tag="YOUR_EMBODIMENT",
)

observations = {
    "images": your_images,
    "proprioception": your_state,  # Must match YOUR_EMBODIMENT dim
    "instruction": your_instruction
}
```

### Real Robot Execution

```python
def deploy_on_robot(policy, robot):
    """Deploy GR00T on real robot."""
    while True:
        # Get observations from robot
        obs = robot.get_observations()
        
        # Get action from policy
        with torch.no_grad():
            action = policy.get_action(obs)
        
        # Execute action
        robot.execute_action(action)
```

## Common Issues

### Issue: "No module named 'gr00t'"
**Solution:** Install GR00T first
```bash
cd ../..
uv pip install -e .
```

### Issue: "CUDA out of memory"
**Solution:** Already using CPU, but if you have GPU:
```python
policy = Gr00tPolicy.from_pretrained(..., device="cpu")
```

### Issue: Model download fails
**Solution:** Set HuggingFace cache
```bash
export HF_HOME=/tmp/hf_cache
python inference_minimal.py
```

### Issue: Shape mismatch error
**Solution:** Check observation dimensions match embodiment
```python
# GR1 requires:
# - images: (B, T, 3, 256, 256)
# - proprioception: (B, 23)  ✅ not (B, 24) or (B, 20)
```

## Key Concepts

### Batch Processing
```python
# Single observation
shape = (1, 1, 3, 256, 256)  # Process one image

# Multiple observations (faster!)
shape = (8, 1, 3, 256, 256)  # Process 8 images at once
```

### Temporal Sequences
```python
# Single frame
images = torch.randn(1, 1, 3, 256, 256)  # T=1

# Multiple frames (gives temporal context)
images = torch.randn(1, 4, 3, 256, 256)  # T=4 (last 4 frames)
```

### Language Instructions
Support natural language commands:
```python
instructions = [
    "pick up the cube",
    "place on the table",
    "open the drawer",
    "press the button",
]

for instr in instructions:
    obs["instruction"] = instr
    action = policy.get_action(obs)
```

## Performance Tips

### 1. Batch Multiple Predictions
```python
# Slow: 10 individual inferences
for obs in observations:
    action = policy.get_action(obs)

# Fast: 1 batch inference
obs_batch = torch.stack(observations)
actions = policy.get_action(obs_batch)
```

### 2. Reuse Policy
```python
# Good: Load once, reuse
policy = Gr00tPolicy.from_pretrained(...)
for _ in range(1000):
    action = policy.get_action(obs)

# Bad: Load every iteration
for _ in range(1000):
    policy = Gr00tPolicy.from_pretrained(...)  # Slow!
```

### 3. Use No-Grad Context
```python
# Good: Disables unnecessary computation
with torch.no_grad():
    action = policy.get_action(obs)

# OK: But slower (computes gradients unnecessarily)
action = policy.get_action(obs)
```

## Next Steps

1. **Read the main README** - Understand GR00T architecture
   ```bash
   cat ../../README.md
   ```

2. **Try Policy API guide** - Learn advanced features
   ```bash
   cat ../../getting_started/policy.md
   ```

3. **Explore other examples** - See full integration
   - `../robocasa/` - Real environment evaluation
   - `../LIBERO/` - Kitchen task learning
   - `../SimplerEnv/` - Google Robot benchmark

4. **Deploy on your robot!** - Use this template with real hardware

## Files in This Example

```
minimal_inference/
├── inference_minimal.py    # Main inference script (THIS FILE)
├── README.md              # Documentation
└── requirements.txt       # Optional: precise versions
```

## Paper & Citation

If you use GR00T in your research, cite:

```bibtex
@inproceedings{gr00tn1_2025,
  title={GR00T N1: An Open Foundation Model for Generalist Humanoid Robots},
  author={NVIDIA and others},
  eprint={2503.14734},
  archivePrefix={arXiv},
  year={2025}
}
```

## Questions?

- **API Reference:** See `../../getting_started/policy.md`
- **Issues:** GitHub issues at https://github.com/NVIDIA/Isaac-GR00T
- **More examples:** Check `../` folder

## License

Apache 2.0 - Same as Isaac GR00T

---

**Happy GR00T-inferencing! 🤖**
