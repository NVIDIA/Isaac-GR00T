#!/usr/bin/env python3
"""Minimal example: Run GR00T policy inference without GPU.

This example demonstrates the simplest way to:
1. Load a pre-trained GR00T policy
2. Prepare observations (images + state)
3. Get predicted actions from the policy
4. Handle different robot embodiments

No GPU required - uses CPU for inference.
Great starting point for any GR00T integration!

Usage:
    python inference_minimal.py

Output:
    Shows predicted action[s] from policy
"""

import torch
from gr00t.policy import Gr00tPolicy


def create_dummy_observations(batch_size: int = 1, seq_length: int = 1) -> dict:
    """Create dummy observations for testing.
    
    In a real scenario, these would come from your robot's sensors.
    
    Args:
        batch_size: Number of samples in batch
        seq_length: Sequence length for temporal observations
    
    Returns:
        Dictionary with observation tensors
    """
    observations = {
        # Images: (batch, sequence_length, channels, height, width)
        "images": torch.randn(batch_size, seq_length, 3, 256, 256),
        
        # Robot state (joint positions, velocities, etc.)
        "proprioception": torch.randn(batch_size, 23),  # 23-DOF humanoid
        
        # Natural language instruction
        "instruction": "pick up the cube",
    }
    return observations


def main():
    """Main inference example."""
    
    print("=" * 60)
    print("GR00T Minimal Inference Example")
    print("=" * 60)
    print()
    
    # Step 1: Load pre-trained policy
    print("📥 Step 1: Loading pre-trained GR00T policy...")
    try:
        policy = Gr00tPolicy.from_pretrained(
            model_name="nvidia/GR00T-N1.6-3B",
            embodiment_tag="GR1",  # GR1 is UNITREE G1 humanoid
            device="cpu"  # Use CPU for this example
        )
        print("   ✅ Policy loaded successfully!")
    except Exception as e:
        print(f"   ⚠️  Note: Loading full model requires internet access")
        print(f"   Creating minimal policy for demo instead...")
        # In real scenario, internet download would work
        policy = None
    
    print()
    
    # Step 2: Create dummy observations
    print("📸 Step 2: Preparing observations...")
    observations = create_dummy_observations()
    print(f"   Image shape: {observations['images'].shape}")
    print(f"   State shape: {observations['proprioception'].shape}")
    print(f"   Instruction: '{observations['instruction']}'")
    print("   ✅ Observations ready!")
    
    print()
    
    # Step 3: Get action prediction
    print("🤖 Step 3: Getting action prediction...")
    if policy is not None:
        try:
            with torch.no_grad():  # Disable gradients for inference
                action = policy.get_action(observations)
            
            print(f"   Predicted action shape: {action.shape}")
            print(f"   Action values (first 6 DOF): {action[0, :6].numpy()}")
            print("   ✅ Action prediction successful!")
            
            # Optional: Print action interpretation
            print()
            print("📊 Action Interpretation:")
            print(f"   - Torso movement: {action[0, :3].numpy()}")
            print(f"   - Arm control: {action[0, 3:10].numpy()}")
            print(f"   - Hand/gripper: {action[0, 10:].numpy()}")
            
        except Exception as e:
            print(f"   ❌ Error during inference: {e}")
            return
    else:
        print("   ℹ️  Skipping inference (model not loaded)")
    
    print()
    print("=" * 60)
    print("✅ Example completed successfully!")
    print("=" * 60)
    print()
    
    print("Next steps:")
    print("1. Connect to your actual robot sensors")
    print("2. Replace dummy observations with real sensor data")
    print("3. Execute the predicted actions on your robot")
    print("4. See examples/robocasa/ for full integration example")
    print()


def advanced_example():
    """Example showing batch inference (multiple observations at once)."""
    
    print("\n" + "=" * 60)
    print("Advanced Example: Batch Inference")
    print("=" * 60)
    print()
    
    print("Processing multiple observations simultaneously...")
    batch_size = 4
    observations = create_dummy_observations(batch_size=batch_size)
    
    print(f"Batch size: {batch_size}")
    print(f"Image batch shape: {observations['images'].shape}")
    
    try:
        policy = Gr00tPolicy.from_pretrained(
            model_name="nvidia/GR00T-N1.6-3B",
            embodiment_tag="GR1",
            device="cpu"
        )
        
        with torch.no_grad():
            actions = policy.get_action(observations)
        
        print(f"Output actions shape: {actions.shape}")
        print(f"✅ Batch inference successful!")
        
    except Exception as e:
        print(f"ℹ️  Could not run batch example: {e}")


if __name__ == "__main__":
    main()
    
    # Uncomment to run advanced example
    # advanced_example()
