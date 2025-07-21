#!/usr/bin/env python3
"""
Simple GR00T Inference with Vocabulary Projection

This script runs basic inference and shows what "words" the VLM is thinking about.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

import gr00t
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP


def setup_model_and_data():
    """Setup the model and dataset"""
    # Paths
    MODEL_PATH = "nvidia/GR00T-N1.5-3B"
    REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
    DATASET_PATH = os.path.join(REPO_PATH, "demo_data/robot_sim.PickNPlace")
    EMBODIMENT_TAG = "gr1"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load policy
    data_config = DATA_CONFIG_MAP["fourier_gr1_arms_only"]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy = Gr00tPolicy(
        model_path=MODEL_PATH,
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=modality_config,
        modality_transform=modality_transform,
        device=device,
    )

    # Load dataset
    dataset = LeRobotSingleDataset(
        dataset_path=DATASET_PATH,
        modality_configs=modality_config,
        video_backend="decord",
        video_backend_kwargs=None,
        transforms=None,
        embodiment_tag=EMBODIMENT_TAG,
    )

    return policy, dataset


def get_vlm_hidden_states(policy, observations):
    """Extract hidden states from the VLM backbone"""
    # Get the backbone model
    backbone = policy.model.backbone

    # Prepare input for backbone
    with torch.no_grad():
        # This mimics what happens in the forward pass
        backbone_input = backbone.prepare_inputs(observations)

        # Get VLM hidden states (before action head)
        vlm_output = backbone(**backbone_input)

        return vlm_output


def simple_vocab_projection(hidden_states, policy, top_k=5):
    """
    Simple vocabulary projection - project hidden states back to vocabulary space
    """
    # Get the tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    except:
        print("Could not load Qwen tokenizer, using a basic one")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Get the VLM model to access its language model head
    vlm_model = policy.model.backbone.eagle_model

    # Try to get the language model head (this varies by model architecture)
    if hasattr(vlm_model, "language_model"):
        lm_head = vlm_model.language_model.lm_head
    elif hasattr(vlm_model, "lm_head"):
        lm_head = vlm_model.lm_head
    else:
        print("Could not find language model head for vocab projection")
        return None

    with torch.no_grad():
        # Project to vocabulary space: [seq_len, hidden_dim] -> [seq_len, vocab_size]
        logits = lm_head(hidden_states)

        # Convert to probabilities
        vocab_probs = torch.softmax(logits, dim=-1)

        # Get top-k words for each position
        seq_len = vocab_probs.shape[0]
        results = []

        for pos in range(min(seq_len, 20)):  # Only show first 20 positions
            top_values, top_indices = torch.topk(vocab_probs[pos], top_k)
            top_words = []

            for idx in top_indices:
                try:
                    word = tokenizer.decode([idx.item()]).strip()
                    if word:  # Only include non-empty words
                        top_words.append(word)
                except:
                    top_words.append(f"<token_{idx.item()}>")

            results.append({"position": pos, "words": top_words[:top_k], "probabilities": top_values.cpu().numpy()})

        return results


def main():
    print("Loading model and dataset...")
    policy, dataset = setup_model_and_data()

    # Get a data point
    step_data = dataset[0]
    print(f"Task description: {step_data.get('annotation.human.action.task_description', 'No description')}")

    # Run normal inference
    print("\nRunning inference...")
    predicted_action = policy.get_action(step_data)

    print("Predicted actions:")
    for key, value in predicted_action.items():
        print(f"  {key}: {value.shape}")

    # Try vocabulary projection
    print("\nAttempting vocabulary projection...")
    try:
        # Get VLM hidden states
        vlm_hidden = get_vlm_hidden_states(policy, step_data)

        if vlm_hidden is not None:
            # Do simple vocabulary projection
            vocab_results = simple_vocab_projection(vlm_hidden, policy, top_k=3)

            if vocab_results:
                print("\nTop words at each position (what the VLM is 'thinking'):")
                for result in vocab_results[:10]:  # Show first 10 positions
                    pos = result["position"]
                    words = result["words"]
                    probs = result["probabilities"]

                    word_prob_pairs = [f"{word}({prob:.3f})" for word, prob in zip(words, probs)]
                    print(f"  Position {pos:2d}: {', '.join(word_prob_pairs)}")
            else:
                print("Could not perform vocabulary projection")

    except Exception as e:
        print(f"Vocabulary projection failed: {e}")
        print("This is normal - vocabulary projection requires access to the language model head")

    # Show the image
    if "video.ego_view" in step_data:
        image = step_data["video.ego_view"][0]
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.title("Robot's view")
        plt.axis("off")
        plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
