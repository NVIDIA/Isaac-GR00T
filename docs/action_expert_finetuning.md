# Fine-tuning N2 with an action expert

N2 can optionally route state and action tokens through a separate, trainable copy of each
transformer block while keeping the pretrained video path frozen. This mode is intended for
adapting action prediction to a downstream embodiment without updating the base model's video
weights.

Enable `action_expert: true` in `GrootN2Config` and initialize the run from a normal dense N2
checkpoint with `start_from_path`. The checkpoint does not need to contain action-expert weights:
the loader initializes the new AdaLN, self-attention Q/K/V/output, text cross-attention, FFN, and
final normalization weights from their dense counterparts. The action/state encoders and action
decoder remain trainable as usual.

The mode requires:

- `action_cfg` to be configured;
- `tune_dit: true` and `train_architecture: full`;
- an asymmetric video/action attention pattern (the default
  `disjoint_video_action` pattern is supported); and
- no tactile or deform token groups.

The asymmetric mask is important: action queries can read video keys and values, but video queries
do not read state/action keys and values. Consequently, updating the expert cannot change the video
token path. `attention_mask_pattern: full` is rejected because it violates that isolation.

This mode adds a second copy of the per-block transformer weights, increasing checkpoint and model
memory substantially even though only the action expert is trainable. It is an experimental
fine-tuning option, not a guarantee against overfitting; downstream runs should still monitor held-
out action metrics and inspect generated video and action rollouts separately.
