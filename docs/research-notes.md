# Archived GR00T Research Notes

This document preserves every unique substantive entry from the former
`.insights` corpus: 37 practitioner/general notes and 32 paper-derived findings.
The graph and index files duplicated these entries and added only projection
metadata, so they are not reproduced separately. Wording is retained from the
source wherever possible; incomplete, informal, outdated, or unverified entries
remain as they appeared and should be checked against current code,
documentation, and primary sources before use.

## Practitioner and general notes

- camera: ceiling better than head (left and right wrist cameras are also used)
- Importance of keeping task definition and environment consistent across simulation and real-robot runs when comparing results.
- If the robot action looks jerky, first check if the training data has the same problem. Try apply low-pass filter to training data. Try more denoising steps in inference.
- Common pattern: open-loop looks acceptable, but robot jitters, misses grasp, does not lift, twists joints, or executes wrong action.
- suggest: add history; camera footage=>better understand edge/corner location from image; collect data only through robot cameras than 3rd person view
- collected 100 episodes, suggest post-train directly, task is easy to overfit with 100-200 episodes
- Which is more important for training data, consistency or diversity? If you aim at medium SR, keeping consistent is OK; if you want very high SR, then diversity is necessary.
- Start with a small operational area, achieve high success rate, then expand distribution.
- GR00T N1.7 supports additional observation history via delta_indices (state or image or both). You will find an explanation in https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_config.md. We recommend adding video/image observation history for your task, before experimenting further with action_prefix conditioning. The reason is that N1.7 was trained with mixed
- Finally, we recommend verifying that the control stack used during data collection matches deployment as closely as possible, including IK solver, low-level controller, control frequency, action representation, and any smoothing or filtering.
- We recommend starting with video/image history only. State history can be tried afterward, but some internal experiments found that adding state history can make the model more likely to overfit.
- stiff control is preferable if possible, easier to learn as reducing latency
- When collecting data by tele-op, look at camera images but not the real robot. Make sure you see what the robot sees. It may affect speed and quality at the beginning, but will be fine after some practice.
- If the inference latency is too large, try to add torch.compile.
- Especially since you wrote your own trainers, you should make sure to recompute post-training dataset stats (see flag in https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/configs/data/data_config.py#L76). In the Isaac-GR00T repo, the flag controls whether the dataset stats computed from your finetuning dataset should replace the normalization stats baked into the pretrained checkpoint. You should make sure the recomputed stats are used during training and inference.
- GR00T N1.7 result sharing: Compared with N1.6, N1.7 performs much better on instruction-following and robustness tasks, and is roughly similar for manipulation tasks.
- You are collecting data in real and evaluating in Sim. It is important for the Sim environment to accurately reflect the real environment. It is worth testing if the policy works well, if trained on data collected in Sim, but also validating performance on the real setup.
- N1.7 supports relative action representations, and we recommend checking whether your 29D action vector should be configured as relative actions where applicable, if not yet. See getting_started/finetune_new_embodiment.md for guidance on configuring relative EEF for your robot. Representing actions as deltas from the current pose (rather than absolute targets) improves generalization and is a key factor in the model's cross-embodiment performance.
- If the task is not common or unseen in pretraining, you better turn on tune_llm and tune_visual to finetune parameters in the LLM backbone and vision encoder. In that case, you should use a lower LR than tuning the action head only. It will use more GPU memories.
- fail + retry behavior in data collection: avoid them, or only keep recovered parts (dagger, annotate good/bad intervals, only train with good)
- When collecting DAgger data, use the human trajectory only and discard the robot trajectory, because the robot trajectory leads the robot to a bad state. We do not want the model to imitate bad robot trajectories.
- Relative action/state is usually better than absolute action/state.
- rl training should be the final step to further boost the performance, dangerous on real robot (need to protect arm from collisions)
- harder tasks >400, cover all states including after first completion/recovery.
- Avoid collecting repetitive, identical demonstrations.
- Batch size 1024, learning rate 1e-4 (same as N1.5).
- Include controlled variation (grasp angle, object pose, limited workspace changes).
- Single-task experiment (pick_block_shape): Training only on a single task performs worse than joint training on all tasks.
- Recommendation to simplify first: Focus on single-object pick-and-place before moving to multi-step or long-horizon tasks.
- abs vs. rel: rel. action space predict may be too small to move; abs. may cause jump; rel. is smoother, generally better on real and def. better in sim.
- If the task requires high visual details, e.g., finding towel corners, you can try increase the image resolution.
- Short horizon in inference is good for action precision, while long horizon is good for action continuity.
- It is not recommended to start with UMI data in post-training, because there is an embodiment gap between the 3rd-party UMI gloves and your robot.
- If it is a new embodiment and unseen task, you can try two-stage training. First, turn off tune_llm and tune_visual and train action head only, it helps the action head to warm up on the new action space. Then turn on tune_llm and tune_visual, it helps the model to better adapt to the new task. If you turn on all parameters from the beginning, the action can be very noisy and may cause loss spikes.
- close loop first asap, see how it fails->adjust collection next
- policy hesitates: task is ambiguous, multiple ways from current obs
- As a first experiment, use one frame of history plus the current frame. Since your current replan interval is 8, this corresponds to video delta_indices=[-8, 0]. This matches our recommendation to use step_size = replan_frequency for the observation history.

## Paper-derived findings

- DreamDojo does not natively support multi-view simulation, which is identified as crucial for state-of-the-art policies
- Embodiment-specific state and action projector modules enable support for different robot embodiments including latent and IDM-based actions.
- GR00T N1 uses Eagle-2 as the base VLM, fine-tuned together with a flow-matching action generation model with action chunking.
- Co-training combines synthetic simulation data (MimicGen, DexMimicGen), neural-generated video data (~300k trajectories, 827 hours), and real-world robot data.
- Video generative models can produce neural trajectories at ~300k trajectory scale for robot training data augmentation.
- The System 1 and System 2 modules are jointly optimized during training rather than trained separately, facilitating coordination between reasoning and actuation
- Cross-attention mechanism bridges the VLM and action generation model instead of mixture-of-experts architecture, providing flexibility in model architecture choices.
- Initializing the last layer of the action MLP with zeros when adding action conditioning to a pretrained model prevents perturbation of the pretrained state and empirically improves physics modeling.
- Training data is organized as a pyramid structure with web data and human videos at the base, synthetic/simulation data in the middle, and real robot data at the top, enabling effective use of heterogeneous data sources
- A single unified model with one set of weights can generate manipulation behaviors across single-arm, bimanual, and humanoid embodiments through embodiment-specific encoders and decoders
- DexMimicGen scaled limited human demonstrations to 780,000 simulation trajectories (6,500 hours equivalent) in just 11 hours of compute time.
- DexMimicGen generates 10,000 demonstrations per source-target receptacle pair from a few dozen teleoperated source demonstrations, yielding 540k total simulation trajectories across 54 receptacle combinations.
- Post-training on task-specific data can cause capability regression: models post-trained exclusively with right-hand demonstrations lose the ability to perform bimanual handover behaviors present in the pre-trained checkpoint.
- Increasing human video data diversity during pretraining improves both physics modeling and controllability for out-of-distribution actions on downstream robot tasks.
- Using middle-layer LLM embeddings (layer 12 for the 2B model) instead of final-layer embeddings resulted in both faster inference speed and higher downstream policy success rate.
- For embodiments with few tasks and limited training data (DexMimicGen Cross-Embodiment Suite), reducing batch size from 1024 to 128 for GR00T-N1-2B was necessary.
- IDM-labeled neural trajectories outperform LAPA labels as training data increases, with the performance gap widening from 30 to 300 demonstrations per task.
- Relative action transformation and chunked action injection each significantly improve simulation quality and action controllability; combining them with temporal consistency loss further enhances both action following and object modeling.
- For distillation to real-time autoregressive generation, having the student model generate N′>N frames (more than the teacher's horizon N) during training and randomly selecting an N-sized window for supervision improves robustness against compounding errors over long horizons.
- Adding a temporal consistency loss (matching predicted velocity transitions between consecutive frames) with coefficient λ=0.1 to standard flow matching loss accelerates learning of action controllability, enhances object completeness, and reduces artifacts.
- Pretraining on human videos despite embodiment gap transfers underlying interaction physics to robot world models due to consistency of physical dynamics between humans and robots
- Human video datasets (Ego4D, Ego-Exo4D, Assembly-101, EPIC-KITCHENS, HOI4D, HoloAssist, RH20T-Human) provide affordances, task semantics, and motion patterns without explicit action labels for pre-training.
- K=4 inference steps for flow-matching denoising works well across all tested robot embodiments.
- Latent action conditioning during pretraining on human videos significantly narrows the performance gap to ground-truth action conditioning, while action-free pretraining provides only marginal benefits over no pretraining.
- Injecting actions into latent frames as temporally-aligned chunks (4 consecutive actions per chunk matching the 4-frame temporal compression ratio) rather than as global conditions mitigates causality confusion and improves action controllability and learning efficiency.
- Policy evaluation in DreamDojo produces absolute success rates that are systematically higher than real-world counterparts, suggesting the model fails to accurately generate nuanced failure modes
- Transforming robot joint poses to relative actions (rebaselining inputs with pose at beginning of each latent frame every 4 timesteps) reduces modeling complexity and enhances generalization to continuous and compositional robot actions compared to absolute poses.
- GR00T N1 uses a dual-system architecture where a Vision-Language Model (System 2) performs reasoning while a Diffusion Transformer (System 1) generates closed-loop motor actions.
- Human video pretraining benefits transfer through distillation, with distilled students from pretrained teachers significantly outperforming those from non-pretrained teachers across all evaluation datasets.
- Using continuous latent actions as unified proxy actions enables self-supervised extraction of semantically meaningful actions between frames, allowing effective transfer of physics and controllability from internet-scale passive video data
- Action-less data sources like human videos can be incorporated into VLA training by using a latent-action codebook and inverse dynamics model to infer pseudo-actions
- DreamDojo struggles to accurately simulate uncommon actions such as slapping and fast waving, indicating limitations in handling actions outside the common distribution of training data
