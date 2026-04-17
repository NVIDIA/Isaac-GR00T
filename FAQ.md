# GR00T N1.7 FAQ

## Infrastructure & Hardware

### Is the data loader GPU-accelerated?

No, the current data loader is CPU-based. However, it has been heavily optimized for multimodal data to ensure it does not become a training bottleneck. We validated this on various configurations, including GB200, H100, and local desktops with RTX 4090 GPUs. We are actively exploring GPU-accelerated approaches for future releases.

### Is the same data loader used for both pre-training and post-training?

Yes, the data loading pipeline is unified across both training stages.

### What is the role of the Policy Remote Server in the deployment diagram?

The Policy Remote Server decouples inference from the physical robot. This allows users to run the policy on a high-compute cluster (e.g., H100s) for faster inference while the robot operates in a separate environment. It separates dependencies and enables scaling beyond the robot's onboard compute.

## Workflow & Architecture

### How is the Whole-Body Control (WBC) stack used with GR00T N models?

The WBC and the GR00T VLA policy are trained independently and coupled at deployment. The typical workflow for integrating them involves four key steps:

1. **Train WBC**: Train a low-level controller for the specific robot hardware.
2. **Collect Data**: Use the trained WBC to collect "loco-manipulation" data.
3. **Fine-tune**: Train the GR00T model on this data.
4. **Deploy**: Run the fine-tuned GR00T policy, which outputs commands to the WBC via an adapter interface.

See the [Loco-Manipulation workflow example](https://github.com/NVlabs/GR00T-WholeBodyControl) for details.

### Why retain only specific LLM layers (e.g., 16 layers) during fine-tuning?

This configuration was empirically tuned for the backbone (e.g., Eagle/Cosmos-Reason). Research suggests early layers capture grammatical structure, while middle-to-late layers are highly expressive. However, the very last layers are often over-optimized for next-token prediction; pruning or freezing them can sometimes yield better representations for vision-language-action alignment.

### How do you verify if the language model is successfully aligned with the action space?

We evaluate this end-to-end via downstream task success. We design evaluation tasks that are ambiguous without language instructions (e.g., "pick the pear" from a bowl of mixed fruit). If the robot succeeds, it confirms the model is correctly grounding language commands into physical actions.

## Data Strategy & Volume

### How much data is required for post-training on a new embodiment or task?

Data requirements depend heavily on task complexity and scene variation. Typical guidelines include:

- **Simple, fixed-location tasks (Pick & Place):** ~100 trajectories.
- **Complex scenes or multi-step tasks:** ~500+ trajectories.
- **Whole-Body Control (High DoF):** ~2,000+ trajectories (e.g., shelf-picking with G1).
- **Fine manipulation:** ~100–500 episodes, ideally with human motion pre-training.

### What is the recommended strategy for improving success rates on hard tasks?

We recommend an iterative approach: start with ~100 teleoperated demonstrations, train a policy, and then use HG-DAgger (Human Gated Dataset Aggregation). Run the policy, intervene when it fails, and add the corrections from those trajectories to the dataset. This helps the model cover out-of-distribution states that pure behavior cloning (BC) might miss, and recover from partial failure states (e.g., a grip slipping or imprecise item placement).

### Does including real-robot data from other embodiments help if I only care about one robot?

Yes. Even if cross-embodiment generalization is not your goal, including diverse real-robot data adds visual diversity and robustness to the VLA's backbone, improving performance on your specific target robot.

### Does GR00T N1.7 support synthetic data generation via Cosmos?

While research models (like DreamGen) show promise, a robust, product-ready pipeline for generating synthetic training data via Cosmos is currently in development and not yet part of the standard release.

## Model Capabilities

### Can the model handle lighting changes or different object colors?

VLMs can struggle with drastic appearance changes (e.g., hard shadows or significant hue shifts). While we haven't released specific lighting ablations, we strongly recommend using color jitter augmentation during training and collecting diverse data (20–50 episodes) under different lighting conditions to prevent overfitting.

### Can GR00T models perform reasoning or Visual Question Answering (VQA)?

The GR00T N1.x series is optimized specifically for action generation, not open-ended reasoning or VQA. Capabilities requiring complex semantic reasoning are targeted for future N2 releases.

### Can the model learn "retry" behaviors?

The current architecture is stateless and does not inherently "know" if a previous attempt failed. While some retry behavior may emerge from high-quality data, explicit recovery strategies are best achieved through DAgger (collecting data on recovery from failure) or Reinforcement Learning (RL), rather than pure Imitation Learning.

### Does the model distinguish between left and right arms in bimanual tasks?

Yes, provided the training data is distinct or annotated (e.g., instructions specifying "left arm" vs. "right arm"). If the dataset contains mixed, unannotated data where both arms perform identical tasks indiscriminately, the model may struggle to distinguish them.

### Is there a zero-shot cross-embodiment VLA model?

No. While cross-embodiment data improves generalization, a true "zero-shot" model (one that works perfectly on a new robot without *any* fine-tuning) does not currently exist in the open VLA landscape.

### Will differences in object shape between training and deployment cause the success rate to drop?

It depends on the degree of deviation. If the target object's shape differs drastically from the training data, performance will likely drop significantly. However, if the shape variation is minor and shares a similar grasping affordance (e.g., a slightly different bottle shape that is still grasped from the side), the model may still succeed, though with potentially lower reliability than on the original objects.

### Has the impact of large viewpoint changes (e.g., head movement) on task difficulty been studied?

Yes. Large viewpoint changes effectively change the observation distribution, which can complicate simple tasks. For example, a "simple" handover becomes complex if the robot's head moves significantly, altering the camera's perspective of its own hands.

- **Current Status:** Most public GR00T demos feature a relatively fixed head position to stabilize observations.
- **Mitigation:** To handle natural head movement, we recommend training with aggressive camera pose augmentation or collecting data that explicitly includes head motion to ensure the policy becomes robust to viewpoint shifts.
