# Deeper Understanding

In this section, we will dive deeper into the training configuration options. And we will also explain more about embodiment tags, modality configs, data transforms, and more.


## Embodiment Action Head Fine-tuning

GR00T is designed to work with different types of robots (embodiments) through specialized action heads. When fine-tuning, you need to specify which embodiment head to train based on your dataset:

1. **Embodiment Tags**
   - Each dataset must be tagged with a specific `EmbodimentTag` (e.g., EmbodimentTag.GR1_UNIFIED) while instantiating the `LeRobotSingleDataset` class
   - An exhaustive list of embodiment tags can be found in `gr00t/data/embodiment_tags.py`
   - This tag determines which action head will be fine-tuned
   - If you have a new embodiment, you can use the `EmbodimentTag.NEW_EMBODIMENT` tag (e.g., `new_embodiment.your_custom_dataset`)

2. **How it Works**
   - When you load your dataset with a specific embodiment tag (e.g., `EmbodimentTag.GR1_UNIFIED`)
   - The model has multiple components that can be configured for fine-tuning (Visual Encoder, Language Model, DiT, etc.)
   - For action heads specifically, only the one corresponding to your specified embodiment tag will be fine-tuned. Other embodiment-specific action heads remain frozen

## Advanced Tuning Parameters

### Model Components

The model has several components that can be fine-tuned independently. You can configure these parameters in the `GR00T_N1_5.from_pretrained` function.

1. **Visual Encoder** (`tune_visual`)
   - Set to `true` if your data has visually different characteristics from the pre-training data
   - Note: This is computationally expensive
   - Default: false


2. **Language Model** (`tune_llm`)
   - Set to `true` only if you have domain-specific language that's very different from standard instructions
   - In most cases, this should be `false`
   - Default: false

3. **Projector** (`tune_projector`)
   - By default, the projector is tuned
   - This helps align the embodiment-specific action and state spaces

4. **Diffusion Model** (`tune_diffusion_model`)
   - By default, the diffusion model is not tuned
   - This is the action head shared by all embodiment projectors

### Understanding Data Transforms

This document explains the different types of transforms used in our data processing pipeline. There are four main categories of transforms:

#### 1. Video Transforms

Video transforms are applied to video data to prepare it for model training. Based on our experimental evaluation, the following combination of video transforms worked best:

- **VideoToTensor**: Converts video data from its original format to PyTorch tensors for processing.
- **VideoCrop**: Crops the video frames, using a scale factor of 0.95 in random mode to introduce slight variations.
- **VideoResize**: Resizes video frames to a standard size (224x224 pixels) using linear interpolation.
- **VideoColorJitter**: Applies color augmentation by randomly adjusting brightness (±0.3), contrast (±0.4), saturation (±0.5), and hue (±0.08).
- **VideoToNumpy**: Converts the processed tensor back to NumPy arrays for further processing.

#### 2. State Transforms

State transforms process robot state information:

- **StateActionToTensor**: Converts state data (like arm positions, hand configurations) to PyTorch tensors.
- **StateActionTransform**: Applies normalization to state data. There are different normalization modes depending on the modality key. You can find the transformation logic in the [state_action.py](../gr00t/data/transform/state_action.py) file.

#### 3. Action Transforms

Action transforms process robot action data:

- **StateActionToTensor**: Similar to state transforms, converts action data to PyTorch tensors.
- **StateActionTransform**: Applies normalization to action data. Like with state data, min-max normalization is used to standardize action values for left/right arms, hands, and waist.

#### 4. Concat Transform

The **ConcatTransform** combines processed data into unified arrays:

- It concatenates video data according to the specified order of video modality keys.
- It concatenates state data according to the specified order of state modality keys.
- It concatenates action data according to the specified order of action modality keys.

This concatenation step is crucial as it prepares the data in the format expected by the model, ensuring that all modalities are properly aligned and ready for training or inference.

#### 5. GR00T Transform

The **GR00TTransform** is a custom transform that prepares the data for the model. It is applied last in the data pipeline.

- It pads the data to the maximum length of the sequence in the batch.
- It creates a dictionary of the data with keys as the modality keys and values as the processed data.

In practice, you typically won't need to modify this transform much, if at all.

### Lerobot Dataset Compatibility

More details about GR00T compatible lerobot datasets can be found in the [LeRobot_compatible_data_schema.md](./LeRobot_compatible_data_schema.md) file.
