# Training with LANCE Datasets

The `ShardedLanceDataset` class provides streaming access to LANCE formatted datasets. This setup supports dual-table architectures typically used for robotic learning:
- **MAIN_DATASET**: Contains the raw observations, images (as bytes), and actions arrays.
- **CURATED_DATASET**: A smaller, filtered dataset containing metadata, episode UUIDs, and quality flags (like `is_parallel_gripper` and `llm_score`).

## Configuration

To use the Lance dataset in your training runs, configure your dataset paths in `DataConfig` as a dictionary with `MAIN_DATASET` and `CURATED_DATASET` keys.

```yaml
data:
  mode: single_turn
  datasets:
    - embodiment_tag: "unitree_g1_full_body_with_waist_height_nav_cmd" # Or your target embodiment
      mix_ratio: 1.0
      dataset_paths:
        - MAIN_DATASET: "gs://your-gcp-bucket/lance/main_data.lance"
          CURATED_DATASET: "gs://your-gcp-bucket/lance/curated_data.lance"
```

## Manipulation-Only Slicing

When training humanoid robots (like the Unitree G1), the dataset automatically filters the `core` arrays to discard locomotion joints (legs, waist) and focus purely on manipulation.
For arrays with >= 29 joints, the first 15 indices are dropped, and the remaining arm joints are concatenated natively with `left_hand` and `right_hand` arrays.

## Features

- **Pushdown Filtering**: Uses PyArrow filtering to iteratively query matched UUIDs without loading massive episodes into memory.
- **Rolling Statistics**: Automatically calculates mean, std, min, max, q01, and q99 normalization bounds over the first 5000 sampled chunk arrays in a highly optimized stream.
- **Dynamic Reshaping**: Automatically reshapes flattened time-series action vectors into proper `(chunk_length, num_joints)` PyTorch tensors based on the configuration horizon.
