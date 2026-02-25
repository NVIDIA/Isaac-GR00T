# GR00T Dataset and Configuration Tools

This directory contains utility scripts for working with GR00T LeRobot format datasets and robot embodiment configurations.

## Available Tools

### 1. Dataset Validator (`validate_dataset.py`)

Validates GR00T LeRobot datasets to ensure they follow the correct format and structure.

**Features:**
- ✅ Directory structure validation
- ✅ Metadata file validation (modality.json, episodes.jsonl, tasks.jsonl, info.json)
- ✅ Modality configuration verification
- ✅ Parquet file structure and content validation
- ✅ Video file validation
- ✅ State/action dimension consistency checks
- ✅ Dataset statistics calculation

**Usage:**

```bash
# Basic validation
python scripts/validate_dataset.py /path/to/dataset

# Verbose output
python scripts/validate_dataset.py /path/to/dataset --verbose
```

**Output:**
The script provides:
- File existence checks
- JSON/JSONL format validation
- Column presence verification
- Sample parquet file inspection
- Summary statistics (number of episodes, frames, total size, etc.)
- Error and warning report

**Example Output:**
```
🔍 Validating GR00T dataset: /path/to/dataset
======================================================================

📂 Checking directory structure...
   ✓ meta/ exists
   ✓ videos/ exists
   ✓ data/ exists

📋 Checking metadata files...
   ✓ modality.json exists
   ✓ episodes.jsonl exists
   ✓ tasks.jsonl exists
   ✓ info.json exists

⚙️  Checking modality configuration...
   ✓ modality.json is valid
   ✓ state configuration found
   ✓ action configuration found

...

📊 Dataset Statistics
----------------------------------------------------------------------
   num_episodes: 50
   num_videos: 50
   num_parquet_files: 50
   total_frames: 25000
   total_data_size_mb: 1234.56

======================================================================
📋 VALIDATION REPORT
======================================================================

✅ ALL VALIDATION CHECKS PASSED!
```

---

### 2. Dataset Inspector (`inspect_dataset.py`)

Provides detailed inspection and analysis of GR00T LeRobot datasets.

**Features:**
- 📂 Dataset structure overview
- 📋 Metadata insights (episodes, tasks, info)
- ⚙️ Detailed modality configuration analysis
- 📊 Comprehensive data statistics
- 🤖 Embodiment type detection hints

**Usage:**

```bash
# Basic inspection
python scripts/inspect_dataset.py /path/to/dataset

# Save report to JSON
python scripts/inspect_dataset.py /path/to/dataset --output dataset_report.json
```

**Output:**
The script provides:
- Directory structure breakdown (chunks, files)
- Metadata overview (number of episodes, tasks, etc.)
- Modality configuration details (state fields, action fields, video streams, annotations)
- Data statistics (total frames, size, file counts)
- Embodiment hints based on action/state dimensions

**Example Output:**
```
🔎 Inspecting GR00T dataset: /path/to/dataset
======================================================================

📂 Dataset Structure
----------------------------------------------------------------------
   Meta files: modality.json, episodes.jsonl, tasks.jsonl, info.json
   Videos: 5 chunks, 50 total MP4 files
   Data: 5 chunks, 50 total parquet files

📋 Metadata Overview
----------------------------------------------------------------------
   Episodes: 50
      Sample: 3 tasks, 500 frames
   Tasks: 12 unique tasks
      Sample: 'pick up the cube from the table'
   Info: {"dataset_name": "example_dataset", ...}

⚙️  Modality Configuration
----------------------------------------------------------------------
   State fields (5):
      • gripper_position: dims [0:3] (size: 3)
      • gripper_angle: dims [3:4] (size: 1)
      • joint_angles: dims [4:10] (size: 6)
      Total state dimension: 10

   Action fields (3):
      • arm_action: dims [0:6] (size: 6)
      • gripper_action: dims [6:7] (size: 1)
      • ee_action: dims [7:10] (size: 3)
      Total action dimension: 10

   Video streams (1):
      • ego_view (original: observation.images.ego_view)

📊 Data Statistics
----------------------------------------------------------------------
   Total frames: 25000
   Total size: 1234.56 MB
   Avg frames per file: 500.0
   Files with annotations: 50
   Unique annotation types: 2

🤖 Embodiment Hints
----------------------------------------------------------------------
   State dimension: 10
   Action dimension: 10
   🦾 Likely arm/gripper embodiment (medium action DOF)

✅ Inspection Complete!
```

**Saving Report:**
Use `--output` to save a detailed JSON report:
```bash
python scripts/inspect_dataset.py /path/to/dataset --output report.json
```

---

## Common Workflows

### 1. Validate a new dataset before training

```bash
python scripts/validate_dataset.py /path/to/your/dataset --verbose
```

If validation passes, your dataset is ready for training!

### 2. Understand your dataset structure

```bash
python scripts/inspect_dataset.py /path/to/your/dataset
```

### 3. Generate a detailed report for documentation

```bash
python scripts/inspect_dataset.py /path/to/your/dataset --output dataset_analysis.json
```

### 4. Debug dataset issues

1. First run the validator to identify specific issues:
   ```bash
   python scripts/validate_dataset.py /path/to/dataset --verbose
   ```

2. Then inspect the dataset structure for more context:
   ```bash
   python scripts/inspect_dataset.py /path/to/dataset
   ```

---

## Requirements

Both scripts require:
- Python 3.10+
- pandas
- numpy
- tqdm

Install with:
```bash
pip install pandas numpy tqdm
```

These are already included in the GR00T environment.

---

## Data Format Reference

For detailed information about the GR00T LeRobot format, see:
- [Data Preparation Guide](../getting_started/data_preparation.md)
- [Modality Configuration Reference](../getting_started/data_preparation.md#the-metamodalityjson-configuration)

---

## Troubleshooting

### Script won't run
- Ensure you're in the GR00T environment: `source .venv/bin/activate`
- Check Python version: `python --version` (should be 3.10+)

### Import errors
- Install missing dependencies: `pip install pandas numpy tqdm`

### Dataset validation fails
- Run with `--verbose` flag for more details
- Check the error messages carefully
- Refer to the [Data Preparation Guide](../getting_started/data_preparation.md)
- Verify your dataset structure matches the LeRobot v2 format

---

### 3. Embodiment Configuration Reference and Debugger (`embodiment_config_reference.py`)

Provides reference information and debugging utilities for GR00T robot embodiment configurations.

**Features:**
- ✅ List all available embodiments (pre-trained and post-training)
- ✅ Show detailed configuration for specific embodiments
- ✅ Display state/action dimensions for each embodiment
- ✅ View modality keys and action configurations
- ✅ Generate configuration templates for new robots
- ✅ Validate custom configuration files
- ✅ Summary table of all embodiments

**Usage:**

```bash
# List all available embodiments
python scripts/embodiment_config_reference.py --list

# Show full summary table with dimensions
python scripts/embodiment_config_reference.py --all

# Show detailed configuration for a specific embodiment
python scripts/embodiment_config_reference.py --show unitree_g1
python scripts/embodiment_config_reference.py --show oxe_google

# Generate template for custom robot configuration
python scripts/embodiment_config_reference.py --template my_robot

# Validate a custom configuration file
python scripts/embodiment_config_reference.py --validate config.py
```

**Output Examples:**

**List command:**
```
📋 Available GR00T Embodiments
======================================================================

✅ unitree_g1               - Unitree G1 humanoid
✅ libero_panda             - Libero Panda robot
✅ oxe_google              - Open-X-Embodiment Google robot
✅ oxe_widowx              - Open-X-Embodiment WidowX robot
✅ oxe_droid               - Open-X-Embodiment DROID robot
✅ behavior_r1_pro         - Behavior R1 Pro robot
⚠️ gr1                     - Custom embodiment (no config)
⚠️ new_embodiment          - Placeholder for new configurations

Total: 9 embodiments
✅ = Configuration available, ⚠️ = Configuration not found
```

**All command (Summary Table):**
```
🤖 Embodiment Configuration Summary
================================================================================

Embodiment           | State Dims | Action Dims | Videos | Status
---------------------------------------------------------------------------
unitree_g1           | 7          | 7           | 1      | ✓
libero_panda         | 7          | 7           | 2      | ✓
oxe_google           | 8          | 7           | 1      | ✓
oxe_widowx           | 8          | 7           | 1      | ✓
oxe_droid            | 2          | 2           | 2      | ✓
behavior_r1_pro      | 21         | 6           | 3      | ✓
gr1                  | —          | —           | —      | No Config
new_embodiment       | —          | —           | —      | No Config
```

**Show command (Detailed Configuration):**
```
🤖 Embodiment Configuration: unitree_g1
======================================================================

Configuration Details:
--------------------------------------------------------------

📊 STATE MODALITY:
   Keys: left_leg, right_leg, waist, left_arm, right_arm, left_hand, right_hand
   Delta Indices: [0]
   Dimension: 7 modalities

🎮 ACTION MODALITY:
   Keys: left_arm, right_arm, left_hand, right_hand, waist, 
         base_height_command, navigate_command
   Delta Indices: [0, 1, 2, ..., 28, 29]
   Action Configs: 7 defined
      [0] rep=RELATIVE, type=NON_EEF
      [1] rep=RELATIVE, type=NON_EEF
      [2-6] rep=ABSOLUTE, type=NON_EEF
   Dimension: 7 modalities

🎬 VIDEO MODALITY:
   Streams: ego_view
   Delta Indices: [0]
```

**Key Information:**

- **State Dims**: Number of state modality groups (e.g., left_arm, gripper)
- **Action Dims**: Number of action modality groups
- **Videos**: Number of camera streams
- **Delta Indices**: Which modalities have temporal deltas
- **Action Configs**: Details about action representations (RELATIVE vs ABSOLUTE)

**Use Cases:**

1. **Understanding embodiment requirements:**
   ```bash
   python scripts/embodiment_config_reference.py --show oxe_droid
   ```

2. **Creating custom robot configuration:**
   ```bash
   python scripts/embodiment_config_reference.py --template my_arm
   # Edit the generated template to match your robot
   ```

3. **Comparing embodiments:**
   ```bash
   python scripts/embodiment_config_reference.py --all
   # Quickly see dimensions and supported features
   ```

---

## Common Workflows

### 1. Complete dataset preparation and validation

```bash
# 1. Prepare your data in LeRobot v2 format
# 2. Create modality.json for your dataset
# 3. Inspect the dataset structure
python scripts/inspect_dataset.py /path/to/dataset

# 4. Validate the dataset before training
python scripts/validate_dataset.py /path/to/dataset --verbose

# 5. Check if embodiment configuration is correct
python scripts/embodiment_config_reference.py --show your_embodiment
```

### 2. Setting up a new robot

```bash
# 1. Generate a configuration template
python scripts/embodiment_config_reference.py --template my_robot

# 2. Customize the template for your robot
# 3. Validate the configuration (once implemented)
python scripts/embodiment_config_reference.py --validate my_robot_config.py

# 4. Prepare your training data following the LeRobot format
# 5. Validate your dataset
python scripts/validate_dataset.py /path/to/my_robot_data
```

### 3. Understanding the pretraining data structure

```bash
# See what embodiments are available
python scripts/embodiment_config_reference.py --all

# Understand a specific embodiment's configuration
python scripts/embodiment_config_reference.py --show unitree_g1
```

---

## Requirements

Both scripts require:
- Python 3.10+
- pandas (for dataset tools) 
- numpy
- tqdm

Install with:
```bash
pip install pandas numpy tqdm
```

These are already included in the GR00T environment.

---

## Contributing

To add more tools:
1. Create a new script in this directory
2. Follow the naming convention: `<tool_name>.py`
3. Update this README with usage examples
4. Include helpful console output with emoji indicators
5. Add meaningful comments and docstrings
