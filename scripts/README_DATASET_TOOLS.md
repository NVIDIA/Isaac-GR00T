# GR00T Dataset Tools

This directory contains utility scripts for working with GR00T LeRobot format datasets.

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

## Contributing

To add more dataset tools:
1. Create a new script in this directory
2. Follow the naming convention: `<tool_name>_dataset.py`
3. Update this README with usage examples
4. Include helpful console output with emoji indicators
