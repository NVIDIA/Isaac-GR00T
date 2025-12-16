# Robot Data Preparation Guide

## Overview

This guide shows how to convert your robot data to work with our flavor of the [LeRobot dataset V2 format](https://github.com/huggingface/lerobot?tab=readme-ov-file#the-lerobotdataset-format) -- `GR00T LeRobot`. While we have added additional structure, our schema maintains full compatibility with the upstream LeRobot v2. The additional metadata and structure allow for more detailed specification and language annotations for your robot data.

> The TLDR: Add a `meta/modality.json` file to your LeRobot v2 dataset and follow the schema below.

## LeRobot v2 Requirements

If you already have a dataset in the LeRobot v2 format, you can skip this section.

If you have a dataset in the LeRobot v3.0 format, please use [this script](../scripts/lerobot_conversion/convert_v3_to_v2.py) to convert it to the LeRobot v2 format.

If you have a dataset in another format, please convert it to the LeRobot v2 format satisfying the following requirements.

### Structure Requirements

The folder should follow a similar structure as below and contain these core folders and files:

```
.
├─meta 
│ ├─episodes.jsonl
│ ├─modality.json # -> GR00T LeRobot specific
│ ├─info.json
│ └─tasks.jsonl
├─videos
│ └─chunk-000
│   └─observation.images.ego_view
│     └─episode_000001.mp4
│     └─episode_000000.mp4
└─data
  └─chunk-000
    ├─episode_000001.parquet
    └─episode_000000.parquet
```

### Video Observations (video/chunk-*)
The videos folder will contain the mp4 files associated with each episode following episode_00000X.mp4 naming format where X indicates the episode number.
**Requirements**:
- Must be stored as MP4 files.
- Should be named using the format: `observation.images.<video_name>`


### Data (data/chunk-*)
The data folder will contain all of the parquet files associated with each episode following episode_00000X.parquet naming format where X indicates the episode number.
Each parquet file will contain:
- State information: stored as observation.state which is a 1D concatenated array of all state modalities.
- Action: stored as action which is a 1D concatenated array of all action modalities.
- Timestamp: stored as timestamp which is a float point number of the starting time.
- Annotations: stored as annotation.<annotation_source>.<annotation_type>(.<annotation_name>) (see the annotation field in the example configuration for example naming.).  No other columns should have the annotation prefix, see the (multiple-annotation-support) if interested in adding multiple annotations.

#### Example Parquet File
Here is a sample of the `cube_to_bowl` dataset that is present in the [demo_data](../demo_data/cube_to_bowl_5/) directory.
```
{
    "observation.state":[-0.01147082911843003,...,0], // concatenated state array based on the modality.json file
    "action":[-0.010770668025204974,...0], // concatenated action array based on the modality.json file
    "timestamp":0.04999995231628418, // timestamp of the observation
    "annotation.human.action.task_description":0, // index of the task description in the meta/tasks.jsonl file
    "task_index":0, // index of the task in the meta/tasks.jsonl file
    "annotation.human.validity":1, // index of the task in the meta/tasks.jsonl file
    "episode_index":0, // index of the episode
    "index":0, // index of the observation. This is a global index across all observations in the dataset.
    "next.reward":0, // reward of the next observation
    "next.done":false // whether the episode is done
}
```

### Meta

- `episodes.jsonl` contains a list of all the episodes in the entire dataset. Each episode contains a list of tasks and the length of the episode.
- `tasks.jsonl` contains a list of all the tasks in the entire dataset.
- `info.json` contains the dataset information.

#### meta/tasks.jsonl
Here is a sample of the `meta/tasks.jsonl` file that contains the task descriptions.
```
{"task_index": 0, "task": "pick the squash from the counter and place it in the plate"}
{"task_index": 1, "task": "valid"}
```

You can refer the task index in the parquet file to get the task description. So in this case, the `annotation.human.action.task_description` for the first observation is "pick the squash from the counter and place it in the plate" and `annotation.human.validity` is "valid".

`tasks.json` contains a list of all the tasks in the entire dataset.

#### meta/episodes.jsonl

Here is a sample of the `meta/episodes.jsonl` file that contains the episode information.

```
{"episode_index": 0, "tasks": [...], "length": 416}
{"episode_index": 1, "tasks": [...], "length": 470}
```

`episodes.json` contains a list of all the episodes in the entire dataset. Each episode contains a list of tasks and the length of the episode.

## GR00T LeRobot Specific Requirements

### The `meta/modality.json` Configuration

We require an additional metadata file `meta/modality.json` that is not present in the standard LeRobot format. This file provides detailed metadata about state and action modalities, enabling:

- **Separate Data Storage and Interpretation:**
  - **State and Action:** Stored as concatenated float32 arrays. The `modality.json` file supplies the metadata necessary to interpret these arrays as distinct, fine-grained fields.
  - **Video:** Stored as separate files, with the configuration file allowing them to be renamed to a standardized format.
  - **Annotations:** Keeps track of all annotation fields. If there are no annotations, do not include the `annotation` field in the configuration file.
- **Fine-Grained Splitting:** Divides the state and action arrays into more semantically meaningful fields.
- **Clear Mapping:** Explicit mapping of data dimensions.
- **Sophisticated Data Transformations:** Supports field-specific normalization and rotation transformations during training.

#### Schema

```json
{
    "state": {
        "<state_key>": {
            "start": <int>,         // Starting index in the state array
            "end": <int>            // Ending index in the state array
        }
    },
    "action": {
        "<action_key>": {
            "start": <int>,         // Starting index in the action array
            "end": <int>            // Ending index in the action array
        }
    },
    "video": {
        "<new_key>": {
            "original_key": "<original_video_key>"
        }
    },
    "annotation": {
        "<annotation_key>": {}  // Empty dictionary to maintain consistency with other modalities
    }
}
```

#### Notes

- All indices are zero-based and follow Python's array slicing convention (`[start:end]`).

## GR00T LeRobot Extensions to Standard LeRobot
GR00T LeRobot is a flavor of the standard LeRobot format with more opinionated requirements:
- We will compute `meta/stats.json` and `meta/relative_stats.json` for each dataset, and store them in the `meta` folder.
- Proprioceptive states must always be included in the "observation.state" keys.
- We support multi-channel annotation formats (e.g., coarsegrained, finetuned), allowing users to add as many annotation channels as needed via the `annotation.<annotation_source>.<annotation_type>` key.
- We require an additional metadata file `meta/modality.json` that is not present in the standard LeRobot format.

### Multiple Annotation Support

To support multiple annotations within a single parquet file, users may add extra columns to the parquet file. Users should treat these columns the same way as the `task_index` column in the original LeRobot v2 dataset:

In LeRobot v2, actual language descriptions are stored in a row of the `meta/tasks.jsonl` file, while the parquet file stores only the corresponding index in the `task_index` column. We follow the same convention and store the corresponding index for each annotation in the `annotation.<annotation_source>.<annotation_type>` column. Although the `task_index` column may still be used for the default annotation, a dedicated column `annotation.<annotation_source>.<annotation_type>` is required to ensure it is loadable by our custom data loader.
