# Examples

## Modality.json

This provides additional examples of `modality.json` for different lerobot datasets. Copy the relevant `modality.json` to the dataset`<DATASET_PATH>/meta/modality.json`


## Eval GR00T N1 on SO100 and SO101

 - [eval_lerobot.py](./eval_lerobot.py): script to evaluate policy.
 - [eval_gr00t_so100.py](./eval_gr00t_so100.py) provides an example of how to use the finetuned model to run policy rollouts on a SO100 robot arm. (Before [Lerobot API redesign PR](https://github.com/huggingface/lerobot/pull/777))

> NOTE: This scripts meant to serve as a template, user will need to modify the script to run on a real robot.


## Tic-Tac-Toe Bot

<img src="./tictac_bot_setup.jpg" alt="Tic Tac Toe Bot" width="500"/>

```mermaid
graph TD
    subgraph "High-level Planner"
        A[Language Description] --> B[<b>VLM</b><br/>GPT-4/Gemini]
        C[Observation<br/>Image] --> B
        B --> D[Language Instruction<br/>e.g. place the circle to the bottom left corner box]
    end

    subgraph "Robot Control"
        E[Robot Observation<br/>Images + Proprioception] --> F[<b>VLA</b><br/>GR00T N1]
        D --> F
        F --> G[Robot Action]
    end

    style B fill:#bbf,stroke:#333,stroke-width:2px
    style F fill:#bbf,stroke:#333,stroke-width:2px
    style G fill:#bfb,stroke:#333,stroke-width:2px
```

This showcases the example of using a VLM as a high-level task planner (system 2) to plan the next action in a tic-tac-toe game, and GR00T N1 as the low-level action executor (system 1). This showcases language-conditioned on a GR00T N1 VLA. (e.g. "Place the circle to the bottom left corner box")

 * Example script: [tictac_bot.py](./tictac_bot.py)
 * [Example dataset](https://huggingface.co/datasets/youliangtan/tictac-bot)

```bash
# server
python scripts/inference_service.py --model_path <YOUR_CHECKPOINT_PATH> --server --data_config so100  --embodiment_tag new_embodiment

# client NOTE: this shouldn't run as it is, user will need to modify the script with relevant configs to make it work.
python tictac_bot.py
```
