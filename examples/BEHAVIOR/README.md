# BEHAVIOR Benchmark Results

<div align="center">
  <video width="640" controls>
    <source src="../../media/behavior_n1d6.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

This is a benchmark of behavior1k from https://behavior.stanford.edu/

We provide a checkpoint: `nvidia/GR00T-N1.6-BEHAVIOR1k` which is post-trained on all 50 tasks. You can use this checkpoint for evaluation. 

## Multi-task (50 tasks) performance
We provide a base model pre-trained on all 50 tasks. It can do reasonably good on all tasks and serves as a good starting point for post-training. Here we use the metric `Task Progress`, which is a denser metric than `Q Score`. 

| Task Name | Task Progress N1.6 | Task Progress Pi0.5 |
| --- | --- | --- |
| sim_behavior_r1_pro/clean_a_trumpet | 60.00% | 8.00% |
| sim_behavior_r1_pro/getting_organized_for_work | 53.57% | 18.57% |
| sim_behavior_r1_pro/boxing_books_up_for_storage | 51.54% | 50.00% |
| sim_behavior_r1_pro/attach_a_camera_to_a_tripod | 46.00% | 34.00% |
| sim_behavior_r1_pro/make_microwave_popcorn | 45.00% | 17.50% |
| sim_behavior_r1_pro/picking_up_trash | 44.87% | 31.54% |
| sim_behavior_r1_pro/turning_on_radio | 43.33% | 16.67% |
| sim_behavior_r1_pro/clearing_food_from_table_into_fridge | 42.31% | 0.00% |
| sim_behavior_r1_pro/canning_food | 37.65% | 0.00% |
| sim_behavior_r1_pro/putting_away_Halloween_decorations | 34.74% | 0.00% |
| sim_behavior_r1_pro/set_up_a_coffee_station_in_your_kitchen | 34.55% | 7.27% |
| sim_behavior_r1_pro/collecting_childrens_toys | 33.81% | 20.48% |
| sim_behavior_r1_pro/preparing_lunch_box | 33.75% | 5.62% |
| sim_behavior_r1_pro/cook_bacon | 33.57% | 20.71% |
| sim_behavior_r1_pro/can_meat | 32.50% | 28.50% |
| sim_behavior_r1_pro/cook_cabbage | 30.67% | 13.33% |
| sim_behavior_r1_pro/carrying_in_groceries | 30.00% | 6.00% |
| sim_behavior_r1_pro/freeze_pies | 28.12% | 13.75% |
| sim_behavior_r1_pro/spraying_for_bugs | 27.50% | 18.75% |
| sim_behavior_r1_pro/bringing_water | 27.50% | 18.75% |
| sim_behavior_r1_pro/clean_boxing_gloves | 27.27% | 10.91% |
| sim_behavior_r1_pro/loading_the_car | 26.28% | 10.83% |
| sim_behavior_r1_pro/spraying_fruit_trees | 26.25% | 13.75% |
| sim_behavior_r1_pro/bringing_in_wood | 24.44% | 1.11% |
| sim_behavior_r1_pro/sorting_household_items | 24.12% | 0.00% |
| sim_behavior_r1_pro/clean_a_patio | 23.33% | 10.00% |
| sim_behavior_r1_pro/sorting_vegetables | 22.78% | 8.33% |
| sim_behavior_r1_pro/wash_a_baseball_cap | 22.73% | 20.00% |
| sim_behavior_r1_pro/setting_the_fire | 22.50% | 5.00% |
| sim_behavior_r1_pro/cleaning_up_plates_and_food | 22.00% | 15.33% |
| sim_behavior_r1_pro/putting_up_Christmas_decorations_inside | 21.49% | 3.68% |
| sim_behavior_r1_pro/hanging_pictures | 20.00% | 5.00% |
| sim_behavior_r1_pro/rearranging_kitchen_furniture | 20.00% | 5.00% |
| sim_behavior_r1_pro/putting_dishes_away_after_cleaning | 19.48% | 8.10% |
| sim_behavior_r1_pro/wash_dog_toys | 19.44% | 6.67% |
| sim_behavior_r1_pro/setting_mousetraps | 19.17% | 4.17% |
| sim_behavior_r1_pro/tidying_bedroom | 18.18% | 11.82% |
| sim_behavior_r1_pro/make_pizza | 18.00% | 0.00% |
| sim_behavior_r1_pro/outfit_a_basic_toolbox | 17.69% | 11.54% |
| sim_behavior_r1_pro/putting_shoes_on_rack | 16.67% | 20.67% |
| sim_behavior_r1_pro/chop_an_onion | 16.36% | 7.27% |
| sim_behavior_r1_pro/assembling_gift_baskets | 15.75% | 4.29% |
| sim_behavior_r1_pro/clean_up_your_desk | 15.05% | 12.17% |
| sim_behavior_r1_pro/moving_boxes_to_storage | 14.44% | 0.00% |
| sim_behavior_r1_pro/hiding_Easter_eggs | 13.64% | 0.00% |
| sim_behavior_r1_pro/picking_up_toys | 13.33% | 5.33% |
| sim_behavior_r1_pro/storing_food | 12.50% | 9.50% |
| sim_behavior_r1_pro/chopping_wood | 10.67% | 1.33% |
| sim_behavior_r1_pro/cook_hot_dogs | 0.28% | 23.85% |
| sim_behavior_r1_pro/slicing_vegetables | 0.14% | 0.00% |
| Average | 26.30% | 11.30% |

## 1. Individual task post-training (Optional)
Starting from the base checkpoint, we post-train on individual tasks and report results for some of them.

| Task Name | Task Progress | Q Score |
| --- | --- | --- |
| sim_behavior_r1_pro/turning_on_radio | 80.56% | 0.70 |
| sim_behavior_r1_pro/chopping_wood | 20.00% | 0.125 |
| sim_behavior_r1_pro/cleaning_up_plates_and_food | 22.00% | 0.11 |
| sim_behavior_r1_pro/setting_mousetraps | 19.17% | 0.10 |

# Fine-tune on BEHAVIOR dataset
First, download our converted BEHAVIOR dataset from HuggingFace
```
huggingface-cli download nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
  --repo-type dataset \
  --include "sim_behavior_r1_pro.*" \
  --local-dir $HOME/gr00t_dataset
```
Using `sim_behavior_r1_pro.*` will download datasets for all 50 tasks. You can replace `sim_behavior_r1_pro.*` with a specific task.

To launch training, run
```
uv run bash examples/BEHAVIOR/finetune_BEHAVIOR.sh
```
Notice the use of `BEHAVIOR_R1_PRO` embodiment tag.


# 2. Evaluate checkpoint

First, follow these steps to setup the BEHAVIOR simulation OmniGibson.
```bash
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K
# checkout to the branch with task progress as metric
git checkout feat/task-progress
# activate the uv env setup by gr00t, if not setup yet, run `cd PATH_TO_GR00T && uv sync --python 3.10 && uv pip install -e .`
source PATH_TO_GR00T/.venv/bin/activate
# Headless/automated installation (auto-accepts NVIDIA Isaac Sim EULA, and BEHAVIOR Dataset License)
bash ./setup_uv.sh
```

Because our evaluation was performed on test cases of [BEHAVIOR Challenge](https://behavior.stanford.edu/challenge/index.html), run the following script to download test cases from their [official HF repo](https://huggingface.co/datasets/behavior-1k/2025-challenge-hidden-instances/).
```bash
python gr00t/eval/sim/BEHAVIOR/prepare_test_instances.py
```

Note that BEHAVIOR sim is built on top of Omniverse and Isaac Sim, it inherits their spec dependencies. For example, GPUs without RT cores (A100, H100) are not supported. We tested on L40 and L40s. See [here](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html) for more information.

Then, run client server evaluation under the project root directory in separate terminals:

**Terminal 1 - Server:**
```bash
uv sync --python 3.10
uv pip install -e .

# replace the model path with the path to your finetuned checkpoint or use the provided checkpoint
uv run gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-BEHAVIOR1k \
    --embodiment-tag BEHAVIOR_R1_PRO \
    --use-sim-policy-wrapper
```

**Terminal 2 - Client:**
```bash
uv run python gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --max_episode_steps=999999999 \
    --env_name sim_behavior_r1_pro/turning_on_radio \
    --n_action_steps 8 \
    --n_envs 1
```
Note that we set `max_episode_steps` to a large value, this is because the BEHAVIOR sim will by default use 2x human steps as the horizon. Setting `max_episode_steps` to a smaller value if you want the evaluation to finish quicker, e.g., for debug purpose. Also, we disable video recording because we found the sim will crash if `decord` is imported in `video_utils.py`.

# Full task list
- sim_behavior_r1_pro/turning_on_radio
- sim_behavior_r1_pro/hanging_pictures
- sim_behavior_r1_pro/make_microwave_popcorn
- sim_behavior_r1_pro/attach_a_camera_to_a_tripod
- sim_behavior_r1_pro/picking_up_trash
- sim_behavior_r1_pro/clean_a_trumpet
- sim_behavior_r1_pro/set_up_a_coffee_station_in_your_kitchen
- sim_behavior_r1_pro/chop_an_onion
- sim_behavior_r1_pro/spraying_for_bugs
- sim_behavior_r1_pro/hiding_Easter_eggs
- sim_behavior_r1_pro/cook_bacon
- sim_behavior_r1_pro/putting_shoes_on_rack
- sim_behavior_r1_pro/clean_boxing_gloves
- sim_behavior_r1_pro/preparing_lunch_box
- sim_behavior_r1_pro/spraying_fruit_trees
- sim_behavior_r1_pro/wash_a_baseball_cap
- sim_behavior_r1_pro/rearranging_kitchen_furniture
- sim_behavior_r1_pro/setting_the_fire
- sim_behavior_r1_pro/bringing_water
- sim_behavior_r1_pro/cook_hot_dogs
- sim_behavior_r1_pro/setting_mousetraps
- sim_behavior_r1_pro/outfit_a_basic_toolbox
- sim_behavior_r1_pro/chopping_wood
- sim_behavior_r1_pro/putting_dishes_away_after_cleaning
- sim_behavior_r1_pro/tidying_bedroom
- sim_behavior_r1_pro/wash_dog_toys
- sim_behavior_r1_pro/can_meat
- sim_behavior_r1_pro/sorting_vegetables
- sim_behavior_r1_pro/clean_a_patio
- sim_behavior_r1_pro/freeze_pies
- sim_behavior_r1_pro/clearing_food_from_table_into_fridge
- sim_behavior_r1_pro/bringing_in_wood
- sim_behavior_r1_pro/cleaning_up_plates_and_food
- sim_behavior_r1_pro/putting_up_Christmas_decorations_inside
- sim_behavior_r1_pro/putting_away_Halloween_decorations
- sim_behavior_r1_pro/cook_cabbage
- sim_behavior_r1_pro/carrying_in_groceries
- sim_behavior_r1_pro/moving_boxes_to_storage
- sim_behavior_r1_pro/getting_organized_for_work
- sim_behavior_r1_pro/sorting_household_items
- sim_behavior_r1_pro/picking_up_toys
- sim_behavior_r1_pro/collecting_childrens_toys
- sim_behavior_r1_pro/make_pizza
- sim_behavior_r1_pro/loading_the_car
- sim_behavior_r1_pro/storing_food
- sim_behavior_r1_pro/clean_up_your_desk
- sim_behavior_r1_pro/canning_food
- sim_behavior_r1_pro/boxing_books_up_for_storage
- sim_behavior_r1_pro/assembling_gift_baskets
- sim_behavior_r1_pro/slicing_vegetables
