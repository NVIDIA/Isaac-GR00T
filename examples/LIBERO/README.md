# LIBERO

Benchmark for studying knowledge transfer in lifelong robot learning. Includes multiple suites: **Spatial** (spatial reasoning), **Object** (object generalization), **Goal** (goal-conditioned learning), and **10 Long** (long-horizon multi-step tasks). Provides RGB images, proprioception data, and language task specifications.

For more information, see the [official website](https://libero-project.github.io/main.html).

---

# LIBERO evaluation benchmark result

> **Note:** The full task list is attached at the end of this document.

| Task      | Success rate       | max_steps | grad_accum_steps | batch_size |
|-----------|--------------------|-----------|------------------|------------|
| Spatial   | 195/200 (97.65%)        | 20K       | 1                | 640        |
| Goal      | 195/200 (97.5%)        | 20K       | 1                | 640        |
| Object    | 197/200 (98.45%)        | 20K       | 1                | 640        |
| 10 (Long) | 189/200 (94.35%)        | 20K       | 1                | 640        |

# Fine-tune LIBERO 10 (long)

To reproduce our finetune results, use the following commands to setup dataset and launch finetune experiments. Please remember to set `WANDB_API_KEY` since `--use-wandb` is turned on by default. If you don't have a WANDB account, please remove this argument:

```bash
huggingface-cli download \
    --repo-type dataset IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot \
    --local-dir examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/

# Copy the patches and run the finetune script
cp -r examples/LIBERO/modality.json examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/meta/
```

Run the finetune script:
```bash
uv run bash examples/LIBERO/finetune_libero_10.sh
```

# Fine-tune LIBERO goal

```bash
huggingface-cli download \
    --repo-type dataset IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot \
    --local-dir examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot/

# Copy the patches and run the finetune script
cp -r examples/LIBERO/modality.json examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot/meta/
## This is a patch for one of the episode where the image seems to be corrupted.
cp examples/LIBERO/patches/episode_000082.mp4 examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot/videos/chunk-000/observation.images.wrist_image/
```

Run the finetune script:
```bash
uv run bash examples/LIBERO/finetune_libero_goal.sh
```

# Fine-tune LIBERO object

```bash
huggingface-cli download \
    --repo-type dataset IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot \
    --local-dir examples/LIBERO/libero_object_no_noops_1.0.0_lerobot/

# Copy the patches and run the finetune script
cp -r examples/LIBERO/modality.json examples/LIBERO/libero_object_no_noops_1.0.0_lerobot/meta/
```

Run the finetune script:
```bash
uv run bash examples/LIBERO/finetune_libero_object.sh
```

# Fine-tune LIBERO spatial

```bash
huggingface-cli download \
    --repo-type dataset IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
    --local-dir examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot/

# Copy the patches and run the finetune script
cp -r examples/LIBERO/modality.json examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot/meta/
```

Run the finetune script:
```bash
uv run bash examples/LIBERO/finetune_libero_spatial.sh
```

# Evaluate checkpoint

First, setup the evaluation simulation environment. This only needs to run once for each simulation benchmark. After it's done, we only need to launch server and client.

```bash
sudo apt update
sudo apt install libegl1-mesa-dev libglu1-mesa
bash gr00t/eval/sim/LIBERO/setup_libero.sh
```

Then, run client server evaluation under the project root directory in separate terminals:

**Terminal 1 - Server:**
```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path /tmp/libero_spatial/checkpoint-20000/ \
    --embodiment-tag LIBERO_PANDA \
    --use-sim-policy-wrapper
```

**Terminal 2 - Client:**
```bash
gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --max_episode_steps=720 \
    --env_name libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \
    --n_action_steps 8 \
    --n_envs 5
```

# Full task list

## Libero 10 (Long)
- `libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`
- `libero_sim/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket`
- `libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`
- `libero_sim/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`
- `libero_sim/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`
- `libero_sim/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy`
- `libero_sim/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate`
- `libero_sim/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket`
- `libero_sim/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`
- `libero_sim/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it`

## Libero Goal
- `libero_sim/open_the_middle_drawer_of_the_cabinet`
- `libero_sim/put_the_bowl_on_the_stove`
- `libero_sim/put_the_wine_bottle_on_top_of_the_cabinet`
- `libero_sim/open_the_top_drawer_and_put_the_bowl_inside`
- `libero_sim/put_the_bowl_on_top_of_the_cabinet`
- `libero_sim/push_the_plate_to_the_front_of_the_stove`
- `libero_sim/put_the_cream_cheese_in_the_bowl`
- `libero_sim/turn_on_the_stove`
- `libero_sim/put_the_bowl_on_the_plate`
- `libero_sim/put_the_wine_bottle_on_the_rack`

## Libero Object
- `libero_sim/pick_up_the_alphabet_soup_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_cream_cheese_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_salad_dressing_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_bbq_sauce_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_ketchup_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_tomato_sauce_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_butter_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_milk_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_chocolate_pudding_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_orange_juice_and_place_it_in_the_basket`

## Libero Spatial
- `libero_sim/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate`