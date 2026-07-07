# GR00T Whole-Body Control Real-World Benchmark

This example reports a real-world Unitree G1 evaluation using **Isaac GR00T N1.7** together with **GR00T Whole-Body Control / GEAR-SONIC**. The benchmark focuses on everyday mobile-manipulation tasks that require walking, table approach, grasping, foot placement, and whole-body pickup motions.

## Summary

GR00T N1.7 with SONIC can execute closed-loop whole-body skills on a real humanoid robot. The main targeted task is **walk to a table and pick up an object**. A single mixed-object policy was trained from demonstrations covering about 50 table-top objects, then evaluated systematically across 8 representative objects in the training set. This follows the object-pickup setting studied in the [SONIC paper](https://nvlabs.github.io/GEAR-SONIC/static/pdf/sonic_paper.pdf), where the robot walks to a table, locates a target object, and grasps it under randomized table heights and object positions. A second task-specific policy was trained for walking to a small table, picking up a soda can, stepping on a trash-can trigger, and dropping the can inside.

For more GR00T Whole-Body Control task examples, see the [GEAR-SONIC project page](https://nvlabs.github.io/GEAR-SONIC/).

## Evaluation Results

| Task Family | Policy Setup | Task Specification | Logged Trials | Successes | Success Rate |
| --- | --- | --- | ---: | ---: | ---: |
| Walk to table and pick up object | Single mixed-object pickup policy | Walk to the table, localize a target object, grasp it, and lift it. Evaluated on 8 objects across 3 table heights and 9 object placements. Autonomous retries count as success. | 216 | 159 | 73.6% |
| Soda can from small table to trash can | Task-specific soda-can policy | Walk toward a small table, pick up a soda can, rotate toward the trash can, step on the trigger, and drop the can inside. | 12 | 8 | 66.7% |

For object pickup, each object was evaluated with 27 trials: 3 table heights (24, 27, and 30 inches) crossed with a 3 by 3 grid of object placements on the table (left/middle/right and front/middle/back). The failures were mostly grasp failures: the gripper missed the object, contacted it from an unstable grasp point, pushed it out of reach, or dropped it into a pose that the policy could not recover from. The soda-can failures were similarly dominated by missed grasps, unstable grasps, or missed trigger steps.

### Object Pickup Breakdown

| Object | Trials | Successes | Success Rate |
| --- | ---: | ---: | ---: |
| Towel | 27 | 24 | 88.9% |
| Shoe | 27 | 14 | 51.9% |
| Apple | 27 | 19 | 70.4% |
| Scoop | 27 | 18 | 66.7% |
| Lamp | 27 | 17 | 63.0% |
| Flashlight | 27 | 20 | 74.1% |
| Fruit | 27 | 26 | 96.3% |
| Sock | 27 | 21 | 77.8% |

## Demo Videos

The MP4 samples below are attached as video files in this repository. Click any preview image to open the corresponding MP4. The source tree intentionally keeps only a curated sample set; full per-trial review videos are better shared as supplementary material through a stable gallery or archive. For more task demos beyond these two quantified results, see the [GEAR-SONIC project page](https://nvlabs.github.io/GEAR-SONIC/).

### Walk To Table And Pick Up Objects

Representative examples are included for the evaluated object pickup task. Captions describe the behavior visible in the clip; some successful clips include retries, regrasping, object contact, or small recovery motions because autonomous recovery was counted as success during evaluation. A trial is considered failure when it stucks for over 30s without progress.

| Object | Example 1 | Example 2 |
| --- | --- | --- |
| Lamp | [<img src="media/g1_real_eval/posters/mixed_pickup_lamp_01_pickup_success.jpg" alt="Lamp pickup from a 24-inch table placement" width="280">](media/g1_real_eval/videos/mixed_pickup_lamp_01_pickup_success.mp4)<br>Picks up the lamp from a 24-inch table | [<img src="media/g1_real_eval/posters/mixed_pickup_lamp_02_base_grasp_success.jpg" alt="Lamp base grasp after contact at a 30-inch table placement" width="280">](media/g1_real_eval/videos/mixed_pickup_lamp_02_base_grasp_success.mp4)<br>Contacts the lamp, then grasps the base |
| Towel | [<img src="media/g1_real_eval/posters/mixed_pickup_towel_01_pickup_success.jpg" alt="Towel pickup from a 24-inch table placement" width="280">](media/g1_real_eval/videos/mixed_pickup_towel_01_pickup_success.mp4)<br>Picks up the towel | [<img src="media/g1_real_eval/posters/mixed_pickup_towel_02_regrasp_success.jpg" alt="Towel grasp, drop, and regrasp sequence" width="280">](media/g1_real_eval/videos/mixed_pickup_towel_02_regrasp_success.mp4)<br>Grasps, drops, and regrasps |
| Apple | [<img src="media/g1_real_eval/posters/mixed_pickup_apple_01_retry_success.jpg" alt="Apple pickup after a retry grasp" width="280">](media/g1_real_eval/videos/mixed_pickup_apple_01_retry_success.mp4)<br>Retries the grasp and succeeds | [<img src="media/g1_real_eval/posters/mixed_pickup_apple_02_adjust_success.jpg" alt="Apple pickup after gripper adjustment over the object" width="280">](media/g1_real_eval/videos/mixed_pickup_apple_02_adjust_success.mp4)<br>Adjusts over the apple before lifting |
| Shoe | [<img src="media/g1_real_eval/posters/mixed_pickup_shoe_01_many_attempts_success.jpg" alt="Shoe pickup after multiple grasp attempts" width="280">](media/g1_real_eval/videos/mixed_pickup_shoe_01_many_attempts_success.mp4)<br>Succeeds after several attempts | [<img src="media/g1_real_eval/posters/mixed_pickup_shoe_02_regrasp_success.jpg" alt="Shoe drop and regrasp success" width="280">](media/g1_real_eval/videos/mixed_pickup_shoe_02_regrasp_success.mp4)<br>Drops the shoe, then regrasps |
| Scoop | [<img src="media/g1_real_eval/posters/mixed_pickup_scoop_01_pickup_success.jpg" alt="Scoop pickup from a 27-inch table placement" width="280">](media/g1_real_eval/videos/mixed_pickup_scoop_01_pickup_success.mp4)<br>Picks up the scoop | [<img src="media/g1_real_eval/posters/mixed_pickup_scoop_02_noisy_success.jpg" alt="Scoop pickup with noisy approach motion" width="280">](media/g1_real_eval/videos/mixed_pickup_scoop_02_noisy_success.mp4)<br>Noisy approach, successful pickup |
| Flashlight | [<img src="media/g1_real_eval/posters/mixed_pickup_flashlight_01_smooth_success.jpg" alt="Smooth flashlight pickup" width="280">](media/g1_real_eval/videos/mixed_pickup_flashlight_01_smooth_success.mp4)<br>Smooth pickup | [<img src="media/g1_real_eval/posters/mixed_pickup_flashlight_02_pickup_success.jpg" alt="Flashlight pickup from a 27-inch table placement" width="280">](media/g1_real_eval/videos/mixed_pickup_flashlight_02_pickup_success.mp4)<br>Picks up the flashlight |
| Fruit | [<img src="media/g1_real_eval/posters/mixed_pickup_fruit_01_pickup_success.jpg" alt="Fruit pickup from a 24-inch table placement" width="280">](media/g1_real_eval/videos/mixed_pickup_fruit_01_pickup_success.mp4)<br>Picks up the fruit | [<img src="media/g1_real_eval/posters/mixed_pickup_fruit_02_regrasp_success.jpg" alt="Fruit grasp, drop, and regrasp success" width="280">](media/g1_real_eval/videos/mixed_pickup_fruit_02_regrasp_success.mp4)<br>Grasps, drops, and regrasps |
| Sock | [<img src="media/g1_real_eval/posters/mixed_pickup_sock_01_top_grasp_success.jpg" alt="Sock top-down grasp success" width="280">](media/g1_real_eval/videos/mixed_pickup_sock_01_top_grasp_success.mp4)<br>Top-down grasp succeeds | [<img src="media/g1_real_eval/posters/mixed_pickup_sock_02_regrasp_success.jpg" alt="Sock grasp, drop, and regrasp success" width="280">](media/g1_real_eval/videos/mixed_pickup_sock_02_regrasp_success.mp4)<br>Drops once, then regrasps |

### Soda Can From Small Table To Trash Can

| Task | Example 1 | Example 2 | Example 3 |
| --- | --- | --- | --- |
| Soda can to trash | [<img src="media/g1_real_eval/posters/soda_can_table_trash_01_smooth_success.jpg" alt="Soda can picked up and dropped into the trash can in one smooth sequence" width="240">](media/g1_real_eval/videos/soda_can_table_trash_01_smooth_success.mp4)<br>Smooth full sequence | [<img src="media/g1_real_eval/posters/soda_can_table_trash_02_second_grasp_step_success.jpg" alt="Soda can succeeds after a second grasp and second trigger step" width="240">](media/g1_real_eval/videos/soda_can_table_trash_02_second_grasp_step_success.mp4)<br>Second grasp and trigger step | [<img src="media/g1_real_eval/posters/soda_can_table_trash_03_clean_success.jpg" alt="Soda can picked up and dropped into the trash can without a visible retry" width="240">](media/g1_real_eval/videos/soda_can_table_trash_03_clean_success.mp4)<br>Single-attempt full sequence |

## Data Collection Experience

We followed the [GR00T Whole-Body Control data collection tutorial](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/data_collection.html) to collect G1 whole-body manipulation datasets.

Training a good policy depends on both collecting high-quality data and configuring training properly. Each demonstration should ideally complete the task on the first attempt, without corrective motions such as re-grasping after a failed grasp or redoing a missed stepping trigger. For the mixed object pickup policy, we trained for 60k iterations at batch size 256 on roughly 18k episodes covering about 50 different objects; for the soda-can-to-trash policy, we trained for 20k iterations at batch size 32 on roughly 150 episodes of this single task.

### Notes

1. The mixed pickup result uses one policy trained from a combined object dataset and evaluated across multiple table-top objects.
2. For the pickup task, we used an in-house UMI gripper design. Similar tasks may require users to build task-appropriate gripper designs for their own hardware setup.

## Data-Train-Eval Workflow

This benchmark used the public GR00T N1.7 and GR00T Whole-Body Control workflow:

1. Collect G1 demonstrations with SONIC teleoperation.
2. Fine-tune GR00T N1.7 with `UNITREE_G1_SONIC`.
3. Run the GR00T policy server and the SONIC robot-side controller.
4. Evaluate in closed loop on the real robot with video recording.

Fine-tuning used the SONIC embodiment tag:

```bash
bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /path/to/your/lerobot_dataset \
  --embodiment-tag UNITREE_G1_SONIC \
  --output-dir /path/to/output_checkpoint \
  --experiment-name g1-sonic-task
```

Closed-loop evaluation used a GR00T policy server:

```bash
python gr00t/eval/run_gr00t_server.py \
  --model-path /path/to/output_checkpoint/checkpoint-<step> \
  --embodiment-tag UNITREE_G1_SONIC \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5550
```

The robot-side controller follows the GR00T Whole-Body Control VLA inference workflow.

### Notes

- The data format should be LeRobot V2 already and ready to train the model.
- It would be beneficial to check open-loop robot joint trajectory matching against training set/validation set before deploying on real robot.
- As suggested in Data Collection Experience, data quality is preferred than quantity during post-training.

## References

- [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- [SONIC paper](https://nvlabs.github.io/GEAR-SONIC/static/pdf/sonic_paper.pdf)
- [GR00T Whole-Body Control documentation](https://nvlabs.github.io/GR00T-WholeBodyControl/)
- [GEAR-SONIC project page](https://nvlabs.github.io/GEAR-SONIC/)
- [GR00T data preparation guide](../../getting_started/data_preparation.md)
- [GR00T policy server/client guide](../../getting_started/policy.md)
