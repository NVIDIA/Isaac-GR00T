# GR00T Whole-Body Control Real-World Benchmark

This example reports a real-world Unitree G1 evaluation using **Isaac GR00T N1.7** together with **GR00T Whole-Body Control / GEAR-SONIC**. The benchmark focuses on everyday mobile-manipulation tasks that require walking, table approach, grasping, foot placement, and whole-body pickup motions.

## Summary

GR00T N1.7 with SONIC can execute closed-loop whole-body skills on a real humanoid robot. The main targeted task is **walk to a table and pick up an object**. A single mixed-object policy was trained across datasets that record G1 walking toward a table and picking up different table-top objects. A second task-specific policy was trained for walking to a small table, picking up a soda can, stepping on a trash-can trigger, and dropping the can inside.

For more GR00T Whole-Body Control task examples, see the [GEAR-SONIC project page](https://nvlabs.github.io/GEAR-SONIC/).

## Evaluation Results

| Task Family | Policy Setup | Task Specification | Logged Trials | Successes | Success Rate |
| --- | --- | --- | ---: | ---: | ---: |
| Walk to table and pick up object | Single mixed-object pickup policy | Walk to the table and grasp the object and lift it. Tested across 8 different objects. Autonomous retries count as success. | 36 | 34 | 94.4% |
| Soda can from small table to trash can | Task-specific soda-can policy | Walk toward a small table, pick up a soda can, rotate toward the trash can, step on the trigger, and drop the can inside. | 12 | 8 | 66.7% |

The failures happened in the tests are mostly unsuccessful grasps. For example, the gripper missed the correct grasp poses and knocked down the soda can and cannot grasp it anymore; the grasping point is not stable and the shoe falls off and cannot find a valid grasp point again.

## Demo Videos

The MP4 samples below are attached as video files in this repository. Click any preview image to open the corresponding MP4. For more task demos beyond these two quantified results, see the [GEAR-SONIC project page](https://nvlabs.github.io/GEAR-SONIC/).

### Walk To Table And Pick Up Objects

Two examples are included for each evaluated object.

| Object | Example 1 | Example 2 |
| --- | --- | --- |
| Lamp | [<img src="media/g1_real_eval/posters/mixed_pickup_lamp_01_clean_success.jpg" alt="Lamp clean pickup" width="280">](media/g1_real_eval/videos/mixed_pickup_lamp_01_clean_success.mp4)<br>Clean pickup | [<img src="media/g1_real_eval/posters/mixed_pickup_lamp_02_retry_success.jpg" alt="Lamp retry pickup" width="280">](media/g1_real_eval/videos/mixed_pickup_lamp_02_retry_success.mp4)<br>Retry pickup |
| Towel | [<img src="media/g1_real_eval/posters/mixed_pickup_towel_01_clean_success.jpg" alt="Towel clean pickup" width="280">](media/g1_real_eval/videos/mixed_pickup_towel_01_clean_success.mp4)<br>Clean pickup | [<img src="media/g1_real_eval/posters/mixed_pickup_towel_02_regrasp_success.jpg" alt="Towel regrasp pickup" width="280">](media/g1_real_eval/videos/mixed_pickup_towel_02_regrasp_success.mp4)<br>Regrasp pickup |
| Apple | [<img src="media/g1_real_eval/posters/mixed_pickup_apple_01_clean_success.jpg" alt="Apple clean pickup" width="280">](media/g1_real_eval/videos/mixed_pickup_apple_01_clean_success.mp4)<br>Clean pickup | [<img src="media/g1_real_eval/posters/mixed_pickup_apple_02_clean_success.jpg" alt="Apple clean pickup 2" width="280">](media/g1_real_eval/videos/mixed_pickup_apple_02_clean_success.mp4)<br>Clean pickup |
| Shoe | [<img src="media/g1_real_eval/posters/mixed_pickup_shoe_01_clean_success.jpg" alt="Shoe clean pickup" width="280">](media/g1_real_eval/videos/mixed_pickup_shoe_01_clean_success.mp4)<br>Clean pickup | [<img src="media/g1_real_eval/posters/mixed_pickup_shoe_02_clean_success.jpg" alt="Shoe clean pickup 2" width="280">](media/g1_real_eval/videos/mixed_pickup_shoe_02_clean_success.mp4)<br>Clean pickup |
| Scoop | [<img src="media/g1_real_eval/posters/mixed_pickup_scoop_01_recover_success.jpg" alt="Scoop recovery pickup" width="280">](media/g1_real_eval/videos/mixed_pickup_scoop_01_recover_success.mp4)<br>Recovery pickup | [<img src="media/g1_real_eval/posters/mixed_pickup_scoop_02_third_grasp_success.jpg" alt="Scoop third-grasp pickup" width="280">](media/g1_real_eval/videos/mixed_pickup_scoop_02_third_grasp_success.mp4)<br>Third-grasp pickup |
| Eraser | [<img src="media/g1_real_eval/posters/mixed_pickup_eraser_01_clean_success.jpg" alt="Eraser clean pickup" width="280">](media/g1_real_eval/videos/mixed_pickup_eraser_01_clean_success.mp4)<br>Clean pickup | [<img src="media/g1_real_eval/posters/mixed_pickup_eraser_02_alignment_success.jpg" alt="Eraser alignment pickup" width="280">](media/g1_real_eval/videos/mixed_pickup_eraser_02_alignment_success.mp4)<br>Alignment pickup |
| Sock | [<img src="media/g1_real_eval/posters/mixed_pickup_sock_01_direct_success.jpg" alt="Sock direct pickup" width="280">](media/g1_real_eval/videos/mixed_pickup_sock_01_direct_success.mp4)<br>Direct pickup | [<img src="media/g1_real_eval/posters/mixed_pickup_sock_02_recovery_success.jpg" alt="Sock recovery pickup" width="280">](media/g1_real_eval/videos/mixed_pickup_sock_02_recovery_success.mp4)<br>Recovery pickup |
| Dragonfruit | [<img src="media/g1_real_eval/posters/mixed_pickup_dragonfruit_01_clean_success.jpg" alt="Dragonfruit clean pickup" width="280">](media/g1_real_eval/videos/mixed_pickup_dragonfruit_01_clean_success.mp4)<br>Clean pickup | [<img src="media/g1_real_eval/posters/mixed_pickup_dragonfruit_02_recovery_success.jpg" alt="Dragonfruit recovery pickup" width="280">](media/g1_real_eval/videos/mixed_pickup_dragonfruit_02_recovery_success.mp4)<br>Recovery pickup |

### Soda Can From Small Table To Trash Can

| Task | Example 1 | Example 2 | Example 3 |
| --- | --- | --- | --- |
| Soda can to trash | [<img src="media/g1_real_eval/posters/soda_can_table_trash_01_smooth_success.jpg" alt="Soda can smooth success" width="240">](media/g1_real_eval/videos/soda_can_table_trash_01_smooth_success.mp4)<br>Smooth success | [<img src="media/g1_real_eval/posters/soda_can_table_trash_02_second_grasp_step_success.jpg" alt="Soda can second-grasp success" width="240">](media/g1_real_eval/videos/soda_can_table_trash_02_second_grasp_step_success.mp4)<br>Second-grasp success | [<img src="media/g1_real_eval/posters/soda_can_table_trash_03_clean_success.jpg" alt="Soda can clean success" width="240">](media/g1_real_eval/videos/soda_can_table_trash_03_clean_success.mp4)<br>Clean success |

## Data Collection Experience

We followed the [GR00T Whole-Body Control data collection tutorial](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/data_collection.html) to collect G1 whole-body manipulation datasets.

Training a good policy depends on both collecting high-quality data and configuring training properly. Each demonstration should ideally complete the task on the first attempt, without corrective motions such as re-grasping after a failed grasp or redoing a missed stepping trigger. For the mixed object pickup policy, we trained for 60k iterations at batch size 256 on roughly 18k episodes covering over 50 different objects; for the soda-can-to-trash policy, we trained for 20k iterations at batch size 32 on roughly 150 episodes of this single task.

### Notes

1. The mixed pickup result uses one policy trained from a combined object dataset and evaluated across multiple table-top objects.
2. For the pickup task, we used an in-house UMI gripper design. Similar tasks may require users to build task-appropriate gripper designs for their own hardware setup.

## Data-Train-Eval Workflow

This benchmark used the public GR00T N1.7 and GR00T Whole-Body Control workflow:

1. Collect G1 demonstrations with SONIC teleoperation.
2. Convert demonstrations to GR00T LeRobot format.
3. Fine-tune GR00T N1.7 with `UNITREE_G1_SONIC`.
4. Run the GR00T policy server and the SONIC robot-side controller.
5. Evaluate in closed loop on the real robot with video recording.

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

## References

- [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- [GR00T Whole-Body Control documentation](https://nvlabs.github.io/GR00T-WholeBodyControl/)
- [GEAR-SONIC project page](https://nvlabs.github.io/GEAR-SONIC/)
- [GR00T data preparation guide](../../getting_started/data_preparation.md)
- [GR00T policy server/client guide](../../getting_started/policy.md)
