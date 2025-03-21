## Policy Deployment

> This tutorial requires user to have a trained model checkpoint and a physical So100 Lerobot robot to run the policy.

In this tutorial session, we will show example scripts and code snippets to deploy a trained policy. We will use the So100 Lerobot arm as an example.

![alt text](../media/so100_eval_demo.gif)

### 1. Load the policy

Run the following command to start the policy server.

```bash
python scripts/inference_service.py --server \
    --model_path <PATH_TO_YOUR_CHECKPOINT> \
    --embodiment_tag new_embodiment \
    --data_config so100 \
    --denoising_steps 4
```

 - Model path is the path to the checkpoint to use for the policy, user should provide the path to the checkpoint after finetuning
 - Denoising steps is the number of denoising steps to use for the policy, we noticed that having a denoising step of 4 is on par with 16
 - Embodiment tag is the tag of the embodiment to use for the policy, user should use new_embodiment when finetuning on a new robot
 - Data config is the data config to use for the policy. Users should use `so100`. If you want to use a different robot, implement your own `ModalityConfig` and `TransformConfig`

### 2. Client node

To deploy the finetuned model, you can use the `scripts/inference_policy.py` script. This script will start a policy server.

The client node can be implemented using the `from gr00t.eval.service import ExternalRobotInferenceClient` class. This class is a standalone client-server class that can be used to communicate with the policy server, with a `get_action()` endpoint as the only interface. 

```python
from gr00t.eval.service import ExternalRobotInferenceClient
from typing import Dict, Any

raw_obs_dict: Dict[str, Any] = {} # fill in the blanks

policy = ExternalRobotInferenceClient(host="localhost", port=5555)
raw_action_chunk: Dict[str, Any] = policy.get_action(raw_obs_dict)
```

User can just copy the class and implement their own client node in a separate isolated environment.

### Example with So100 Lerobot arm

We provide a sample client node implementation for the So100 Lerobot arm. Please refer to the example script `scripts/eval_gr00t_so100.py` for more details.


User can run the following command to start the client node.
```bash
python examples/eval_gr00t_so100.py \
 --use_policy --host <YOUR_POLICY_SERVER_HOST> \
 --port <YOUR_POLICY_SERVER_PORT> \
 --camera_index <YOUR_CAMERA_INDEX>
```

This will activate the robot, and call the `action = get_action(obs)` endpoint of the policy server to get the action, then execute the action on the robot.
