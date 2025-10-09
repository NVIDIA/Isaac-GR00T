# Real-Time-Chunking (RTC) with Gr00t

Within this package, the Real-Time Action Chunking technique proposed by [Physical Intelligence](https://www.physicalintelligence.company/research/real_time_chunking) is implemented. In the following, details about the implementation are given.

## Architecture
As the updated version of the denoising algorithm of Physical Intelligence is applied to the denoising iteration of a flow model, the denoising iteration of the [flow_matching_action_head of Gr00t](https://github.com/NVIDIA/Isaac-GR00T/blob/4ea96a16b15cfdbbd787b6b4f519a12687281330/gr00t/model/action_head/flow_matching_action_head.py#L353) needs to be changed.
This is done with `RTCFlowmatchingActionHead` in `rtc_flow_matching_action_head.py` which inherits from the original `FlowmatchingActionHead` and adds a `get_realtime_action()` function which uses the denoising algorithm of Physical Intelligence.

In order to use the new `RTCFlowmatchingActionHead` we also need to extend `RTCGr00t1_5` in `rtc_gr00t.py` and use the `RTCFlowmatchingActionHead`. 
Analogously, the `RTCGr00tPolicy` in `rtc_policy.py` extends the the original `Gr00tPolicy` and uses `RTCGr00t1_5`.

The full realtime-action-chunking procedure is managed in the `rtc_controller.py` using the same syntax as the paper of Physical Intelligence.
The algorithm implements a buffer with predicted actions, from which actions can be requested through the step-method. 
Also, a thread with the inference-loop is started, which executes an inference-call every time a certain amount of actions was consumed.
The whole procedure is synchronized with a lock and a condition variable.

## Usage

### 1. Start the RTC inference server
The RTC inference server must be started first with docker-compose:
```bash
docker compose --profile inference up --build   
```
The following paramters are taken from environment variables that can be defined in the `.env` file:
* `INFERENCE_PORT` *(port of the inference server)*
* `EMBODIMENT_TAG` *(embodiment key)*
* `DATA_CONFIG` *(data config key)*
* `MODEL_HOST_PATH` *(path to the model checkpoint to be used)*
* `HF_MODEL_PATH` *(path to the model on huggingface, if this path is provided, the model given by `MODEL_HOST_PATH` is ignored)*

Custom data configs can be added in `custom_data_config.py` and registered in `extended_data_config.py`.

### 2. Test the inference server
To test the inference server you can use the `tests/test_inference.py`. Make sure to that the fake observations in the test match the data config.

**Important:** The container must be running before you can execute commands in it!


```bash
docker compose --profile inference exec inference bash -c "cd /workspace/Isaac-GR00T && /workspace/venv/bin/python tests/test_inference.py"
```

## Visualization
A visualization of the inpainting-like behavior of RTC can be found and evaluated with [evaluate_rtc.ipynb](evaluation/evaluate_rtc.ipynb). 

## Evaluation
More details about the RTC algorithm and an evaluation of our implementation can be found in `tng_rtc/docs/rtc_evaluation.md`
