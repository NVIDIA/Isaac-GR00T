# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
echo "Important Notes:"
echo "1: The max batch of engine size is set to 8 in the reference case. "
echo "2: The MIN_LEN/OPT_LEN/MAX_LEN for LLM, DiT, VLLN-VLSelfAttention models is set to 80/283/300."
echo "If your inference batch size exceeds 8 or the MIN_LEN/OPT_LEN/MAX_LEN for LLM, DiT, VLLN-VLSelfAttention not fit your use case, please set it to your actual batch size and length variables."

export PATH=/usr/src/tensorrt/bin:$PATH

# Define length variables
MIN_LEN=80
OPT_LEN=283
MAX_LEN=300

if [ -e /usr/src/tensorrt/bin/trtexec ]; then
    echo "The file /usr/src/tensorrt/bin/trtexec exists."
else
    echo "The file /usr/src/tensorrt/bin/trtexec does not exist. Please install tensorrt"
fi

mkdir -p gr00t_engine

# VLLN-VLSelfAttention
echo "------------Building vlln_vl_self_attention Model--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers --onnx=gr00t_onnx/action_head/vlln_vl_self_attention.onnx --saveEngine=gr00t_engine/vlln_vl_self_attention.engine --minShapes=backbone_features:1x${MIN_LEN}x2048 --optShapes=backbone_features:1x${OPT_LEN}x2048 --maxShapes=backbone_features:8x${MAX_LEN}x2048 > gr00t_engine/vlln_vl_self_attention.log 2>&1

# DiT Model
echo "------------Building DiT Model--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers --onnx=gr00t_onnx/action_head/DiT.onnx --saveEngine=gr00t_engine/DiT.engine --minShapes=sa_embs:1x49x1536,vl_embs:1x${MIN_LEN}x2048,timesteps_tensor:1  --optShapes=sa_embs:1x49x1536,vl_embs:1x${OPT_LEN}x2048,timesteps_tensor:1  --maxShapes=sa_embs:8x49x1536,vl_embs:8x${MAX_LEN}x2048,timesteps_tensor:8 > gr00t_engine/build_DiT.log 2>&1

# State Encoder
echo "------------Building State Encoder--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers --onnx=gr00t_onnx/action_head/state_encoder.onnx --saveEngine=gr00t_engine/state_encoder.engine --minShapes=state:1x1x64,embodiment_id:1  --optShapes=state:1x1x64,embodiment_id:1 --maxShapes=state:8x1x64,embodiment_id:8 > gr00t_engine/build_state_encoder.log 2>&1

# Action Encoder
echo "------------Building Action Encoder--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers --onnx=gr00t_onnx/action_head/action_encoder.onnx --saveEngine=gr00t_engine/action_encoder.engine --minShapes=actions:1x16x32,timesteps_tensor:1,embodiment_id:1  --optShapes=actions:1x16x32,timesteps_tensor:1,embodiment_id:1  --maxShapes=actions:8x16x32,timesteps_tensor:8,embodiment_id:8 > gr00t_engine/build_action_encoder.log 2>&1

# Action Decoder
echo "------------Building Action Decoder--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers --onnx=gr00t_onnx/action_head/action_decoder.onnx --saveEngine=gr00t_engine/action_decoder.engine --minShapes=model_output:1x49x1024,embodiment_id:1  --optShapes=model_output:1x49x1024,embodiment_id:1  --maxShapes=model_output:8x49x1024,embodiment_id:8 > gr00t_engine/build_action_decoder.log 2>&1

# VLM-ViT
echo "------------Building VLM-ViT--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers --onnx=gr00t_onnx/eagle2/vit.onnx  --saveEngine=gr00t_engine/vit.engine --minShapes=pixel_values:1x3x224x224,position_ids:1x256 --optShapes=pixel_values:1x3x224x224,position_ids:1x256 --maxShapes=pixel_values:8x3x224x224,position_ids:8x256  > gr00t_engine/vit.log 2>&1

# VLM-LLM
echo "------------Building VLM-LLM--------------------"
trtexec --verbose --stronglyTyped --separateProfileRun --noDataTransfers --onnx=gr00t_onnx/eagle2/llm.onnx  --saveEngine=gr00t_engine/llm.engine --minShapes=input_ids:1x${MIN_LEN},vit_embeds:1x256x1152,attention_mask:1x${MIN_LEN} --optShapes=input_ids:1x${OPT_LEN},vit_embeds:1x256x1152,attention_mask:1x${OPT_LEN} --maxShapes=input_ids:8x${MAX_LEN},vit_embeds:8x256x1152,attention_mask:8x${MAX_LEN} > gr00t_engine/llm.log 2>&1
