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
echo "2: The MIN_LEN/OPT_LEN/MAX_LEN for LLM, DiT, VLLN-VLSelfAttention models is set to 80/296/300."
echo "If your inference batch size exceeds 8 or the MIN_LEN/OPT_LEN/MAX_LEN for LLM, DiT, VLLN-VLSelfAttention not fit your use case, please set it to your actual batch size and length variables."

export PATH=/usr/src/tensorrt/bin:$PATH

VIDEO_VIEWS=${VIDEO_VIEWS:-1}   # Options: 1 (default), 2
# Define length variables
# If only one video view, use these lengths; else (for two video views) use different lengths
if [ "$VIDEO_VIEWS" = "2" ]; then
    # two video views
    MIN_LEN=80
    OPT_LEN=568
    MAX_LEN=600
else   # one video view (default)
    MIN_LEN=80
    OPT_LEN=296
    MAX_LEN=300
fi

# Define precision settings (can be overridden via environment variables)
VIT_DTYPE=${VIT_DTYPE:-fp8}     # Options: fp16, fp8
LLM_DTYPE=${LLM_DTYPE:-nvfp4}   # Options: fp16, nvfp4, nvfp4_full, fp8
DIT_DTYPE=${DIT_DTYPE:-fp8}     # Options: fp16, fp8

# Define max batch size (default 8, will be overridden for nvfp4 LLM variants)
MAX_BATCH=${MAX_BATCH:-8}

echo "Building TensorRT engines with the following precisions:"
echo "  ViT: ${VIT_DTYPE}"
echo "  LLM: ${LLM_DTYPE}"
echo "  DiT: ${DIT_DTYPE}"
echo "  Video Views: ${VIDEO_VIEWS}"
echo "  MAX_BATCH: ${MAX_BATCH}"
echo "  MIN_LEN: ${MIN_LEN}"
echo "  OPT_LEN: ${OPT_LEN}"
echo "  MAX_LEN: ${MAX_LEN}"

if [ ! -e /usr/src/tensorrt/bin/trtexec ]; then
    echo "The file /usr/src/tensorrt/bin/trtexec does not exist. Please install tensorrt"
    exit 1
fi

mkdir -p gr00t_engine

# VLLN-VLSelfAttention
echo "------------Building vlln_vl_self_attention Model--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers --onnx=gr00t_onnx/action_head/vlln_vl_self_attention.onnx --saveEngine=gr00t_engine/vlln_vl_self_attention.engine --minShapes=backbone_features:1x${MIN_LEN}x2048 --optShapes=backbone_features:1x${OPT_LEN}x2048 --maxShapes=backbone_features:${MAX_BATCH}x${MAX_LEN}x2048 > gr00t_engine/vlln_vl_self_attention.log 2>&1

# DiT Model
echo "------------Building DiT Model (${DIT_DTYPE})--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers --onnx=gr00t_onnx/action_head/DiT_${DIT_DTYPE}.onnx --saveEngine=gr00t_engine/DiT_${DIT_DTYPE}.engine --minShapes=sa_embs:1x49x1536,vl_embs:1x${MIN_LEN}x2048,timesteps_tensor:1  --optShapes=sa_embs:1x49x1536,vl_embs:1x${OPT_LEN}x2048,timesteps_tensor:1  --maxShapes=sa_embs:${MAX_BATCH}x49x1536,vl_embs:${MAX_BATCH}x${MAX_LEN}x2048,timesteps_tensor:${MAX_BATCH} > gr00t_engine/DiT_${DIT_DTYPE}.log 2>&1

# State Encoder
echo "------------Building State Encoder--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers --onnx=gr00t_onnx/action_head/state_encoder.onnx --saveEngine=gr00t_engine/state_encoder.engine --minShapes=state:1x1x64,embodiment_id:1  --optShapes=state:1x1x64,embodiment_id:1 --maxShapes=state:${MAX_BATCH}x1x64,embodiment_id:${MAX_BATCH} > gr00t_engine/state_encoder.log 2>&1

# Action Encoder
echo "------------Building Action Encoder--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers --onnx=gr00t_onnx/action_head/action_encoder.onnx --saveEngine=gr00t_engine/action_encoder.engine --minShapes=actions:1x16x32,timesteps_tensor:1,embodiment_id:1  --optShapes=actions:1x16x32,timesteps_tensor:1,embodiment_id:1  --maxShapes=actions:${MAX_BATCH}x16x32,timesteps_tensor:${MAX_BATCH},embodiment_id:${MAX_BATCH} > gr00t_engine/action_encoder.log 2>&1

# Action Decoder
echo "------------Building Action Decoder--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers --onnx=gr00t_onnx/action_head/action_decoder.onnx --saveEngine=gr00t_engine/action_decoder.engine --minShapes=model_output:1x49x1024,embodiment_id:1  --optShapes=model_output:1x49x1024,embodiment_id:1  --maxShapes=model_output:${MAX_BATCH}x49x1024,embodiment_id:${MAX_BATCH} > gr00t_engine/action_decoder.log 2>&1

# VLM-ViT
echo "------------Building VLM-ViT (${VIT_DTYPE})--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers --onnx=gr00t_onnx/eagle2/vit_${VIT_DTYPE}.onnx  --saveEngine=gr00t_engine/vit_${VIT_DTYPE}.engine --minShapes=pixel_values:1x3x224x224,position_ids:1x256 --optShapes=pixel_values:${VIDEO_VIEWS}x3x224x224,position_ids:${VIDEO_VIEWS}x256 --maxShapes=pixel_values:${MAX_BATCH}x3x224x224,position_ids:${MAX_BATCH}x256  > gr00t_engine/vit_${VIT_DTYPE}.log 2>&1

# VLM-LLM
echo "------------Building VLM-LLM (${LLM_DTYPE})--------------------"
# Validate LLM_DTYPE
if [[ ! "$LLM_DTYPE" =~ ^(fp16|nvfp4|nvfp4_full|fp8)$ ]]; then
    echo "Error: LLM_DTYPE must be 'fp16', 'nvfp4', 'nvfp4_full', or 'fp8', got '${LLM_DTYPE}'"
    exit 1
fi

# Override max batch size for nvfp4 variants (require fixed shapes for Myelin)
if [[ "$LLM_DTYPE" =~ ^nvfp4 ]]; then
    LLM_MAX_BATCH=1  # Fixed shapes (required for Myelin)
else
    LLM_MAX_BATCH=${MAX_BATCH}  # Use the default MAX_BATCH
fi

trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers \
    --onnx=gr00t_onnx/eagle2/llm_${LLM_DTYPE}.onnx \
    --saveEngine=gr00t_engine/llm_${LLM_DTYPE}.engine \
    --minShapes=inputs_embeds:1x${MIN_LEN}x2048,attention_mask:1x${MIN_LEN} \
    --optShapes=inputs_embeds:1x${OPT_LEN}x2048,attention_mask:1x${OPT_LEN} \
    --maxShapes=inputs_embeds:${LLM_MAX_BATCH}x${MAX_LEN}x2048,attention_mask:${LLM_MAX_BATCH}x${MAX_LEN} \
    > gr00t_engine/llm_${LLM_DTYPE}.log 2>&1

echo ""
echo "============================================================"
echo "TensorRT Engine Build Complete!"
echo "============================================================"
echo "Built engines with the following configurations:"
echo "  ViT: ${VIT_DTYPE}"
echo "  LLM: ${LLM_DTYPE}"
echo "  DiT: ${DIT_DTYPE}"
echo ""
echo "Engines saved in: gr00t_engine/"
echo "Build logs saved in: gr00t_engine/*.log"
echo "============================================================"
