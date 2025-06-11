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

import atexit
import ctypes
import os

import tensorrt as trt
import torch


def torch_type(trt_type):
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.bool: torch.bool,
        trt.uint8: torch.uint8,
        trt.int64: torch.int64,
    }
    if trt_type in mapping:
        return mapping[trt_type]

    raise TypeError(
        f"Could not resolve TensorRT datatype to an equivalent numpy datatype. {trt_type}"
    )


class Engine(object):
    def __init__(self, file, plugins=[]):
        super().__init__()

        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")

        self.plugins = [ctypes.CDLL(plugin, ctypes.RTLD_GLOBAL) for plugin in plugins]
        self.file = file
        self.load(file)

        def destroy(self):
            del self.execution_context
            del self.handle

        atexit.register(destroy, self)
        self.print()

    def print(self):
        if int(os.getenv("LOCAL_RANK", -1)) not in [0, -1]:
            return

        print("============= TRT Engine Detail =============")
        print(f"Engine file: {self.file}")
        print(f"Inputs: {len(self.in_meta)}")
        for ib, item in enumerate(self.in_meta):
            tensor_name, shape, dtype = item[:3]
            print(f"   {ib}. {tensor_name}: {'x'.join(map(str, shape))} [{dtype}]")

        print(f"Outputs: {len(self.out_meta)}")
        for ib, item in enumerate(self.out_meta):
            tensor_name, shape, dtype = item[:3]
            print(f"   {ib}. {tensor_name}: {'x'.join(map(str, shape))} [{dtype}]")
        print("=============================================")

    def load(self, file):
        runtime = trt.Runtime(self.logger)

        with open(file, "rb") as f:
            self.handle = runtime.deserialize_cuda_engine(f.read())
            assert (
                self.handle is not None
            ), f"Failed to deserialize the cuda engine from file: {file}"

        self.execution_context = self.handle.create_execution_context()
        self.meta, self.in_meta, self.out_meta = [], [], []
        for tensor_name in self.handle:
            shape = self.handle.get_tensor_shape(tensor_name)
            dtype = torch_type(self.handle.get_tensor_dtype(tensor_name))
            if self.handle.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.in_meta.append([tensor_name, shape, dtype])
            else:
                self.out_meta.append([tensor_name, shape, dtype])

    def __call__(self, *args, **inputs):
        return self.forward(*args, **inputs)

    def set_runtime_tensor_shape(self, name, shape):
        self.execution_context.set_input_shape(name, shape)

    def forward(self, *args, **kwargs):
        return_list = kwargs.pop("return_list", False)
        reference_tensors = []
        stream = torch.cuda.current_stream()
        for iarg, x in enumerate(args):
            name, shape, dtype = self.in_meta[iarg]
            runtime_shape = self.execution_context.get_tensor_shape(name)
            assert isinstance(x, torch.Tensor), f"Unsupported tensor type: {type(x)}"
            assert runtime_shape == x.shape, f"Invalid input shape: {runtime_shape} != {x.shape}"
            assert (
                dtype == x.dtype
            ), f"Invalid tensor dtype, excepted dtype is {dtype}, but got {x.dtype}"
            assert x.is_cuda, f"Invalid tensor device, excepted device is cuda, but got {x.device}"
            x = x.cuda().contiguous()
            self.execution_context.set_tensor_address(name, x.data_ptr())
            reference_tensors.append(x)

        for name, shape, dtype in self.in_meta:
            if name not in kwargs:
                continue

            runtime_shape = self.execution_context.get_tensor_shape(name)
            x = kwargs[name]
            assert isinstance(x, torch.Tensor), f"Unsupported tensor[{name}] type: {type(x)}"
            assert (
                runtime_shape == x.shape
            ), f"Invalid input[{name}] shape: {x.shape}, but the expected shape is: {runtime_shape}"
            assert (
                dtype == x.dtype
            ), f"Invalid tensor[{name}] dtype, expected dtype is {dtype}, but got {x.dtype}"
            assert (
                x.is_cuda
            ), f"Invalid tensor[{name}] device, expected device is cuda, but got {x.device}"
            x = x.cuda().contiguous()
            self.execution_context.set_tensor_address(name, x.data_ptr())
            reference_tensors.append(x)

        for item in self.out_meta:
            name = item[0]
            runtime_shape = self.execution_context.get_tensor_shape(name)
            output_tensor = torch.zeros(
                *runtime_shape, dtype=item[2], device=reference_tensors[0].device
            )
            self.execution_context.set_tensor_address(name, output_tensor.data_ptr())
            reference_tensors.append(output_tensor)

        self.execution_context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        assert len(reference_tensors) == len(self.in_meta) + len(
            self.out_meta
        ), f"Invalid input tensors. The expected I/O tensors are {len(self.in_meta) + len(self.out_meta)}, but got {len(reference_tensors)}"

        if return_list:
            return [
                reference_tensors[len(self.in_meta) + i] for i, item in enumerate(self.out_meta)
            ]
        else:
            return {
                item[0]: reference_tensors[len(self.in_meta) + i]
                for i, item in enumerate(self.out_meta)
            }
