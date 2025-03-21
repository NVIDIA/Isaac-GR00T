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

"""
Functions that work on nested structures of torch.Tensor or numpy array
"""


from collections.abc import Sequence

import numpy as np
import torch
import tree


def any_describe_str(x, shape_only=False):
    """
    Describe type, shape, device, data type (of np array/tensor)
    Very useful for debugging
    """
    t = type(x)
    tname = type(x).__name__
    if isinstance(x, np.ndarray):
        shape = list(x.shape)
        if x.size == 1:
            if shape_only:
                return f"np scalar: {x.item()} {shape}"
            else:
                return f"np scalar: {x.item()} {shape} {x.dtype}"
        else:
            if shape_only:
                return f"np: {shape}"
            else:
                return f"np: {shape} {x.dtype}"
    elif torch.is_tensor(x):
        shape = list(x.size())
        if x.numel() == 1:
            if shape_only:
                return f"torch scalar: {x.item()} {shape}"
            else:
                return f"torch scalar: {x.item()} {shape} {x.dtype} {x.device}"
        else:
            if shape_only:
                return f"torch: {shape}"
            else:
                return f"torch: {shape} {x.dtype} {x.device}"
    elif isinstance(x, str):
        return x
    elif isinstance(x, Sequence):
        return f"{tname}[{len(x)}]"
    elif x is None:
        return "None"
    elif np.issubdtype(t, np.number) or np.issubdtype(t, np.bool_):
        return f"{tname}: {x}"
    else:
        return f"{tname}"


def any_describe(x, msg="", *, shape_only=False):
    # from omlet.utils import yaml_dumps
    from pprint import pprint

    if isinstance(x, str) and msg != "":
        x, msg = msg, x

    if msg:
        msg += ": "
    print(msg, end="")
    pprint(tree.map_structure(lambda i: any_describe_str(i, shape_only=shape_only), x))
