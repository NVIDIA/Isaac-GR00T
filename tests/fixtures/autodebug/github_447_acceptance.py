# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Solution-independent CPU acceptance check for held-out GitHub issue #447."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

source = Path("gr00t/policy/server_client.py")

msgpack = types.ModuleType("msgpack")
msgpack.packb = lambda *_args, **_kwargs: b""  # type: ignore[attr-defined]
msgpack.unpackb = lambda *_args, **_kwargs: {}  # type: ignore[attr-defined]
sys.modules["msgpack"] = msgpack

numpy = types.ModuleType("numpy")
numpy.ndarray = type("ndarray", (), {})  # type: ignore[attr-defined]
numpy.load = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
numpy.save = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
sys.modules["numpy"] = numpy

bound_addresses: list[str] = []


class FakeSocket:
    def bind(self, address: str) -> None:
        bound_addresses.append(address)


class FakeContext:
    def socket(self, socket_type: object) -> FakeSocket:
        del socket_type
        return FakeSocket()


zmq = types.ModuleType("zmq")
zmq.Context = FakeContext  # type: ignore[attr-defined]
zmq.REP = object()  # type: ignore[attr-defined]
zmq.REQ = object()  # type: ignore[attr-defined]
zmq.LAST_ENDPOINT = object()  # type: ignore[attr-defined]
zmq.error = types.SimpleNamespace(ZMQError=RuntimeError)  # type: ignore[attr-defined]
sys.modules["zmq"] = zmq

gr00t = types.ModuleType("gr00t")
gr00t.__path__ = []  # type: ignore[attr-defined]
policy_package = types.ModuleType("gr00t.policy")
policy_package.__path__ = []  # type: ignore[attr-defined]
data_package = types.ModuleType("gr00t.data")
data_package.__path__ = []  # type: ignore[attr-defined]
types_module = types.ModuleType("gr00t.data.types")
types_module.ModalityConfig = type("ModalityConfig", (), {})  # type: ignore[attr-defined]
utils_module = types.ModuleType("gr00t.data.utils")
utils_module.to_json_serializable = lambda value: value  # type: ignore[attr-defined]
policy_module = types.ModuleType("gr00t.policy.policy")
policy_module.BasePolicy = type("BasePolicy", (), {})  # type: ignore[attr-defined]
sys.modules.update(
    {
        "gr00t": gr00t,
        "gr00t.policy": policy_package,
        "gr00t.data": data_package,
        "gr00t.data.types": types_module,
        "gr00t.data.utils": utils_module,
        "gr00t.policy.policy": policy_module,
    }
)

spec = importlib.util.spec_from_file_location("gr00t.policy.server_client", source)
assert spec is not None
assert spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.PolicyServer.run = lambda _self: None
constructed_servers: list[object] = []
original_init = module.PolicyServer.__init__


def recording_init(self: object, *args: object, **kwargs: object) -> None:
    original_init(self, *args, **kwargs)
    constructed_servers.append(self)


module.PolicyServer.__init__ = recording_init


class DummyPolicy:
    def get_action(self) -> None:
        return None

    def reset(self) -> None:
        return None


token = object()
module.PolicyServer.start_server(DummyPolicy(), port=5555, host="127.0.0.1", api_token=token)
assert bound_addresses[-1] == "tcp://127.0.0.1:5555"
assert constructed_servers[-1].api_token is token
module.PolicyServer.start_server(DummyPolicy(), port=5555)
assert bound_addresses[-1] == "tcp://*:5555"
