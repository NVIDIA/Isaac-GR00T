# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Test PolicyServer and PolicyClient ZMQ communication.

Uses a mock policy to avoid loading real model weights. The server is
started in a background thread and the client connects on localhost.
"""

import threading
import time

from gr00t.data.types import ModalityConfig
from gr00t.policy.server_client import MsgSerializer, PolicyClient, PolicyServer
import numpy as np
import pytest


class MockPolicy:
    """Minimal mock that satisfies BasePolicy interface without ABC enforcement."""

    def __init__(self):
        self.strict = False
        self._reset_count = 0

    def get_action(self, observation, options=None):
        # Echo back a dummy action dict derived from observation keys
        action = {"joint_pos": np.zeros(7, dtype=np.float32)}
        info = {"mock": True}
        return action, info

    def reset(self, options=None):
        self._reset_count += 1
        return {"reset_count": self._reset_count}

    def get_modality_config(self):
        return {
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=["joint_pos"],
            )
        }

    def check_observation(self, observation):
        pass

    def check_action(self, action):
        pass


def _find_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server_client():
    """Start a PolicyServer on a random port and yield a connected client."""
    port = _find_free_port()
    policy = MockPolicy()
    server = PolicyServer(policy, host="127.0.0.1", port=port)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(0.3)  # give ZMQ socket time to bind

    client = PolicyClient(host="127.0.0.1", port=port, timeout_ms=5000)
    yield client, server, policy

    # Cleanup: kill server, close sockets, terminate contexts
    try:
        client.kill_server()
    except Exception:
        server.running = False
    thread.join(timeout=2)
    try:
        client.socket.close(linger=0)
        client.context.term()
    except Exception:
        pass
    try:
        server.socket.close(linger=0)
        server.context.term()
    except Exception:
        pass


@pytest.mark.timeout(30)
class TestPolicyServerClient:
    """Test ZMQ roundtrip communication."""

    def test_ping(self, server_client):
        client, _, _ = server_client
        assert client.ping() is True

    def test_get_action_roundtrip(self, server_client):
        client, _, _ = server_client
        obs = {"state": {"joint_pos": np.zeros(7, dtype=np.float32)}}
        result = client.call_endpoint("get_action", {"observation": obs})
        action, info = result
        assert "joint_pos" in action
        np.testing.assert_array_equal(action["joint_pos"], np.zeros(7, dtype=np.float32))

    def test_reset(self, server_client):
        client, _, policy = server_client
        result = client.call_endpoint("reset", {"options": None})
        assert result["reset_count"] == 1
        result = client.call_endpoint("reset", {"options": None})
        assert result["reset_count"] == 2

    def test_get_modality_config(self, server_client):
        client, _, _ = server_client
        config = client.get_modality_config()
        assert "state" in config
        assert isinstance(config["state"], ModalityConfig)
        assert config["state"].modality_keys == ["joint_pos"]

    def test_kill_server(self):
        """Test that kill_server stops the server loop."""
        port = _find_free_port()
        policy = MockPolicy()
        server = PolicyServer(policy, host="127.0.0.1", port=port)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        time.sleep(0.3)

        client = PolicyClient(host="127.0.0.1", port=port, timeout_ms=5000)
        assert client.ping()
        client.kill_server()
        thread.join(timeout=3)
        assert not thread.is_alive(), "Server thread should have stopped"
        client.socket.close(linger=0)
        client.context.term()
        server.socket.close(linger=0)
        server.context.term()

    def test_unknown_endpoint_returns_error(self, server_client):
        client, _, _ = server_client
        with pytest.raises(RuntimeError, match="Unknown endpoint"):
            client.call_endpoint("nonexistent_endpoint", requires_input=False)


@pytest.mark.timeout(30)
class TestPolicyServerAuth:
    """Test API token authentication."""

    def test_valid_token(self):
        port = _find_free_port()
        token = "test-secret-123"
        server = PolicyServer(MockPolicy(), host="127.0.0.1", port=port, api_token=token)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        time.sleep(0.3)

        client = PolicyClient(host="127.0.0.1", port=port, timeout_ms=5000, api_token=token)
        assert client.ping()
        client.kill_server()
        thread.join(timeout=3)
        client.socket.close(linger=0)
        client.context.term()
        server.socket.close(linger=0)
        server.context.term()

    def test_invalid_token(self):
        port = _find_free_port()
        server = PolicyServer(MockPolicy(), host="127.0.0.1", port=port, api_token="correct")
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        time.sleep(0.3)

        client = PolicyClient(host="127.0.0.1", port=port, timeout_ms=5000, api_token="wrong")
        with pytest.raises(RuntimeError, match="Unauthorized"):
            client.call_endpoint("ping", requires_input=False)

        # Clean up
        valid_client = PolicyClient(
            host="127.0.0.1", port=port, timeout_ms=5000, api_token="correct"
        )
        valid_client.kill_server()
        thread.join(timeout=3)
        for c in [client, valid_client]:
            c.socket.close(linger=0)
            c.context.term()
        server.socket.close(linger=0)
        server.context.term()


class TestMsgSerializer:
    """Test msgpack serialization helpers."""

    def test_roundtrip_dict(self):
        data = {"key": "value", "number": 42}
        assert MsgSerializer.from_bytes(MsgSerializer.to_bytes(data)) == data

    def test_roundtrip_numpy(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = MsgSerializer.from_bytes(MsgSerializer.to_bytes(arr))
        np.testing.assert_array_equal(result, arr)

    def test_roundtrip_modality_config(self):
        config = ModalityConfig(delta_indices=[0, 1], modality_keys=["x", "y"])
        result = MsgSerializer.from_bytes(MsgSerializer.to_bytes(config))
        assert isinstance(result, ModalityConfig)
        assert result.modality_keys == ["x", "y"]

    def test_encode_rejects_object_dtype_ndarray(self):
        # Object-dtype ndarrays would be pickled by msgpack_numpy. The
        # pre-MR-307 implementation used np.save(..., allow_pickle=False)
        # which raised. Restore that safety contract on encode.
        arr = np.array([{"any": "object"}, [1, 2, 3]], dtype=object)
        with pytest.raises(TypeError, match="object-dtype"):
            MsgSerializer.to_bytes(arr)

    def test_decode_rejects_object_dtype_ndarray_bin_keys(self):
        # Forge a wire payload that claims object-dtype with msgpack-bin
        # (bytes) keys — the shape msgpack_numpy itself emits. We must
        # reject it before mnp.decode would call pickle.loads.
        import msgpack_numpy as mnp

        forged = mnp.packb(
            {b"nd": True, b"kind": b"O", b"type": "|O", b"shape": (1,), b"data": b""}
        )
        with pytest.raises(ValueError, match="object-dtype"):
            MsgSerializer.from_bytes(forged)

    def test_decode_rejects_object_dtype_ndarray_str_keys(self):
        # Defense-in-depth: enforce the allow_pickle=False contract even
        # for str-keyed forged payloads. msgpack_numpy 0.4.8 happens not
        # to pickle-decode this shape (it only matches bytes keys), but
        # MsgSerializer must not depend on mnp's internal key conventions.
        import msgpack_numpy as mnp

        forged = mnp.packb({"nd": True, "kind": "O", "type": "|O", "shape": (1,), "data": b""})
        with pytest.raises(ValueError, match="object-dtype"):
            MsgSerializer.from_bytes(forged)

    def test_decode_rejects_object_dtype_ndarray_int_nd(self):
        # Defense-in-depth: a forged payload that uses msgpack int 1 in
        # place of bool True for ``nd`` (different msgpack type codes,
        # both Python-truthy) must still be rejected. mnp 0.4.8's own
        # check is ``is True`` so this variant doesn't currently reach
        # pickle.loads inside mnp.decode either, but the MsgSerializer
        # contract is "any payload claiming nd + kind=O is refused"
        # regardless of mnp's identity-check semantics.
        import msgpack_numpy as mnp

        forged = mnp.packb({b"nd": 1, b"kind": b"O", b"type": "|O", b"shape": (1,), b"data": b""})
        with pytest.raises(ValueError, match="object-dtype"):
            MsgSerializer.from_bytes(forged)

    def test_decode_raises_on_marker_without_payload(self):
        # A truncated / malformed message that carries the marker but no
        # 'as_json' field used to fall through to a plain dict — silent
        # corruption. Now it must raise.
        import msgpack_numpy as mnp

        malformed = mnp.packb({"__ModalityConfig__": True})
        with pytest.raises(ValueError, match="Malformed ModalityConfig payload"):
            MsgSerializer.from_bytes(malformed)
