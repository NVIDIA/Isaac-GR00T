"""
Standalone Piper Client - Can run in any environment
Only requires installation: pip install msgpack numpy pyzmq
"""
import io
from typing import Any
import msgpack
import numpy as np
import zmq


class MsgSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj):
        if not isinstance(obj, dict):
            return obj
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


class SimplePolicyClient:
    def __init__(self, host: str = "localhost", port: int = 5555, timeout_ms: int = 15000):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._init_socket()

    def _init_socket(self):
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call_endpoint(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> Any:
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        
        self.socket.send(MsgSerializer.to_bytes(request))
        message = self.socket.recv()
        response = MsgSerializer.from_bytes(message)

        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def get_action(self, observation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.call_endpoint("get_action", {"observation": observation, "options": None})
        return tuple(response)

    def reset(self) -> dict[str, Any]:
        return self.call_endpoint("reset", {"options": None})

    def __del__(self):
        self.socket.close()
        self.context.term()


if __name__ == "__main__":
    # Create client
    client = SimplePolicyClient(host="127.0.0.1", port=5555)
    
    # Reset policy
    client.reset()
    
    # Execute inference loop
    for step in range(10):
        observation = {
            "video": {
                "hand_camera": np.random.randint(0, 256, (1, 1, 480, 640, 3), dtype=np.uint8),
                "third_camera": np.random.randint(0, 256, (1, 1, 480, 640, 3), dtype=np.uint8),
            },
            "state": {
                "joint_states": np.random.rand(1, 1, 6).astype(np.float32),
                "gripper_distance": np.random.rand(1, 1, 1).astype(np.float32),
            },
            "language": {
                "annotation.human.action.task_description": [["pick and place task"]],
            }
        }
        
        action, info = client.get_action(observation)
        print(f"Step {step}: action={action}")

