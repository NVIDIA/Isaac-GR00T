"""
A drop-in client/server transport layer using Python pickle for serialization over ZeroMQ.

This module mirrors the BaseInferenceServer/BaseInferenceClient pattern but uses
pickle.dumps/loads instead of torch.save/load so that clients need not depend on PyTorch.
"""

import pickle
import zmq
from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass
class PickleEndpointHandler:
    handler: Callable
    requires_input: bool = True


class PickleInferenceServer:
    """
    An inference server that uses pickle for serialization over ZeroMQ.
    Can register multiple endpoints, each handling a different request type.
    """

    def __init__(self, host: str = "*", port: int = 5555):
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: Dict[str, PickleEndpointHandler] = {}

        # Default control endpoints
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True):
        """
        Register a new endpoint for handling requests.

        Args:
            name: Name of the endpoint.
            handler: Callable to handle the request data.
            requires_input: Whether the handler expects input data.
        """
        self._endpoints[name] = PickleEndpointHandler(handler, requires_input)

    def _kill_server(self):
        """Stop the server loop."""
        self.running = False

    def _handle_ping(self) -> dict:
        """Respond to a ping to indicate the server is alive."""
        return {"status": "ok", "message": "Server is running"}

    def run(self):
        """Run the server loop, listening for incoming requests."""
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"PickleInferenceServer listening on {addr}")
        while self.running:
            try:
                raw = self.socket.recv()
                request = pickle.loads(raw)

                endpoint = request.get("endpoint", "get_action")
                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")

                handler = self._endpoints[endpoint]
                result = (
                    handler.handler(request.get("data", {}))
                    if handler.requires_input
                    else handler.handler()
                )
                self.socket.send(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
            except Exception as e:
                print(f"Error in PickleInferenceServer: {e}")
                import traceback

                print(traceback.format_exc())
                self.socket.send(pickle.dumps({"error": str(e)}, protocol=pickle.HIGHEST_PROTOCOL))


class PickleInferenceClient:
    """
    Client for communicating with a PickleInferenceServer.
    """

    def __init__(self, host: str = "localhost", port: int = 5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")

    def call_endpoint(
        self, endpoint: str, data: Dict[str, Any] = None, requires_input: bool = True
    ) -> Dict[str, Any]:
        """
        Call a registered endpoint on the server.

        Args:
            endpoint: Name of the endpoint to call.
            data: Payload dict to send (if any).
            requires_input: Whether the endpoint expects input data.

        Returns:
            The response dict from the server.
        """
        request: Dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data

        self.socket.send(pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL))
        raw = self.socket.recv()
        response = pickle.loads(raw)

        if "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def ping(self) -> bool:
        """Check if the server is alive."""
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            return False

    def kill_server(self):
        """Stop the remote server."""
        self.call_endpoint("kill", requires_input=False)

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Get an action from the server by forwarding observations."""
        return self.call_endpoint("get_action", observations)


class ExternalPickleInferenceClient(PickleInferenceClient):
    """
    Client for communicating with a PickleInferenceServer externally.
    """

    def __init__(self, host: str = "localhost", port: int = 5555):
        super().__init__(host, port)

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Get an action from the server by forwarding observations."""
        return self.call_endpoint("get_action", observations)
