#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025, Astribot Co., Ltd.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: Astribot Team
# -----------------------------------------------------------------------------

"""
File: gr00t_inference_bridge_server.py
Brief: ZeroMQ bridge server for gr00t inference.
       Receives observations from Astribot control client (Python 3.8),
       converts them to gr00t format, calls gr00t Policy directly for inference,
       and returns actions back to the client.

Overview
========
1. ZeroMQ REP socket server listening for inference requests
2. Receives observation dictionary from client
3. Converts observation format for gr00t (images, state, language)
4. Calls gr00t Policy.get_action() directly
5. Returns action dictionary to client

Usage
=====
Run this in Python 3.10 environment with gr00t installed:
    python gr00t_inference_bridge_server.py \
        --embodiment-tag NEW_EMBODIMENT \
        --model-path outputs/astribot/checkpoint-30000 \
        --device cuda:0 \
        --host 0.0.0.0 \
        --port 5555 \
        --strict
"""

import argparse
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict

try:
    import zmq
except ImportError:
    print("Error: pyzmq is required. Install with: pip install pyzmq")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Error: numpy is required. Install with: pip install numpy")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("Error: opencv-python is required. Install with: pip install opencv-python")
    sys.exit(1)

try:
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
except ImportError:
    print("Error: gr00t is required. Install gr00t package in Python 3.10 environment.")
    sys.exit(1)

from gr00t_bridge_serializer import MsgSerializer


@dataclass
class EndpointHandler:
    """Handler configuration for an endpoint."""
    handler: Callable
    requires_input: bool = True


class InferenceBridgeServer:
    """
    ZeroMQ server that bridges Astribot observations to gr00t inference.
    
    Receives observations in Astribot format, converts them to gr00t format,
    calls the gr00t Policy directly, and returns actions.
    """
    
    def __init__(
        self,
        embodiment_tag: EmbodimentTag,
        model_path: str,
        device: str = "cuda",
        strict: bool = True,
        host: str = "*",
        port: int = 5555,
        api_token: str = None,
    ):
        """
        Initialize the bridge server.
        
        Args:
            embodiment_tag: Embodiment tag for gr00t policy
            model_path: Path to the model checkpoint directory
            device: Device to run the model on (default: "cuda")
            strict: Whether to enforce strict input and output validation (default: True)
            host: Host to bind the bridge server to (use "*" for all interfaces)
            port: Port to bind the bridge server to
            api_token: Optional API token for authentication
        """
        # Check if model path exists
        if model_path.startswith("/") and not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist")
        
        # Initialize gr00t Policy directly
        print(f"Loading gr00t policy...")
        print(f"  Embodiment tag: {embodiment_tag}")
        print(f"  Model path: {model_path}")
        print(f"  Device: {device}")
        print(f"  Strict: {strict}")
        
        self.policy = Gr00tPolicy(
            embodiment_tag=embodiment_tag,
            model_path=model_path,
            device=device,
            strict=strict,
        )
        print("Gr00t policy loaded successfully!")
        
        # Initialize ZeroMQ socket
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        bind_address = f"tcp://{host}:{port}"
        self.socket.bind(bind_address)
        self.api_token = api_token
        
        # Register endpoints
        self._endpoints: Dict[str, EndpointHandler] = {}
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("get_action", self._handle_get_action, requires_input=True)
        
        print(f"Bridge server initialized. Listening on {bind_address}")
    
    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True):
        """
        Register an endpoint handler.
        
        Args:
            name: Endpoint name
            handler: Handler function
            requires_input: Whether the handler requires input data
        """
        self._endpoints[name] = EndpointHandler(handler, requires_input)
    
    def _validate_token(self, request: Dict) -> bool:
        """Validate API token if configured."""
        if self.api_token is None:
            return True
        return request.get("api_token") == self.api_token
    
    def _handle_ping(self) -> Dict:
        """Handle ping request for health check."""
        return {"status": "ok", "message": "Bridge server is running"}
    
    def _handle_get_action(self, observation: Dict, options: Dict = None) -> tuple:
        """
        Handle get_action request.
        
        Args:
            observation: Observation dictionary in Astribot format
            options: Optional parameters for get_action
        
        Returns:
            Tuple of (action_dict, info_dict) - matches PolicyServer behavior
        """
        # Convert observation to gr00t format
        gr00t_observation = self._convert_observation_to_gr00t(observation)
        
        # Call gr00t Policy directly (pass options if provided)
        action_dict, info = self.policy.get_action(gr00t_observation, options=options)
        
        # Return tuple directly (PolicyServer returns the result directly, not wrapped)
        # Note: msgpack will convert tuple to list during serialization, but that's fine
        return action_dict, info
    
    def _convert_observation_to_gr00t(self, observation: Dict) -> Dict:
        """
        Convert observation from Astribot format to gr00t format.
        
        Astribot format:
        - images: dict with keys 'head_rgbd', 'left_wrist_rgbd', 'right_wrist_rgbd', 'torso_rgbd'
          Each image is BGR format numpy array (H, W, 3)
        - state: dict with keys 'arm_left', 'arm_right', 'gripper_left', 'gripper_right', 'head', 'torso'
          Each value is a numpy array of appropriate shape
        - task_instruction: string
        
        gr00t format:
        - video: dict with keys 'head', 'hand_left', 'hand_right', 'torso'
          Each is (1, 1, H, W, 3) uint8 RGB array
          head: (720, 1280), hands: (360, 640), torso: (720, 1280)
        - state: dict with same keys, each is (1, 1, D) float32 array
        - language: dict with key 'annotation.human.task_description', value is [[string]]
        
        Args:
            observation: Observation in Astribot format
        
        Returns:
            Observation in gr00t format
        """
        gr00t_obs = {}
        
        # Convert images
        images = observation.get("images", {})
        gr00t_obs["video"] = {}
        
        # Head image
        if "head_rgbd" in images:
            head_img = images["head_rgbd"]
            # Convert BGR to RGB
            head_img_rgb = head_img[:, :, ::-1] if head_img.ndim == 3 else head_img
            # Resize to (720, 1280) if needed
            # if head_img_rgb.shape[:2] != (720, 1280):
            #     head_img_rgb = cv2.resize(head_img_rgb, (1280, 720), interpolation=cv2.INTER_LINEAR)
            # Add batch and temporal dimensions: (1, 1, H, W, 3)
            gr00t_obs["video"]["head"] = head_img_rgb.astype(np.uint8)[np.newaxis, np.newaxis, :, :, :]
        
        # Left hand image
        if "left_wrist_rgbd" in images:
            hand_left_img = images["left_wrist_rgbd"]
            hand_left_rgb = hand_left_img[:, :, ::-1] if hand_left_img.ndim == 3 else hand_left_img
            # if hand_left_rgb.shape[:2] != (360, 640):
            #     hand_left_rgb = cv2.resize(hand_left_rgb, (640, 360), interpolation=cv2.INTER_LINEAR)
            gr00t_obs["video"]["hand_left"] = hand_left_rgb.astype(np.uint8)[np.newaxis, np.newaxis, :, :, :]
        
        # Right hand image
        if "right_wrist_rgbd" in images:
            hand_right_img = images["right_wrist_rgbd"]
            hand_right_rgb = hand_right_img[:, :, ::-1] if hand_right_img.ndim == 3 else hand_right_img
            # if hand_right_rgb.shape[:2] != (360, 640):
            #     hand_right_rgb = cv2.resize(hand_right_rgb, (640, 360), interpolation=cv2.INTER_LINEAR)
            gr00t_obs["video"]["hand_right"] = hand_right_rgb.astype(np.uint8)[np.newaxis, np.newaxis, :, :, :]
        
        # Torso image
        if "torso_rgbd" in images:
            torso_img = images["torso_rgbd"]
            # Convert BGR to RGB
            torso_img_rgb = torso_img[:, :, ::-1] if torso_img.ndim == 3 else torso_img
            # Resize to (720, 1280) if needed (same as head)
            # if torso_img_rgb.shape[:2] != (720, 1280):
            #     torso_img_rgb = cv2.resize(torso_img_rgb, (1280, 720), interpolation=cv2.INTER_LINEAR)
            # Add batch and temporal dimensions: (1, 1, H, W, 3)
            gr00t_obs["video"]["torso"] = torso_img_rgb.astype(np.uint8)[np.newaxis, np.newaxis, :, :, :]
        
        # Convert state
        state = observation.get("state", {})
        gr00t_obs["state"] = {}
        
        for key in ["arm_left", "arm_right", "gripper_left", "gripper_right", "head", "torso"]:
            if key in state:
                state_value = state[key]
                # Ensure it's a numpy array and float32
                if not isinstance(state_value, np.ndarray):
                    state_value = np.array(state_value, dtype=np.float32)
                else:
                    state_value = state_value.astype(np.float32)
                
                # Add batch and temporal dimensions: (1, 1, D)
                if state_value.ndim == 1:
                    gr00t_obs["state"][key] = state_value[np.newaxis, np.newaxis, :]
                else:
                    # If already has dimensions, ensure it's (1, 1, D)
                    gr00t_obs["state"][key] = state_value.reshape(1, 1, -1)
        
        # Convert language/task instruction
        task_instruction = observation.get("task_instruction", "")
        gr00t_obs["language"] = {
            "annotation.human.task_description": [[task_instruction]]
        }
        
        return gr00t_obs
    
    def run(self):
        """Run the server loop."""
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Bridge server is ready and listening on {addr}")
        print("Waiting for requests...")
        
        while self.running:
            try:
                # Receive request
                message = self.socket.recv()
                request = MsgSerializer.from_bytes(message)
                
                # Validate token
                if not self._validate_token(request):
                    response = MsgSerializer.to_bytes({"error": "Unauthorized: Invalid API token"})
                    self.socket.send(response)
                    continue
                
                # Get endpoint
                endpoint = request.get("endpoint", "get_action")
                
                if endpoint not in self._endpoints:
                    error_msg = f"Unknown endpoint: {endpoint}"
                    print(f"Error: {error_msg}")
                    response = MsgSerializer.to_bytes({"error": error_msg})
                    self.socket.send(response)
                    continue
                
                # Call handler
                handler = self._endpoints[endpoint]
                if handler.requires_input:
                    data = request.get("data", {})
                    # For get_action endpoint, extract observation and options from data
                    if endpoint == "get_action":
                        observation = data.get("observation", data)
                        options = data.get("options", None)
                        result = handler.handler(observation=observation, options=options)
                    else:
                        result = handler.handler(**data)
                else:
                    result = handler.handler()
                
                # Send response
                response = MsgSerializer.to_bytes(result)
                self.socket.send(response)
                
            except KeyboardInterrupt:
                print("\nReceived interrupt signal. Shutting down...")
                self.running = False
                break
            except Exception as e:
                print(f"Error in server: {e}")
                traceback.print_exc()
                error_response = MsgSerializer.to_bytes({"error": str(e)})
                try:
                    self.socket.send(error_response)
                except:
                    pass  # Socket may be closed
        
        # Cleanup
        self.socket.close()
        self.context.term()
        print("Bridge server shut down.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Gr00t Inference Bridge Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python gr00t_inference_bridge_server.py \\
        --embodiment-tag NEW_EMBODIMENT \\
        --model-path outputs/astribot/checkpoint-30000 \\
        --device cuda:0 \\
        --host 0.0.0.0 \\
        --port 5555 \\
        --strict
        """
    )
    
    # Gr00t policy configs
    parser.add_argument(
        "--embodiment-tag",
        type=str,
        required=True,
        help="Embodiment tag (e.g., NEW_EMBODIMENT)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model checkpoint directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on (default: cuda)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enforce strict input and output validation"
    )
    
    # Server configs
    parser.add_argument(
        "--host",
        type=str,
        default="*",
        help="Host to bind the server to (default: *, all interfaces)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="Port to bind the server to (default: 5555)"
    )
    parser.add_argument(
        "--api-token",
        type=str,
        default=None,
        help="Optional API token for authentication"
    )
    
    args = parser.parse_args()
    
    # Convert embodiment tag string to EmbodimentTag enum
    # Handle both enum names (NEW_EMBODIMENT) and values (new_embodiment)
    try:
        tag_str = args.embodiment_tag.lower()  # Convert to lowercase
        # Try to find matching enum by value
        embodiment_tag = None
        for tag in EmbodimentTag:
            if tag.value == tag_str or tag.name.lower() == tag_str:
                embodiment_tag = tag
                break
        if embodiment_tag is None:
            raise ValueError(f"No matching enum found for '{args.embodiment_tag}'")
    except (ValueError, AttributeError) as e:
        print(f"Error: Invalid embodiment tag '{args.embodiment_tag}'")
        print(f"Valid tags: {[tag.value for tag in EmbodimentTag]}")
        sys.exit(1)
    
    print("=" * 60)
    print("Gr00t Inference Bridge Server")
    print("=" * 60)
    print(f"Embodiment tag: {embodiment_tag.value}")
    print(f"Model path: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Strict: {args.strict}")
    print(f"Server: {args.host}:{args.port}")
    print("=" * 60)
    
    try:
        server = InferenceBridgeServer(
            embodiment_tag=embodiment_tag,
            model_path=args.model_path,
            device=args.device,
            strict=args.strict,
            host=args.host,
            port=args.port,
            api_token=args.api_token,
        )
        server.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


