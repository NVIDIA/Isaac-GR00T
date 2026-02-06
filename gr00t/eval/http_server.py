#!/usr/bin/env python3
"""
GR00T HTTP Server Module

This module provides HTTP server functionality for GR00T model inference.
It exposes a REST API for easy integration with web applications and other services.

Dependencies:
    => Server: `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`
"""

import json
import logging
import time
import traceback
from typing import Any, Dict, Optional

import base64
import numpy as np
import json_numpy
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from gr00t.model.policy import Gr00tPolicy, Gr00tInpaintingPolicy

# Patch json to handle numpy arrays
json_numpy.patch()



def decode_numpy_from_base64(obj):
    """Decode numpy array from C++ base64 format."""
    if isinstance(obj, dict) and "__ndarray__" in obj:
        raw_bytes = base64.b64decode(obj["__ndarray__"])
        dtype = np.dtype(obj.get("dtype", "uint8"))
        shape = tuple(obj.get("shape", []))
        return np.frombuffer(raw_bytes, dtype=dtype).reshape(shape)
    return obj


class HTTPInferenceServer:
    def __init__(
        self, policy: Gr00tPolicy, port: int, host: str = "0.0.0.0", api_token: Optional[str] = None
    ):
        """
        A simple HTTP server for GR00T models; exposes `/act` to predict an action for a given observation.
            => Takes in observation dict with numpy arrays
            => Returns action dict with numpy arrays

        If the policy is a ``Gr00tInpaintingPolicy``, a ``POST /reset`` endpoint
        is also registered to clear the action buffer between episodes.
        """
        self.policy = policy
        self.port = port
        self.host = host
        self.api_token = api_token
        self.app = FastAPI(title="GR00T Inference Server", version="1.0.0")

        # Register endpoints
        self.app.post("/act")(self.predict_action)
        self.app.get("/health")(self.health_check)
        self.app.post("/reset")(self.reset_action_buffer)

    def predict_action(self, payload: Dict[str, Any]) -> JSONResponse:
        """Predict action from observation."""
        try:
            # Handle double-encoded payloads (for compatibility)
            if "encoded" in payload:
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Validate required fields
            if "observation" not in payload:
                raise HTTPException(
                    status_code=400, detail="Missing 'observation' field in payload"
                )

            obs = payload["observation"]
            
             # Decode image from base64 if present
            if "video.camera" in obs:
                obs["video.camera"] = decode_numpy_from_base64(obs["video.camera"])
            # print(obs["video.camera"])

            # Run inference
            start_time = time.time()
            action = self.policy.get_action(obs)
            end_time = time.time()
            print("time taken to get action: ", end_time - start_time)
            print(action)

            # Return action as JSON with numpy arrays
            action = {k: v.tolist() for k, v in action.items()}
            # print(action)
            return JSONResponse(content=action)

        except Exception as e:
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'observation': dict} where observation contains the required modalities.\n"
                "Example observation keys: video.ego_view, state.left_arm, state.right_arm, etc."
            )
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    def reset_action_buffer(self) -> Dict[str, str]:
        """Reset the inpainting action buffer (call between episodes).

        Only has an effect when the policy is a ``Gr00tInpaintingPolicy``.
        Safe to call on a regular ``Gr00tPolicy`` -- it simply returns OK.
        """
        if hasattr(self.policy, "reset_action_buffer"):
            self.policy.reset_action_buffer()
            print("Action buffer reset (inpainting state cleared)")
            return {"status": "reset", "inpainting": True}
        return {"status": "reset", "inpainting": False}

    def health_check(self) -> Dict[str, str]:
        """Health check endpoint."""
        is_inpainting = isinstance(self.policy, Gr00tInpaintingPolicy)
        return {"status": "healthy", "model": "GR00T", "inpainting": is_inpainting}

    def run(self) -> None:
        """Start the HTTP server."""
        is_inpainting = isinstance(self.policy, Gr00tInpaintingPolicy)
        print(f"Starting GR00T HTTP server on {self.host}:{self.port}")
        print(f"Inpainting mode: {is_inpainting}")
        print("Available endpoints:")
        print("  POST /act   - Get action prediction from observation")
        print("  POST /reset - Reset inpainting action buffer (between episodes)")
        print("  GET  /health - Health check")
        uvicorn.run(self.app, host=self.host, port=self.port)


def create_http_server(
    policy: Gr00tPolicy, port: int, host: str = "0.0.0.0", api_token: Optional[str] = None
) -> HTTPInferenceServer:
    """Factory function to create an HTTP inference server."""
    return HTTPInferenceServer(policy, port, host, api_token)
