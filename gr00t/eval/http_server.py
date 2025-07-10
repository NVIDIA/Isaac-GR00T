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
import traceback
from typing import Any, Dict, Optional

import json_numpy
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from gr00t.model.policy import Gr00tPolicy

# Patch json to handle numpy arrays
json_numpy.patch()


class HTTPInferenceServer:
    def __init__(
        self, policy: Gr00tPolicy, port: int, host: str = "0.0.0.0", api_token: Optional[str] = None
    ):
        """
        A simple HTTP server for GR00T models; exposes `/act` to predict an action for a given observation.
            => Takes in observation dict with numpy arrays
            => Returns action dict with numpy arrays
        """
        self.policy = policy
        self.port = port
        self.host = host
        self.api_token = api_token
        self.app = FastAPI(title="GR00T Inference Server", version="1.0.0")

        # Register endpoints
        self.app.post("/act")(self.predict_action)
        self.app.get("/health")(self.health_check)

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

            # Run inference
            action = self.policy.get_action(obs)

            # Return action as JSON with numpy arrays
            return JSONResponse(content=action)

        except Exception as e:
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'observation': dict} where observation contains the required modalities.\n"
                "Example observation keys: video.ego_view, state.left_arm, state.right_arm, etc."
            )
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    def health_check(self) -> Dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "model": "GR00T"}

    def run(self) -> None:
        """Start the HTTP server."""
        print(f"Starting GR00T HTTP server on {self.host}:{self.port}")
        print("Available endpoints:")
        print("  POST /act - Get action prediction from observation")
        print("  GET  /health - Health check")
        uvicorn.run(self.app, host=self.host, port=self.port)


def create_http_server(
    policy: Gr00tPolicy, port: int, host: str = "0.0.0.0", api_token: Optional[str] = None
) -> HTTPInferenceServer:
    """Factory function to create an HTTP inference server."""
    return HTTPInferenceServer(policy, port, host, api_token)
