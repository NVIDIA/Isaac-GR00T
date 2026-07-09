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
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from groot.core.policy import base


class DebugLoggingPolicy(base.PolicyWrapper):
    """Wraps a policy to print/save observations passed to ``get_action``, for debugging.

    Each ``get_action`` call's observation is dumped to ``<save_dir>/step_NNNNNN/``:
        - np.ndarray values as ``.npy``
        - uint8 image arrays (``(T, H, W, C)`` or ``(H, W, C)``, C in {1, 3}) also as ``.png``
        - ``list[bytes]`` jpeg frames as ``.jpg``
        - everything else recorded in ``meta.json``

    All policy behavior is delegated to the wrapped policy; this wrapper only
    observes the inputs.
    """

    def __init__(self, policy: base.BasePolicy, save_dir: str | Path, *, strict: bool = True):
        super().__init__(policy, strict=strict)
        self._save_dir = Path(save_dir)
        if self._save_dir.exists():
            shutil.rmtree(self._save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._step = 0
        print(f"[debug] Saving received obs to {self._save_dir}", flush=True)

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._log_obs(observation)
        return self.policy.get_action(observation, options)

    # --- forward the BasePolicy abstract-method contract to the inner policy ---
    def check_observation(self, observation: dict[str, Any]) -> None:
        self.policy.check_observation(observation)

    def check_action(self, action: dict[str, Any]) -> None:
        self.policy.check_action(action)

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.policy._get_action(observation, options)

    def get_modality_config(self):
        return self.policy.get_modality_config()

    # --- debug logging ---
    def _log_obs(self, obs: dict[str, Any]) -> None:
        from PIL import Image

        shapes: dict[str, str] = {}
        timestamps: dict[str, Any] = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                shapes[k] = f"{tuple(v.shape)} {v.dtype}"
            else:
                shapes[k] = type(v).__name__
            if "timestamp" in k.lower() or "time" in k.lower():
                timestamps[k] = v.tolist() if isinstance(v, np.ndarray) else v
        print(f"[obs] shapes={shapes}", flush=True)
        print(f"[obs] timestamps={timestamps}", flush=True)

        step_dir = self._save_dir / f"step_{self._step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        meta: dict[str, Any] = {"shapes": {}, "non_array": {}}
        for k, v in obs.items():
            safe_k = k.replace("/", "__")
            if isinstance(v, np.ndarray):
                np.save(step_dir / f"{safe_k}.npy", v)
                meta["shapes"][k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
                if v.dtype == np.uint8 and v.ndim == 4 and v.shape[-1] in (1, 3):
                    for i in range(v.shape[0]):
                        frame = v[i]
                        if frame.shape[-1] == 1:
                            frame = frame[..., 0]
                        Image.fromarray(frame).save(step_dir / f"{safe_k}__frame_{i:02d}.png")
                elif v.dtype == np.uint8 and v.ndim == 3 and v.shape[-1] in (1, 3):
                    frame = v if v.shape[-1] == 3 else v[..., 0]
                    Image.fromarray(frame).save(step_dir / f"{safe_k}.png")
            elif isinstance(v, list) and v and isinstance(v[0], (bytes, bytearray)):
                # JPEG-encoded video frames sent as list[bytes]
                # (real-robot wrappers serialize frames client-side).
                for i, frame_bytes in enumerate(v):
                    (step_dir / f"{safe_k}__frame_{i:02d}.jpg").write_bytes(bytes(frame_bytes))
                meta["shapes"][k] = {"len": len(v), "dtype": "jpeg_bytes"}
            else:
                try:
                    json.dumps(v)
                    meta["non_array"][k] = v
                except (TypeError, ValueError):
                    meta["non_array"][k] = str(v)
        (step_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        self._step += 1
