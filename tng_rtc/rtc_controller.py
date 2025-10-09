import copy
import time
import zoneinfo
import threading
from datetime import datetime
from typing import Any, Dict
from collections import deque

import torch

from .rtc_policy import RTCGr00tPolicy

from gr00t.model.policy import squeeze_dict_values


INIT_INFERENCE_DELAY_ESTIMATE: float = 0.13334      # estimate time needed for inference
CONTROL_FREQUENCY_HZ: int = 30

class RTCController:
    def __init__(self,
                 flow_policy: RTCGr00tPolicy,
                 prediction_horizon: int = 16,              # 16 is the default predict horizon of gr00t
                 min_exec_horizon: int = 6,
                 delay_buf_size: int = 10,
                 d_init: int = int(INIT_INFERENCE_DELAY_ESTIMATE * CONTROL_FREQUENCY_HZ)):

        self.flow_policy = flow_policy
        self.H = prediction_horizon
        self.s_min = min_exec_horizon
        self.d_init = d_init

        self.t: int = 0
        self.A_cur: torch.Tensor | None = None
        self.A_cur_transformed: list[Dict[str, Any]] | None = None
        self.o_cur: dict[str, Any] | None = None

        self.Q = deque([self.d_init], maxlen=delay_buf_size)

        self.M = threading.Lock()
        self.C = threading.Condition(self.M)

        self._infer_th = threading.Thread(target=self.__inference_loop, daemon=True)
        self._infer_th.start()

        # last time inference was executed
        self.last_time=time.perf_counter()

        berlin = zoneinfo.ZoneInfo("Europe/Berlin")
        self.start_time = datetime.now(berlin).strftime("%Y-%m-%d %H:%M:%S")

        print("Starting inference at: ", self.start_time, flush=True) # flush=True so that the print is shown in the docker logs
    
    def step(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        obs_for_shared = dict(obs)
        with self.C:
            if self.A_cur is None:                  # called when the controller starts and inference-loop was not called yet
                bootstrap = self.flow_policy.get_realtime_action(copy.deepcopy(obs_for_shared))
                self.__save_action(bootstrap)

            self.t += 1
            self.o_cur = obs_for_shared
            self.C.notify()

            if (self.t - 1) >= len(self.A_cur_transformed):
                single_action = self.A_cur_transformed[-1]
                raise RuntimeError("Action index out of range: ", self.t - 1)
            else:
                single_action = self.A_cur_transformed[self.t - 1]
        return single_action

    def __inference_loop(self):
        while True:
            with self.C:
                while self.t < self.s_min:
                    self.C.wait()
                
                self.__inference_step()

                self.C.notify_all()

    def __inference_step(self):
        s = self.t

        A_cur_squeezed = self.A_cur.squeeze()
        A_prev = A_cur_squeezed[s:self.H]
        if len(A_prev) < self.H:
            A_prev = torch.cat((A_prev, torch.zeros((self.H - len(A_prev), *A_prev.shape[1:]), device=A_prev.device)), 0)
        A_prev = A_prev.unsqueeze(0)

        d = max(self.Q)

        o_cur = self.o_cur  # safe reference to observation while holding the lock to avoid side effects

        self.C.release()
        realtime_action = self.flow_policy.get_realtime_action(o_cur, A_prev, d, s, self.H)
        self.C.acquire()

        self.t = self.t - s
        self.__save_action(realtime_action)

        self.Q.append(self.t)

        print(f"[inference]  latency={time.perf_counter()-self.last_time:.4f}s  s={s}  d={d}  self.t={self.t}", flush=True)
        self.last_time=time.perf_counter()

    def __save_action(self, action: torch.Tensor):
        self.A_cur = action

        # convert A_cur to a list of actions
        unnormalized_action = self.flow_policy._get_unnormalized_action(self.A_cur)
        unnormalized_action = squeeze_dict_values(unnormalized_action)

        self.A_cur_transformed = [{} for _ in range(self.H)]
        for k, v in unnormalized_action.items():
            for i in range(len(v)):
                self.A_cur_transformed[i][k] = v[i]
