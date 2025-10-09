from typing import Dict, Any

from gr00t.eval.robot import RobotInferenceServer

from .rtc_policy import RTCGr00tPolicy
from .rtc_controller import RTCController


class RTCInferenceServer(RobotInferenceServer):
    def __init__(self, model: RTCGr00tPolicy, host = "*", port = 8000, api_token = None, pred_horiton=16):
        super().__init__(model, host, port, api_token)

        self.rtc_controller = RTCController(model, prediction_horizon=pred_horiton, min_exec_horizon=int(0.4 * pred_horiton))

        self.register_endpoint("get_action", self._get_action)

    def _get_action(self, obs: Dict[str, Any]):
        return self.rtc_controller.step(obs)
