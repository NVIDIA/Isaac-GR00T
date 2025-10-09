import torch

from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHeadConfig
from gr00t.model.gr00t_n1 import GR00T_N1_5
from transformers.feature_extraction_utils import BatchFeature

from .rtc_flow_matching_action_head import RTCFlowmatchingActionHead


class RTCGr00t1_5(GR00T_N1_5):

    def __init__(self, config, local_model_path):
        super().__init__(config, local_model_path)

        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = RTCFlowmatchingActionHead(action_head_cfg)
    
    def get_realtime_action(
        self,
        inputs: dict,
        A_prev: torch.Tensor,
        d: int,
        s: int,
        H: int,
        max_guidance_weight=5,
        weight_function = "exp",
        start_actions = None
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_realtime_action(backbone_outputs, action_inputs, A_prev, d, s, H, max_guidance_weight, weight_function, start_actions)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs
