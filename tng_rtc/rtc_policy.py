import torch
from typing import Dict, Any

from gr00t.model.policy import Gr00tPolicy, COMPUTE_DTYPE, unsqueeze_dict_values
from .rtc_gr00t import RTCGr00t1_5


class RTCGr00tPolicy(Gr00tPolicy):

    def get_realtime_action(self, observations: Dict[str, Any], A_prev: torch.Tensor | None = None, d: int = 0, s: int = 0, H: int = 0, max_guidance_weight=5, weight_function="exp", start_actions = None) -> torch.Tensor:
        # let the get_action handles both batch and single input
        is_batch = self._check_state_is_batched(observations)
        if not is_batch:
            observations = unsqueeze_dict_values(observations)
        # Apply transforms
        normalized_input = self.apply_transforms(observations)

        if A_prev is None:
            normalized_action = self._get_action_from_normalized_input(normalized_input, start_actions)
        else:
            normalized_action = self._get_realtime_action_from_normalized_input(normalized_input, A_prev, d, s, H, max_guidance_weight, weight_function, start_actions)

        # return normalized actions instead of unnormalized
        return normalized_action
    
    def _get_realtime_action_from_normalized_input(self, normalized_input: Dict[str, Any], A_prev: torch.Tensor, d: int, s: int, H: int, max_guidance_weight=5, weight_function="exp", start_actions = None) -> torch.Tensor:
        # Set up autocast context if needed
        with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            model_pred = self.model.get_realtime_action(normalized_input, A_prev, d, s, H, max_guidance_weight, weight_function, start_actions)

        normalized_action = model_pred["action_pred"].float()
        return normalized_action
    
    def _load_model(self, model_path):
        model = RTCGr00t1_5.from_pretrained(model_path, torch_dtype=COMPUTE_DTYPE)
        model.eval()  # Set model to eval mode

        # Update action_horizon to match modality config
        # Get the expected action horizon from the modality config
        expected_action_horizon = len(self._modality_config["action"].delta_indices)

        if expected_action_horizon != model.action_head.config.action_horizon:
            print(
                f"Policy: Recreating action head with action_horizon {expected_action_horizon} (was {model.action_head.config.action_horizon})"
            )

            # Update the action head config
            new_action_head_config = model.action_head.config
            new_action_head_config.action_horizon = expected_action_horizon

            # Import the FlowmatchingActionHead class
            from tng_rtc.rtc_flow_matching_action_head import (
                RTCFlowmatchingActionHead,
            )

            # Create new action head with updated config
            new_action_head = RTCFlowmatchingActionHead(new_action_head_config)

            # Copy the weights from the old action head to the new one
            new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)

            # Replace the action head
            model.action_head = new_action_head

            # Update model config AND the action_head_cfg dictionary that gets saved
            model.config.action_horizon = expected_action_horizon
            model.action_horizon = expected_action_horizon
            model.config.action_head_cfg["action_horizon"] = expected_action_horizon

        model.to(device=self.device)  # type: ignore

        self.model = model
