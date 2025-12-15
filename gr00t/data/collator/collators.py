from typing import Any, Dict, List

import torch


class BasicDataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        fields = features[0].keys()
        batch = {}
        for key in fields:
            batch[key] = torch.stack([item[key] for item in features])
        return batch
