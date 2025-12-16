from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from transformers import ProcessorMixin

from gr00t.data.types import EmbodimentTag, ModalityConfig


class BaseProcessor(ProcessorMixin):
    def __call__(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Process a list of messages and return a dictionary of model inputs.

        Args:
            messages (list[dict[str, Any]]): List of messages to process.

        Returns:
            dict[str, Any]: Dictionary of model inputs.

        Example:
        >>> processor = BaseProcessor()
        >>> messages = [
        >>>    {"type": MessageType.START_OF_EPISODE.value, "content": ""},
        >>>    {"type": MessageType.EPISODE_STEP.value, "content": VLAStepData},
        >>>    {"type": MessageType.TEXT.value, "role" : "user", "content": "Please give me the apple"},
        >>>    {"type": MessageType.TEXT.value, "role" : "assistant", "content": "I need to move my left hand to get the apple"},
        >>>    {"type": MessageType.EPISODE_STEP.value, "content": VLAStepData},
        >>>    {"type": MessageType.EPISODE_STEP.value, "content": VLAStepData},
        >>>    {"type": MessageType.END_OF_EPISODE.value, "content": ""},
        >>> ]
        >>> model_input = processor(messages)
        >>> print(model_input)
        """
        raise NotImplementedError("Subclasses must implement __call__")

    def decode_action(
        self,
        action: np.ndarray,
        embodiment_tag: EmbodimentTag,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Decode the action from the model output."""
        raise NotImplementedError("Subclasses must implement decode_action")

    @property
    def collator(self):
        raise NotImplementedError("Subclasses must implement collator")

    @abstractmethod
    def set_statistics(self, statistics: dict[str, Any], override: bool = False) -> None:
        """Set normalization statistics."""
        pass

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def get_modality_configs(self) -> dict[str, dict[str, ModalityConfig]]:
        """Get the modality configurations.

        Returns:
            dict[str, dict[str, ModalityConfig]]: The modality configurations, where
                modality_configs[embodiment_tag][modality] = ModalityConfig
        """
        return getattr(self, "modality_configs")


class ShardedDataset(ABC):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of shards."""
        pass

    @abstractmethod
    def get_shard_length(self, idx: int) -> int:
        """Get the length of the shard at index idx."""
        pass

    @abstractmethod
    def get_shard(self, idx: int) -> list:
        """Get the shard at index idx."""
        pass

    def set_processor(self, processor: BaseProcessor):
        self.processor = processor

    def get_dataset_statistics(self) -> dict[str, Any]:
        """Get the dataset statistics. This is only required for dataloaders for robtics datasets."""
        raise NotImplementedError()


# # Example chat formats (processor input)
# # Single step
# messages = [
#     {"type": "episode_step", "content": VLAStepData},
# ]
# # Single episode
# messages = [
#     {"type": MessageType.START_OF_EPISODE.value, "content": ""},
#     {"type": MessageType.EPISODE_STEP.value, "content": VLAStepData},
#     {"type": MessageType.EPISODE_STEP.value, "content": VLAStepData},
#     {"type": MessageType.EPISODE_STEP.value, "content": VLAStepData},
#     {"type": MessageType.END_OF_EPISODE.value, "content": ""},
# ]
# # Multiple episodes
# messages = [
#     {"type": MessageType.START_OF_EPISODE.value, "content": ""},
#     {"type": MessageType.EPISODE_STEP.value, "content": VLAStepData},
#     {"type": MessageType.EPISODE_STEP.value, "content": VLAStepData},
#     {"type": MessageType.END_OF_EPISODE.value, "content": ""},
#     {"type": MessageType.START_OF_EPISODE.value, "content": ""},
#     {"type": MessageType.EPISODE_STEP.value, "content": VLAStepData},
#     {"type": MessageType.EPISODE_STEP.value, "content": VLAStepData},
#     {"type": MessageType.END_OF_EPISODE.value, "content": ""},
# ]

# # Example usage
# messages = dataset[idx]
# model_input = processor(messages)
# model_output = model(**model_input) # or model.generate(**model_input)
# decoded_action = processor.decode_action(model_output)
