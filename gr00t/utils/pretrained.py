import os
import inspect
import pickle
import json
from pathlib import Path
from functools import wraps
from dataclasses import is_dataclass
from huggingface_hub import ModelHubMixin, ModelCard, HfApi, snapshot_download
from huggingface_hub.utils import validate_hf_hub_args, logging
from typing import Dict, Optional, Type, Union, TypeVar
from gr00t.data.dataset import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag


# see huggingface_hub.ModelHubMixin for more details
# https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hub_mixin.py

# Flexible readme card
DEFAULT_MODEL_CARD = """
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

This model has been pushed to the Hub using the [PytorchModelHubMixin](https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin) integration:
- Code: {{ repo_url | default("[More Information Needed]", true) }}
- Paper: {{ paper_url | default("[More Information Needed]", true) }}
- Docs: {{ docs_url | default("[More Information Needed]", true) }}

{{ model_readme | default("# Gr00t", true) }}
"""
T = TypeVar("T", bound="ModelHubMixin")

logger = logging.get_logger(__name__)


def set_docstring(fn):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__doc__ = fn.__doc__
        return wrapper

    return decorator


def serializeEmbodimentTag(tag: EmbodimentTag) -> str:
    return tag.value


def deserializeEmbodimentTag(tag: str) -> EmbodimentTag:
    return EmbodimentTag(tag)


class ModalityConfigDict(dict):
    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError(f"Keys must be strings, got {type(key).__name__}")

        if not isinstance(value, ModalityConfig):
            raise TypeError(f"Values must be ModalityConfig objects, got {type(value).__name__}")

        super().__setitem__(key, value)


def serializeModalityConfig(x):
    return {k: v.model_dump_json() for k, v in x.items()}


def deserializeModalityConfig(data) -> ModalityConfigDict:
    return ModalityConfigDict({k: ModalityConfig.model_validate_json(v) for k, v in data.items()})


class Gr00tMixin(ModelHubMixin):
    def __init_subclass__(
        cls,
        *args,
        # if these parameters are not defined in the ancestor class
        # we default to these params and pass them to ModelHubMixin
        model_card_template: str = DEFAULT_MODEL_CARD,
        library_name: str = "gr00t",
        repo_url="https://github.com/NVIDIA/Isaac-GR00T",
        **kwargs,
    ) -> None:
        # update tags list
        kwargs.setdefault("tags", [])
        if "gr00t" not in kwargs["tags"]:
            kwargs["tags"].append("gr00t")

        # add modality config to coders
        if "coders" not in kwargs:
            kwargs["coders"] = {}
        if ModalityConfigDict not in kwargs["coders"]:
            kwargs["coders"][ModalityConfigDict] = (
                serializeModalityConfig,
                deserializeModalityConfig,
            )
            kwargs["coders"][EmbodimentTag] = (
                serializeEmbodimentTag,
                deserializeEmbodimentTag,
            )

        super().__init_subclass__(
            *args, model_card_template=model_card_template, library_name=library_name, repo_url=repo_url, **kwargs
        )

    def _save_pretrained(self, save_directory: Path):
        # create necessary directories
        save_directory.mkdir(parents=True, exist_ok=True)
        (save_directory / "experiment_cfg").mkdir(parents=True, exist_ok=True)

        # save serializable class init parms in config of its own
        config_path = save_directory / "class_config.json"
        matadatas = self._hub_mixin_config.pop("metadatas")  # type: ignore
        class_config = self._hub_mixin_config
        # update model_path on each save, including when pushing to hub
        class_config["model_path"] = getattr(self, "_hub_mixin_repo_id", str(save_directory))  # type: ignore
        # del attribute after save to differentiate between saving locally and pushing
        # even if it does not exist
        if hasattr(self, "_hub_mixin_repo_id"):
            delattr(self, "_hub_mixin_repo_id")
        config_str = json.dumps(class_config, sort_keys=True, indent=2)
        config_path.write_text(config_str)
        # save metadata in subfolder
        metadata_path = save_directory / "experiment_cfg" / "metadata.json"
        metadata_path.write_text(json.dumps(matadatas, sort_keys=True, indent=2))
        # save composed modality
        composed_modality_path = save_directory / "composed_modality.pickle"
        with composed_modality_path.open("wb") as f:
            pickle.dump(self._modality_transform, f)  # type: ignore

        # IMPORTANT: save model and its config
        # keeping the model in the base directory so that its from_pretrained will stay compatible
        self.model.save_pretrained(save_directory)  # type: ignore

    # update model card content
    def generate_model_card(self, *args, **kwargs) -> ModelCard:
        """Generate a model card for this model.
        1- get the model_readme content from the class attribute
        class Model(PretrainedModel):
            model_readme = "# Gr00t"
            def __init__(...)

        2- get the model_readme content from the parameter
        class Model(PretrainedModel,
                    model_readme = "# Gr00t"
                    ):
            def __init__(...)
        3- use the default one defined in the DEFAULT_MODEL_CARD above
        """
        # update model_readme using class attrribute or a parameter that has been defined in the superior class
        model_readme = getattr(self, "model_readme", None)
        if model_readme is None:
            model_readme = self._hub_mixin_config.get("model_readme", None)  # type: ignore

        card = ModelCard.from_template(
            card_data=self._hub_mixin_info.model_card_data,
            template_str=self._hub_mixin_info.model_card_template,
            repo_url=self._hub_mixin_info.repo_url,
            docs_url=self._hub_mixin_info.docs_url,
            **kwargs,
        )
        return card

    # switched from config.json to class_config.json
    # instantiating directly here instead of inside _from_pretrained
    @classmethod
    @validate_hf_hub_args
    @set_docstring(ModelHubMixin.from_pretrained)
    def from_pretrained(
        cls: Type[T],
        pretrained_model_name_or_path: Union[str, Path],
        *,
        force_download: bool = False,
        resume_download: Optional[bool] = None,
        proxies: Optional[Dict] = None,
        token: Optional[Union[str, bool]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        local_files_only: bool = False,
        revision: Optional[str] = None,
        **model_kwargs,
    ) -> T:
        model_id = str(pretrained_model_name_or_path)
        config_file: Optional[str] = None
        local_path = Path(model_id)
        if local_path.exists() and (local_path / "class_config.json").exists():
            repo_path = local_path
        else:
            repo_path = snapshot_download(
                model_id,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        config_file = os.path.join(repo_path, "class_config.json")

        # Ensure repo_path is a Path object
        snapshot_folder = Path(repo_path)

        # Read config
        config = None
        if config_file is not None:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)

            # Decode custom types in config
            for key, value in config.items():
                if key in cls._hub_mixin_init_parameters:
                    expected_type = cls._hub_mixin_init_parameters[key].annotation
                    if expected_type is not inspect.Parameter.empty:
                        config[key] = cls._decode_arg(expected_type, value)

            # Populate model_kwargs from config
            for param in cls._hub_mixin_init_parameters.values():
                if param.name not in model_kwargs and param.name in config:
                    model_kwargs[param.name] = config[param.name]

            # Check if `config` argument was passed at init
            if "config" in cls._hub_mixin_init_parameters and "config" not in model_kwargs:
                # Decode `config` argument if it was passed
                config_annotation = cls._hub_mixin_init_parameters["config"].annotation
                config = cls._decode_arg(config_annotation, config)

                # Forward config to model initialization
                model_kwargs["config"] = config

            # Inject config if `**kwargs` are expected
            if is_dataclass(cls):
                for key in cls.__dataclass_fields__:
                    if key not in model_kwargs and key in config:  # type: ignore
                        model_kwargs[key] = config[key]  # type: ignore
            elif any(
                param.kind == inspect.Parameter.VAR_KEYWORD for param in cls._hub_mixin_init_parameters.values()
            ):
                for key, value in config.items():  # type: ignore
                    if key not in model_kwargs:
                        model_kwargs[key] = value

            # Finally, also inject if `_from_pretrained` expects it
            if cls._hub_mixin_inject_config and "config" not in model_kwargs:
                model_kwargs["config"] = config

        # if composed modality exists
        composed_modality_path = snapshot_folder / "composed_modality.pickle"
        if composed_modality_path.exists():
            print("exists")
            with composed_modality_path.open("rb") as f:
                composed_modality = pickle.load(f)
            model_kwargs["modality_transform"] = composed_modality
        model_kwargs["model_path"] = str(model_id)
        instance = cls(**model_kwargs)

        # Implicitly set the config as instance attribute if not already set by the class
        # This way `config` will be available when calling `save_pretrained` or `push_to_hub`.
        if config is not None and (getattr(instance, "_hub_mixin_config", None) in (None, {})):
            instance._hub_mixin_config = config

        return instance

    @set_docstring(ModelHubMixin.push_to_hub)
    def push_to_hub(
        self,
        repo_id,
        *,
        config=None,
        commit_message="Push model using huggingface_hub.",
        private=None,
        token=None,
        branch=None,
        create_pr=None,
        allow_patterns=None,
        ignore_patterns=None,
        delete_patterns=None,
        model_card_kwargs=None,
    ):
        api = HfApi(token=token)
        repo_id = api.create_repo(repo_id=repo_id, private=private, exist_ok=True).repo_id
        self._hub_mixin_repo_id = repo_id
        return super().push_to_hub(
            repo_id,
            config=config,
            commit_message=commit_message,
            private=private,
            token=token,
            branch=branch,
            create_pr=create_pr,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            delete_patterns=delete_patterns,
            model_card_kwargs=model_card_kwargs,
        )
