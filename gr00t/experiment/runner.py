import json
import os
from pathlib import Path
from enum import Enum, auto

import torch
from transformers import TrainingArguments, set_seed

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.trainer import DualBrainTrainer
from gr00t.model.backbone.eagle2_hg_model.inference_eagle_repo import EagleProcessor
from gr00t.model.gr00t_n1 import GR00T_N1
from gr00t.model.transforms import DefaultDataCollatorGR00T
from gr00t.utils.experiment import (
    CheckpointFormatCallback,
    safe_save_model_for_hf_trainer,
)

# New: Enum to manage training lifecycle states
class TrainingState(Enum):
    INIT = auto()
    LOADING_DATA = auto()
    TRAINING = auto()
    SAVING = auto()
    COMPLETED = auto()
    ERROR = auto()


class TrainRunner:
    def __init__(
        self,
        model: GR00T_N1,
        training_args: TrainingArguments,
        train_dataset: LeRobotSingleDataset,
        resume_from_checkpoint: bool = False,
    ):
        # Track training state
        self.state = TrainingState.INIT
        print(f"[STATE] Current state: {self.state.name}")

        self.training_args = training_args
        self.output_dir = Path(training_args.output_dir)
        self.exp_cfg_dir = self.output_dir / "experiment_cfg"
        self.exp_cfg_dir.mkdir(parents=True, exist_ok=True)
        self.resume_from_checkpoint = resume_from_checkpoint
        self.train_dataset = train_dataset

        training_args.run_name = (
            training_args.output_dir.split("/")[-1]
            if training_args.run_name is None
            else training_args.run_name
        )
        print(f"Run name: {training_args.run_name}")

        # Move to LOADING_DATA state
        self.state = TrainingState.LOADING_DATA
        print(f"[STATE] Current state: {self.state.name}")

        data_collator = DefaultDataCollatorGR00T(
            processor=EagleProcessor(),
        )

        compute_dtype = torch.float16 if training_args.bf16 else torch.float32
        set_seed(training_args.seed)

        trainer = self.create_trainer(
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            compute_dtype=compute_dtype,
        )
        self.trainer = trainer

        self.rank = int(os.environ.get("RANK", 0))
        if self.rank == 0:
            metadata_json = {}
            metadata_path = self.exp_cfg_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata_json = json.load(f)
            metadata_json.update(
                {train_dataset.tag: train_dataset.metadata.model_dump(mode="json")}
            )
            with open(metadata_path, "w") as f:
                json.dump(metadata_json, f, indent=4)

        report_to = training_args.report_to
        if report_to == "wandb":
            if "WANDB_PROJECT" not in os.environ:
                os.environ["WANDB_PROJECT"] = "gr00t-training"
            if "WANDB_RUN_ID" not in os.environ:
                runtime_id = os.environ.get("RUNTIME_ID", None)
                if runtime_id:
                    os.environ["WANDB_RUN_ID"] = runtime_id
            os.environ["WANDB_DIR"] = training_args.output_dir

            wandb_config_file = self.output_dir / "wandb_config.json"
            with open(wandb_config_file, "w") as f:
                json.dump(
                    {
                        "project": os.environ.get("WANDB_PROJECT", ""),
                        "run_id": os.environ.get("WANDB_RUN_ID", ""),
                    },
                    f,
                )
            training_args.report_to = ["wandb"]
        else:
            tensorboard_dir = Path(training_args.output_dir) / "runs"
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            print(f"TensorBoard logs will be saved to: {tensorboard_dir}")
            training_args.report_to = ["tensorboard"]

    def create_trainer(
        self,
        model,
        training_args,
        train_dataset,
        data_collator,
        compute_dtype,
        global_batch_size=None,
    ):
        if global_batch_size is not None:
            bs = training_args.per_device_train_batch_size
            num_gpus = torch.cuda.device_count()
            grad_acc = max(1, global_batch_size // (bs * num_gpus))
            training_args.gradient_accumulation_steps = grad_acc
            print(
                f"Set global batch size to {global_batch_size}, set gradient accumulation steps to {grad_acc}"
            )

        trainer = DualBrainTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            compute_dtype=compute_dtype,
        )

        ckpt_format_callback = CheckpointFormatCallback(
            run_name=training_args.run_name, exp_cfg_dir=self.exp_cfg_dir
        )
        trainer.add_callback(ckpt_format_callback)

        print(
            f"train dataloader length: {len(trainer.get_train_dataloader())}\n"
            f"train dataset length: {len(trainer.train_dataset)}\n"
            f"GPU memory before training: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024} GB",
            flush=True,
        )
        return trainer

    def train(self):
        try:
            self.state = TrainingState.TRAINING
            print(f"[STATE] Current state: {self.state.name}")

            self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)

            self.state = TrainingState.SAVING
            print(f"[STATE] Current state: {self.state.name}")

            self.trainer.save_state()

            safe_save_model_for_hf_trainer(
                trainer=self.trainer,
                output_dir=self.training_args.output_dir,
            )

            self.state = TrainingState.COMPLETED
            print(f"[STATE] Current state: {self.state.name}")

        except Exception as e:
            self.state = TrainingState.ERROR
            print(f"[STATE] Current state: {self.state.name}")
            print(f"[ERROR] Training failed: {e}")
