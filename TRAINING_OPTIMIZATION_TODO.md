# Fine-tuning performance TODOs

Findings from a static read of the fine-tuning path
(`launch_finetune.py` → `experiment.py:run()` → `Gr00tTrainer` → dataset/collator),
recorded 2026-08-04.

**Nothing here has been benchmarked.** There was no venv on the machine at the time of
the review, so every claim below comes from reading the code, not from measurement.
Sizing numbers are estimates and marked as such.

---

## Applied

### 1. Use fused AdamW for fine-tuning — `gr00t/experiment/launch_finetune.py`

`config.training.optim` was hardcoded to `"adamw_torch"`, overriding the repo's own
default of `"adamw_torch_fused"` (`gr00t/configs/training/training_config.py:55`) on
every run that goes through `examples/finetune.sh`. `optim` is not exposed in
`FinetuneConfig`, so the `-- <extra args>` passthrough could not reach it either —
editing the launcher was the only way to change it.

Same math: HF maps `adamw_torch_fused` to `torch.optim.AdamW(fused=True)`, a single
multi-tensor CUDA kernel instead of a per-parameter Python loop. The override dates to
the N1.6 release (`4e62473`) with no explanatory comment.

Changed to `"adamw_torch_fused"`.

**Still to verify:** step-time delta. Expect the win to be largest with the default
trainable set (`tune_llm=False`, `tune_visual=False` → DiT + projector + VLLN, i.e. many
small tensors), and to shrink as batch size grows and the optimizer step becomes a
smaller fraction of the step.

### 2. Load the frozen backbone in bf16 — `gr00t/experiment/launch_finetune.py`

`config.model.load_bf16` was hardcoded to `False`. In `qwen3_backbone.py:199` the fp32
master-weight cast is *gated on `load_bf16`*:

```python
if load_bf16 and trainable_params_fp32:
    # cast trainable parameters to fp32
```

So with `load_bf16=False`, `from_pretrained` receives no `torch_dtype`, the bf16 Cosmos
checkpoint materializes in fp32, and nothing is ever cast — the run pays ~2× backbone
memory for weights that `bf16=True` autocast converts to bf16 in the forward anyway.

With the defaults (`tune_llm=False`, `tune_visual=False`) the entire backbone is frozen,
so it carries no optimizer state and has no reason to sit in fp32. Setting `load_bf16=True`
keeps frozen params in bf16 while still casting anything trainable back to fp32 via the
branch above. `warn_configs` (`experiment.py:99-104`) warns when
`backbone_trainable_params_fp32` is not `True`, which suggests `load_bf16=True` +
`trainable_params_fp32=True` is the intended pairing.

Changed to `True`.

**Still to verify — do this before a long run:** loss-parity check against `load_bf16=False`
over a few hundred steps, and measure peak memory + load time. This is the one applied
change that alters numerics (frozen backbone weights are bf16 rather than fp32-autocast-to-bf16).
Expected saving is roughly 2 GB on the truncated backbone (`select_layer=12` pops LLM
layers above 12), plus faster startup.

---

## Open

### 3. Dataloader is probably the bottleneck — untested

Current defaults: `examples/finetune.sh:10` → 4 workers; `FinetuneConfig:128` and
`TrainingConfig:108` → 2.

All of the per-sample work runs in worker processes:

- albumentations augmentation + color jitter
- `Gr00tN1d7DataCollator.__call__` runs the **full Qwen3-VL image processor** (resize,
  normalize, patchify, for every image in the batch) at
  `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py:190`. `collate_fn` executes in the
  worker, so this parallelizes — but only across however many workers are configured.

Two things to try:

- Raise `DATALOADER_NUM_WORKERS` (free to test; pick relative to core count).
- Set `prefetch_factor`. It is never set anywhere in the repo — see the
  `dataloader_params` dict in `gr00t/experiment/trainer.py:206-212` — so torch's default
  of 2 applies. 4 is worth trying alongside more workers.

**Measure first.** `config.training.enable_profiling` (`experiment.py:338`) already emits
chrome traces per rank. Run ~200 steps with it on to establish whether the run is
data-bound or compute-bound before tuning anything here.

### 4. Dead code in `gr00t/experiment/trainer.py:51-115`

`_BatchIterator` and `_PrefetchIterator` (~65 lines, including a hand-rolled threaded
prefetcher with a `queue.Queue(maxsize=4)`) are referenced nowhere in the repo.
`get_train_dataloader` builds a plain `torch.utils.data.DataLoader`.

Not a performance issue, but actively misleading:

- the module docstring advertises data-loading/forward-pass profiling that no longer exists
- the `Gr00tTrainer` class docstring claims it "bypasses torch dataloader and makes data
  collator async", which it does not

Delete the classes, fix both docstrings.

### 5. Dead accuracy-logging block in `gr00t/experiment/trainer.py:289-325`

Guarded on `"labels" in inputs`. N1.7's collator emits `state` / `action` / `vlm_content`,
and the model returns `{"loss", "action_loss", ...}` — no `labels`, no `logits`. This is
token-prediction machinery left over from an earlier architecture and never fires.

Harmless today, but if it ever did fire it would `.cpu()` the full logits and call
`_nested_gather` every `logging_steps` — a hard device sync on the training path. Better
deleted than left armed. `_batch_accuracy` (`trainer.py:118-149`) goes with it.

Minor cleanups in the same file:

- `get_train_dataloader` sets `self.args.ignore_data_skip = True`, already set at
  `experiment.py:302`
- `compute_loss` calls `super().compute_loss(..., return_outputs=True)` unconditionally
- commented-out `torch.save` debug lines and an `import ipdb` at `trainer.py:276-281`

### 6. Test configs no longer mirror production fine-tuning

`tests/gr00t/experiment/test_experiment_run.py:117,133` and
`tests/gr00t/experiment/_run_distributed_experiment.py:65,82` build their config by hand
and still set `load_bf16 = False` / `optim = "adamw_torch"`.

They construct configs independently rather than importing from `launch_finetune.py`, so
items 1 and 2 above did **not** break them — but the tests now exercise a different
configuration than a real fine-tune does. Either update them to match, or factor the
shared launcher settings into one place both can import.

---

## Considered and rejected

- **`gradient_checkpointing`** — already `False` (`training_config.py:94`), which is the
  right default for speed. It is the memory lever if a larger batch OOMs, and pairs with
  the gradient-accumulation work in PR #726. No change needed.
- **`torch.compile`** — `TrainingArguments` supports `torch_compile`, but the dynamic
  image-grid shapes coming out of Qwen3-VL would likely cause recompilation churn. Not
  low-hanging; revisit only if profiling shows a compute-bound run with stable shapes.
