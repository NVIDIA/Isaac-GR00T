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

"""Profile the CPU eval-path image preprocessing of Gr00tN1d7Processor.

Times the per-control-step image pipeline:

    numpy frames -> eval albumentations transform -> layout conversions ->
    chat template -> collator tokenize (incl. the HF image-processor call)

Two pipelines are interleaved A/B within the same process (never sequential
processes, so both see the same cache/OS state):

  * ``reference`` — a verbatim copy of the pre-optimization code path
    (`_get_vlm_inputs` albumentations-eval branch + `_apply_vlm_processing`
    with the numpy->torch->numpy->PIL round trips), kept here so before/after
    can always be measured in one process.
  * ``current`` — whatever `Gr00tN1d7Processor._get_vlm_inputs` does now.

Sub-stage timings are measured on the reference pipeline (and the current
pipeline's stages where they differ).

Usage:
    PYTHONPATH=. python tools/perf/profile_preprocessing.py \
        [--checkpoint /path/to/GR00T-N1.7-3B/snapshot] [--iters 50]
"""

import argparse
from pathlib import Path
import statistics
import time

import numpy as np
from PIL import Image
import torch


DEFAULT_CKPT = "nvidia/GR00T-N1.7-3B"

LANGUAGE = "pick up the red block and place it in the bin"


# ---------------------------------------------------------------------------
# Reference (pre-optimization) pipeline: verbatim copy of the original
# `_get_vlm_inputs` (albumentations eval branch) + `_apply_vlm_processing`,
# with per-stage instrumentation.
# ---------------------------------------------------------------------------


def reference_pipeline(proc, image_keys, images, language, stage_acc=None):
    """Original eval-path code, instrumented. Returns (vlm_content, stages)."""
    t = {}

    def tick():
        return time.perf_counter()

    # Stage 1: albumentations chain per frame, incl. the per-frame np.array
    # copy done by apply_with_replay, and Stage 2: per-frame torch conversion.
    transform = proc.eval_image_transform
    temporal_stacked_images = {}
    alb_s = conv_s = 0.0
    for view in image_keys:
        transformed_tensors = []
        for img in images[view]:
            t0 = tick()
            img_array = np.array(img)
            augmented = transform(image=img_array)
            img_array = augmented["image"]
            t1 = tick()
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            transformed_tensors.append(img_tensor)
            t2 = tick()
            alb_s += t1 - t0
            conv_s += t2 - t1
        t0 = tick()
        temporal_stacked_images[view] = torch.stack(transformed_tensors)
        conv_s += tick() - t0
    t["albumentations"] = alb_s

    # Stage 2 (cont.): stack views, flatten, back to numpy
    t0 = tick()
    stacked_images = (
        torch.stack([temporal_stacked_images[view] for view in image_keys], dim=1)
        .flatten(0, 1)
        .numpy()
    )  # (T*V, C, H, W)
    t["np/torch conversions"] = conv_s + (tick() - t0)

    # Stage 3: PIL round trip
    t0 = tick()
    pil_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in stacked_images]
    t["PIL conversion"] = tick() - t0

    # Stage 4: chat template
    conversation = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in pil_images],
                {"type": "text", "text": language},
            ],
        }
    ]
    t0 = tick()
    text = proc.processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )
    t["chat template"] = tick() - t0

    if stage_acc is not None:
        for k, v in t.items():
            stage_acc.setdefault(k, []).append(v)
    return {"text": text, "images": pil_images}


def current_pipeline(proc, image_keys, images, language):
    """Whatever the (possibly optimized) production code does now."""
    vlm_inputs = proc._get_vlm_inputs(
        image_keys=image_keys,
        images=images,
        masks=None,
        image_transform=proc.eval_image_transform,
        language=language,
    )
    return vlm_inputs["vlm_content"]


def tokenize(collator, vlm_content, stage_acc=None, label=""):
    """Collator side: the full processor call (HF image processor + tokenizer)."""
    t0 = time.perf_counter()
    out = collator.processor(
        text=[vlm_content["text"]],
        images=vlm_content["images"],
        return_tensors="pt",
        padding=True,
    )
    t1 = time.perf_counter()
    if stage_acc is not None:
        stage_acc.setdefault(f"collator tokenize (incl. image proc){label}", []).append(t1 - t0)
    return out


def time_image_processor(collator, vlm_content, stage_acc, label):
    """Standalone timing of the HF image-processor call (outside the totals)."""
    t0 = time.perf_counter()
    collator.processor.image_processor(images=vlm_content["images"], return_tensors="pt")
    stage_acc.setdefault(f"HF image processor alone{label}", []).append(time.perf_counter() - t0)


def make_inputs(n_cameras, frames=2, hw=(256, 256), seed=0):
    rng = np.random.default_rng(seed)
    image_keys = [f"cam{i}" for i in range(n_cameras)]
    images = {
        k: rng.integers(0, 256, (frames, hw[0], hw[1], 3), dtype=np.uint8) for k in image_keys
    }
    return image_keys, images


def summarize(samples_s):
    ms = [s * 1e3 for s in samples_s]
    return (
        f"mean {statistics.mean(ms):7.3f}  median {statistics.median(ms):7.3f}  "
        f"stdev {statistics.stdev(ms) if len(ms) > 1 else 0.0:6.3f} ms"
    )


def run_case(proc, collator, n_cameras, hw, iters, warmup):
    image_keys, images = make_inputs(n_cameras, hw=hw)

    # Warm both pipelines (processor kernels, allocator, OS caches).
    for _ in range(warmup):
        tokenize(collator, reference_pipeline(proc, image_keys, images, LANGUAGE))
        tokenize(collator, current_pipeline(proc, image_keys, images, LANGUAGE))

    stages = {}
    totals_ref, totals_cur = [], []
    for i in range(iters):
        # Interleave A/B within the same process; alternate order each iter.
        order = [("ref", reference_pipeline), ("cur", current_pipeline)]
        if i % 2:
            order.reverse()
        for name, fn in order:
            t0 = time.perf_counter()
            if fn is reference_pipeline:
                vc = fn(proc, image_keys, images, LANGUAGE, stage_acc=stages)
            else:
                vc = fn(proc, image_keys, images, LANGUAGE)
            tokenize(
                collator,
                vc,
                stage_acc=stages,
                label=" [ref]" if name == "ref" else " [cur]",
            )
            total = time.perf_counter() - t0
            (totals_ref if name == "ref" else totals_cur).append(total)
            time_image_processor(collator, vc, stages, " [ref]" if name == "ref" else " [cur]")

    print(f"\n=== {n_cameras} camera(s) x 2 frames @ {hw[0]}x{hw[1]} (iters={iters}) ===")
    print(f"  TOTAL reference (pre-opt) : {summarize(totals_ref)}")
    print(f"  TOTAL current (this tree) : {summarize(totals_cur)}")
    try:
        from scipy import stats

        t_stat, p_value = stats.ttest_ind(totals_ref, totals_cur, equal_var=False)
        print(f"  Welch's t-test ref vs cur : t={t_stat:.2f}, p={p_value:.2e}")
    except ImportError:
        pass
    for k in sorted(stages):
        print(f"    {k:42s}: {summarize(stages[k])}")

    # Sanity: both pipelines must produce identical model inputs.
    ref = tokenize(collator, reference_pipeline(proc, image_keys, images, LANGUAGE))
    cur = tokenize(collator, current_pipeline(proc, image_keys, images, LANGUAGE))
    for k in ref:
        assert torch.equal(ref[k], cur[k]), f"MISMATCH between reference and current for {k}"
    print("  parity: reference and current model inputs are bitwise identical")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    torch.set_num_threads(torch.get_num_threads())  # no-op; keeps torch init out of timings
    proc = Gr00tN1d7Processor.from_pretrained(Path(args.checkpoint))
    proc.eval()
    collator = proc.collator
    print(f"HF image processor: {type(collator.processor.image_processor).__name__}")

    for n_cameras, hw in [(1, (256, 256)), (3, (256, 256)), (1, (240, 320))]:
        run_case(proc, collator, n_cameras, hw, args.iters, args.warmup)


if __name__ == "__main__":
    main()
