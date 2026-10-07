# LIBERO family benchmarks

This guide reproduces GR00T N2 training and compares N2 and N1.7 closed-loop
evaluation for LIBERO-10, LIBERO-Spatial, LIBERO-Object, and LIBERO-Goal. Each checked-in
benchmark recipe pins its dataset, base checkpoint, training configuration, and
standard ten-task evaluation suite.

## Requirements

- Linux with eight H100 GPUs to reproduce the documented training runs
- Access to the pinned N2 base checkpoint on Hugging Face
- The locked training environment installed with `uv sync --frozen --extra training`
- Enough local storage for the dataset, checkpoints, reports, and rollout videos

Install the pinned LIBERO simulator and LIBERO-PRO assets once from the package
root:

```bash
groot/infra/closed_loop/backends/libero_pro/setup.sh
```

## Benchmark recipes

The documented runs used one eight-GPU H100 node. Each recipe trains for 10,000
steps with global batch 384, then evaluates the final checkpoint on 50 episodes
for each task in its standard suite. The recipe JSON is the source of truth for
the dataset revision, base checkpoint, experiment, and evaluation settings.

| Benchmark | Recipe JSON | Dataset | Standard suite |
| --- | --- | --- | --- |
| LIBERO-10 | [`libero_10_n2.json`](../groot/infra/benchmark/specs/libero_10_n2.json) | `IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot` | `libero_pro_10` |
| LIBERO-Spatial | [`libero_spatial_n2.json`](../groot/infra/benchmark/specs/libero_spatial_n2.json) | `IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot` | `libero_pro_spatial` |
| LIBERO-Object | [`libero_object_n2.json`](../groot/infra/benchmark/specs/libero_object_n2.json) | `IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot` | `libero_pro_object` |
| LIBERO-Goal | [`libero_goal_n2.json`](../groot/infra/benchmark/specs/libero_goal_n2.json) | `IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot_fixed` | `libero_pro_goal` |

Launch one benchmark from the package root:

```bash
uv run --no-sync python -m groot.infra.oss train-eval \
  --benchmark-recipe libero_10_n2 \
  --output-dir ./outputs/n2-libero-10

uv run --no-sync python -m groot.infra.oss train-eval \
  --benchmark-recipe libero_spatial_n2 \
  --output-dir ./outputs/n2-libero-spatial

uv run --no-sync python -m groot.infra.oss train-eval \
  --benchmark-recipe libero_object_n2 \
  --output-dir ./outputs/n2-libero-object

uv run --no-sync python -m groot.infra.oss train-eval \
  --benchmark-recipe libero_goal_n2 \
  --output-dir ./outputs/n2-libero-goal
```

Add `--dry-run` to resolve and inspect a recipe without downloading data,
training, or evaluating. Each completed run writes its final checkpoint and
`benchmark-report.json` below the selected output directory.

## Closed-loop results

N2 results use checkpoints at 10,000 cumulative post-training steps; N1.7
results use the original 20,000-step CI checkpoints. Both models use the same
task sets and rollout settings: 50 episodes per task, seed 0, a 400-action-step
limit, eight executed actions per chunk, and four denoising steps. Each
standard-suite task row contains 50 episodes; each perturbation family
contains 500 episodes.

### LIBERO-10

| Task / perturbation | N2 success rate | N1.7 success rate |
| --- | ---: | ---: |
| **Standard** | | |
| **Overall (500 episodes)** | **94.0%** | **85.2%** |
| **Pro** | | |
| Object (500 episodes) | 68.2% | 47.2% |
| Position swap (500 episodes) | 14.2% | 0.0% |
| Language (500 episodes) | 95.2% | 83.0% |
| Task (500 episodes) | 21.4% | 3.2% |
| **Overall (2,000 episodes)** | **49.8%** | **33.4%** |

### LIBERO-Spatial

| Task / perturbation | N2 success rate | N1.7 success rate |
| --- | ---: | ---: |
| **Standard** | | |
| Bowl between plate and ramekin → plate (50 episodes) | 94% | 92.0% |
| Bowl next to ramekin → plate (50 episodes) | 100% | 98.0% |
| Bowl at table center → plate (50 episodes) | 100% | 100.0% |
| Bowl on cookie box → plate (50 episodes) | 100% | 100.0% |
| Bowl in cabinet top drawer → plate (50 episodes) | 100% | 100.0% |
| Bowl on ramekin → plate (50 episodes) | 40% | 76.0% |
| Bowl next to cookie box → plate (50 episodes) | 100% | 100.0% |
| Bowl on stove → plate (50 episodes) | 100% | 98.0% |
| Bowl next to plate → plate (50 episodes) | 92% | 90.0% |
| Bowl on wooden cabinet → plate (50 episodes) | 100% | 96.0% |
| **Overall (500 episodes)** | **92.6%** | **95.0%** |
| **Pro** | | |
| Language (500 episodes) | 92.0% | 91.6% |
| Object (500 episodes) | 96.6% | 94.8% |
| Position swap (500 episodes) | 41.0% | 3.0% |
| Task (500 episodes) | 30.4% | 53.8% |
| **Overall (2,000 episodes)** | **65.0%** | **60.8%** |

### LIBERO-Object

| Task / perturbation | N2 success rate | N1.7 success rate |
| --- | ---: | ---: |
| **Standard** | | |
| Alphabet soup → basket (50 episodes) | 100% | 98.0% |
| BBQ sauce → basket (50 episodes) | 100% | 98.0% |
| Butter → basket (50 episodes) | 100% | 98.0% |
| Chocolate pudding → basket (50 episodes) | 100% | 100.0% |
| Cream cheese → basket (50 episodes) | 100% | 100.0% |
| Ketchup → basket (50 episodes) | 100% | 100.0% |
| Milk → basket (50 episodes) | 100% | 100.0% |
| Orange juice → basket (50 episodes) | 92% | 98.0% |
| Salad dressing → basket (50 episodes) | 100% | 98.0% |
| Tomato sauce → basket (50 episodes) | 100% | 100.0% |
| **Overall (500 episodes)** | **99.2%** | **99.0%** |
| **Pro** | | |
| Language (500 episodes) | 98.6% | 98.2% |
| Object (500 episodes) | 85.6% | 90.4% |
| Position swap (500 episodes) | 10.0% | 0.0% |
| Task (500 episodes) | 0.0% | 9.8% |
| **Overall (2,000 episodes)** | **48.5%** | **49.6%** |

### LIBERO-Goal

| Task / perturbation | N2 success rate | N1.7 success rate |
| --- | ---: | ---: |
| **Standard** | | |
| Open the middle drawer of the cabinet (50 episodes) | 92% | 100.0% |
| Put the bowl on the stove (50 episodes) | 100% | 100.0% |
| Put the wine bottle on top of the cabinet (50 episodes) | 88% | 88.0% |
| Open the top drawer and put the bowl inside (50 episodes) | 98% | 98.0% |
| Put the bowl on top of the cabinet (50 episodes) | 98% | 100.0% |
| Push the plate to the front of the stove (50 episodes) | 100% | 98.0% |
| Put the cream cheese in the bowl (50 episodes) | 100% | 100.0% |
| Turn on the stove (50 episodes) | 100% | 100.0% |
| Put the bowl on the plate (50 episodes) | 98% | 100.0% |
| Put the wine bottle on the rack (50 episodes) | 100% | 96.0% |
| **Overall (500 episodes)** | **97.4%** | **98.0%** |
| **Pro** | | |
| Language (500 episodes) | 97.8% | 95.6% |
| Object (500 episodes) | 87.0% | 82.8% |
| Position swap (500 episodes) | 23.6% | 1.0% |
| Task (500 episodes) | 8.8% | 10.4% |
| **Overall (2,000 episodes)** | **54.3%** | **47.4%** |
