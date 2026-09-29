# LIBERO family benchmarks

This guide reproduces GR00T N2 training and closed-loop evaluation for
LIBERO-10, LIBERO-Spatial, LIBERO-Object, and LIBERO-Goal. Each checked-in
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

These closed-loop results use checkpoints at 10,000 cumulative post-training
steps. Each standard-suite row contains 50 episodes.

### LIBERO-10

| Standard suite | Episodes | Successes | Success rate |
| --- | ---: | ---: | ---: |
| **Overall** | **500** | **470** | **94.0%** |

| LIBERO-PRO perturbation | Episodes | Successes | Success rate |
| --- | ---: | ---: | ---: |
| Object | 500 | 341 | 68.2% |
| Position swap | 500 | 71 | 14.2% |
| Language | 500 | 476 | 95.2% |
| Task | 500 | 107 | 21.4% |
| **Overall** | **2,000** | **995** | **49.8%** |

### LIBERO-Spatial

| Original task | Successes | Success rate |
| --- | ---: | ---: |
| Bowl between plate and ramekin → plate | 47/50 | 94% |
| Bowl next to ramekin → plate | 50/50 | 100% |
| Bowl at table center → plate | 50/50 | 100% |
| Bowl on cookie box → plate | 50/50 | 100% |
| Bowl in cabinet top drawer → plate | 50/50 | 100% |
| Bowl on ramekin → plate | 20/50 | 40% |
| Bowl next to cookie box → plate | 50/50 | 100% |
| Bowl on stove → plate | 50/50 | 100% |
| Bowl next to plate → plate | 46/50 | 92% |
| Bowl on wooden cabinet → plate | 50/50 | 100% |
| **Overall** | **463/500** | **92.6%** |

| LIBERO-PRO perturbation | Successes | Success rate |
| --- | ---: | ---: |
| Language | 460/500 | 92.0% |
| Object | 483/500 | 96.6% |
| Position swap | 205/500 | 41.0% |
| Task | 152/500 | 30.4% |
| **Overall** | **1,300/2,000** | **65.0%** |

### LIBERO-Object

| Original task | Successes | Success rate |
| --- | ---: | ---: |
| Alphabet soup → basket | 50/50 | 100% |
| BBQ sauce → basket | 50/50 | 100% |
| Butter → basket | 50/50 | 100% |
| Chocolate pudding → basket | 50/50 | 100% |
| Cream cheese → basket | 50/50 | 100% |
| Ketchup → basket | 50/50 | 100% |
| Milk → basket | 50/50 | 100% |
| Orange juice → basket | 46/50 | 92% |
| Salad dressing → basket | 50/50 | 100% |
| Tomato sauce → basket | 50/50 | 100% |
| **Overall** | **496/500** | **99.2%** |

| LIBERO-PRO perturbation | Successes | Success rate |
| --- | ---: | ---: |
| Language | 493/500 | 98.6% |
| Object | 428/500 | 85.6% |
| Position swap | 50/500 | 10.0% |
| Task | 0/500 | 0.0% |
| **Overall** | **971/2,000** | **48.5%** |

### LIBERO-Goal

| Original task | Successes | Success rate |
| --- | ---: | ---: |
| Open the middle drawer of the cabinet | 46/50 | 92% |
| Put the bowl on the stove | 50/50 | 100% |
| Put the wine bottle on top of the cabinet | 44/50 | 88% |
| Open the top drawer and put the bowl inside | 49/50 | 98% |
| Put the bowl on top of the cabinet | 49/50 | 98% |
| Push the plate to the front of the stove | 50/50 | 100% |
| Put the cream cheese in the bowl | 50/50 | 100% |
| Turn on the stove | 50/50 | 100% |
| Put the bowl on the plate | 49/50 | 98% |
| Put the wine bottle on the rack | 50/50 | 100% |
| **Overall** | **487/500** | **97.4%** |

| LIBERO-PRO perturbation | Successes | Success rate |
| --- | ---: | ---: |
| Language | 489/500 | 97.8% |
| Object | 435/500 | 87.0% |
| Position swap | 118/500 | 23.6% |
| Task | 44/500 | 8.8% |
| **Overall** | **1,086/2,000** | **54.3%** |
