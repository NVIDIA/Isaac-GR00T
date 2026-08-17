# Research playbook: n17es-genie-sim-if (best 0.6919, baseline 0.5957, target 1.0; from 53 experiments)
# Filtered to idea_type: history_ablation

## Priors
- The sheet scores are cross-task directional priors, not stack-bowls scores.
- N1.7ES with unified inference stack reached 0.93-0.94 partial score elsewhere.
- N1.7ES + LWM reached 0.98 partial score elsewhere, the strongest prior in the sheet.
- N1.7ES + TTT reached 0.885 partial score elsewhere.
- Minimal WAM +1 latent history helped, but disjoint/full attention variants regressed.
- Scratch N1.7ES and DC-2B scratch were weak; prefer fine-tuning-friendly changes.
- N2 planning calls out history encoding ablation from separate past/current/future chunks toward explicit current-frame emphasis: [-20,-15,-10,-5], [0], [5,10,15,20].
- N2 planning calls out history dropout as easier under explicit history/current grouping.
- N2 planning calls out chunk-wise autoregressive inference as a possible quality/speed tradeoff for 40-step tasks.
- N2 carryover work suspects train/inference path mismatch around unified inference and MinWAM-TTT; prefer bounded checks or switches inside training/inference-side code.

## What worked (build on these)
- [n17es-genie-sim-if-0005 +0.0338] Bias GENIE_SIM IF visual history cropping 60% toward current-frame-only samples while preserving occasional longer histories

## What regressed (do not repeat)
- [n17es-genie-sim-if-0037 0.6819] Synchronize GENIE_SIM IF random visual-history suffix cropping with proprioceptive state-history cropping
- [n17es-genie-sim-if-0045 0.664] Add endpoint-preserving history crop that keeps first/latest visual frames and masks unselected state history
- [n17es-genie-sim-if-0013 0.6581] Synchronize proprioceptive state-history suffix cropping with the existing current-biased visual crop
- [n17es-genie-sim-if-0029 0.6524] Thin GENIE_SIM IF training visual history with a stride-2 selection that always preserves the latest frame
- [n17es-genie-sim-if-0021 0.6483] Replace blanket proprioceptive state dropout with current-preserving history-state dropout during GENIE_SIM IF finetuning
- [n17es-genie-sim-if-0017 0.6466] Add deterministic visual-history frame striding after the current-biased suffix crop
- [n17es-genie-sim-if-0009 0.6455] Add 50% visual-history frame dropout after the existing suffix crop, always preserving the latest frame
- [n17es-genie-sim-if-0033 0.6393] Add stochastic visual-history dropout during GENIE_SIM IF fine-tuning, preserving the latest frame while thinning stale frames
- [n17es-genie-sim-if-0041 0.6315] Append one extra copy of each latest camera frame after the sampled visual history for GENIE_SIM IF fine-tuning and saved inference processors

## Infra failures (not hypothesis-level)
- [n17es-genie-sim-if-0001] Repeat the latest visual frame in N1.7 GENIE_SIM history so the VLM/action head emphasize the current observation -> It significantly increased the training time cost and reached the 10 hour timeout limit.
- [n17es-genie-sim-if-0025] Preserve the two most recent visual frames under the existing current-biased random history crop -> eval

## Last 5 experiments
- iter 49 submitted score None (history_ablation)
- iter 50 submitted score None (inference_path_ablation)
- iter 51 fatal_failed score None (recipe_tuning)
- iter 52 submitted score None (free_roam)
- iter 53 submitted score None (history_ablation)

## Current best branch
- research/auto-n17es-genie-sim-if-1781809036-36
