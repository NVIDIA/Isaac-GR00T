# Research playbook: n17es-genie-sim-manip (best 0.4774, baseline 0.358, target 1.0; from 59 experiments)
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
- (none yet)

## What regressed (do not repeat)
- [n17es-genie-sim-manip-0025 0.4676] Duplicate the latest proprioceptive state token twice before action tokens to emphasize current-frame state context while preserving history
- [n17es-genie-sim-manip-0005 0.3939] Repeat the latest visual frame group in N1.7 processor inputs so current-frame cues receive extra prompt-token salience without dropping history
- [n17es-genie-sim-manip-0029 0.3717] Drop older proprioceptive state tokens during training while always preserving the latest/current state token
- [n17es-genie-sim-manip-0033 0.3704] Add a learned current-state summary token after proprio history and before action tokens
- [n17es-genie-sim-manip-0037 0.3688] Trim fully masked padded future action tokens from the N1.7 DiT train/inference context so GENIE_SIM focuses on the valid 16-step manipulation horizon
- [n17es-genie-sim-manip-0041 0.3683] Drop older proprio history tokens during training while preserving current proprio state
- [n17es-genie-sim-manip-0013 0.3335] Replace all-state dropout with current-preserving state-history dropout so GENIE_SIM manipulation keeps reliable current proprioception while still regularizing older state tokens
- [n17es-genie-sim-manip-0057 0.3287] Apply a serialized latest-frame visual history window for GENIE_SIM training and inference
- [n17es-genie-sim-manip-0021 0.3211] Exponentially decay supervised GENIE_SIM action loss across valid action tokens to emphasize near-term/current actions while retaining the 40-token context
- [n17es-genie-sim-manip-0053 0.311] Replace full proprio dropout with 50% training-only dropout on non-current state history tokens while preserving the latest state token

## Infra failures (not hypothesis-level)
- [n17es-genie-sim-manip-0001] Train with 50% current-frame visual history dropout while preserving full-history eval -> branch_rebase

## Last 5 experiments
- iter 57 regressed score 0.3287 (history_ablation)
- iter 58 resumable_failed score None (inference_path_ablation)
- iter 59 submitted score None (recipe_tuning)
- iter 60 submitted score None (free_roam)
- iter 61 submitted score None (history_ablation)

## Current best branch
- research/auto-n17es-genie-sim-manip-1781491296-24
