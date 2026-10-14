# Agent-guided feature checklist

Phase 1 uses the README and linked user guides as its feature inventory. Start
from a plain-language goal, read its guide, and use the documented interface.
The [onboarding skill](../skills/onboarding/SKILL.md) supplies orientation and
routing; specialized skills remain for research and debugging. A separate skill
per CLI command is unnecessary when the guide and parser source explain the interface.

## Feature inventory

Commands below run from the OSS package root. `oss <command>` is shorthand in
this table for `uv run python -m groot.infra.oss <command>`. Success checks are
runtime goals, not claims that those goals have already been verified here.

| User goal / example request | Guide and callable interface | Required inputs | Observable success |
| --- | --- | --- | --- |
| “Show me a menu of the current CLI commands.” / `$onboarding` | [CLI onboarding](../skills/onboarding/SKILL.md), `oss.py` parser source and imported helpers, read as text | Repository files only; no Python, uv, GPU, or credentials | Agent generates choices from actual subcommands, shows debug/research as separate agent workflows, accepts number/name/goal, and carries existing context to the selected skill; no workload launches just to show the menu |
| “Explain the key features and help me choose what to try.” | [README](../README.md), [onboarding](../skills/onboarding/SKILL.md) | User's interests; no GPU or credentials | Explains workflows, maps relevant directories, gives a suitable next step and guide; launches nothing for a tour |
| “Set up training / inference on this machine.” | [Installation](installation.md), platform-specific `uv sync` and dependency checks | OS, architecture, GPU, selected environment | Selected environment imports required modules and reports an available CUDA GPU |
| “Prepare the small example dataset.” | [Minimal example](minimal_example.md), `uv run python examples/prepare_minimal_dataset.py --output-dir <path>` | Network and writable storage | Pinned three-episode SO100 dataset and GR00T metadata exist at the reported path |
| “Verify the complete first-run example.” | [Minimal example](minimal_example.md), onboarding execution walkthrough | Training environment, H100-class 80 GB GPU, model access and writable paths | Optimizer completes step 1, `checkpoint-1` exists, eval inference writes JSON, and policy client ping succeeds; smoke server stopped |
| “Post-train on my demonstrations.” | [README](../README.md#inputs-and-outputs), `oss train` | Local LeRobot data, matching embodiment, model access, GPU, output directory | Requested optimizer steps complete and final checkpoint exists; report its path |
| “Use a different starting checkpoint or change training settings.” | [Model artifacts](../README.md#model-artifacts), `oss train --pretrain-checkpoint ... --set ...` | Accessible compatible checkpoint, dataset contract and intended overrides | Dry-run resolves intended settings; real run produces a checkpoint using those settings |
| “Evaluate predicted actions on my dataset.” | [Minimal example](minimal_example.md#5-run-open-loop-evaluation), `oss eval` | Checkpoint, local dataset, matching embodiment, inference environment, GPU | Inference completes and requested JSON contains evaluation metrics |
| “Train and evaluate the resulting checkpoint.” | [Minimal example](minimal_example.md#7-run-training-and-evaluation-together), `oss train-eval` | Training and open-loop inputs | Eval uses the newly trained final checkpoint and writes JSON |
| “Serve my policy to a client.” | [Minimal example](minimal_example.md#6-start-the-policy-server), `oss policy-server`; `PolicyClient` | Checkpoint, matching embodiment, inference environment, GPU, available port | Endpoint becomes ready and `PolicyClient.ping()` succeeds; track the server process and stop it when the smoke is done |
| “Measure task success in simulation.” | [Closed-loop evaluation](../README.md#closed-loop-evaluation), `oss closed-loop-eval` | Installed backend, suite assets, matching checkpoint/embodiment, GPU, episodes and output path; optional Ray address | Requested episodes complete; report includes successes, episodes, per-task results and optional videos |
| “Evaluate LIBERO or LIBERO-PRO.” | [LIBERO guide](libero_family_benchmarks.md), LIBERO setup script and `oss closed-loop-eval` | Pinned simulator and PRO assets, selected suite, matching checkpoint | Requested tasks/episodes complete and success-rate report identifies the suite |
| “Evaluate RoboCasa365.” | [RoboCasa365](../README.md#robocasa365), backend `setup.sh`, `oss closed-loop-eval` | Pinned simulator checkout, approved suite, compatible checkpoint and GPU | Report covers requested episodes using the backend's task horizons |
| “Evaluate RoboDojo.” | [RoboDojo](../README.md#robodojo), backend `setup.sh`, `oss closed-loop-eval` | Linux, NVIDIA GPU, user acceptance of Isaac Sim EULA, pinned runtime and ARX X5 suite | Requested episode count is retained, including replacement of unstable layout seeds; report exists |
| “Compare success at different step budgets.” | [Budget metrics](../README.md#closed-loop-evaluation), `--success-step-budgets`, optional `--max-episode-steps` | Common tasks, episode limits and budget horizon for comparison | Summary/per-task results include budget success rates and AUC; failures stay in denominator |
| “Reproduce a LIBERO training benchmark.” | [Benchmark recipes](libero_family_benchmarks.md#benchmark-recipes), `oss train-eval --benchmark-recipe <name>` | Clean checkout, eight H100 GPUs for documented runs, pinned data/model access, external outputs | Final checkpoint and `benchmark-report.json` identify training and evaluation inputs/results |
| “Record a reproducible benchmark with agreed metric ranges.” | [Benchmark reports](../README.md#reproducible-benchmark-reports), recipe JSON and `--benchmark-recipe` | Recipe-owned settings, pinned dataset revision or local fingerprint, clean source | Report records completed stages and artifacts; agreed ranges pass or command exits nonzero; empty expectations mean completed only |
| “Build or reuse TensorRT engines.” | [Deployment](../scripts/deployment/README.md), `oss build-trt`, or `--enable-trt` / `--trt-engine-dir` on inference | Compatible checkpoint, TensorRT environment, target GPU, matching codec/embodiment; external cache | Build prints `TRT_ENGINE_DIR`; manifest authenticates engines/runtime/checkpoint; accelerated inference succeeds |
| “Use CUDA graph replay with TensorRT.” | [Deployment](../scripts/deployment/README.md#use-an-explicitly-managed-engine-bundle), `--use-cuda-graph` | Valid bundle and fixed input shapes; real-shaped client warmup for serving | Warmup reports ready and subsequent inference succeeds; measure performance on target hardware |
| “Deploy on Jetson Thor.” | [Thor installation](installation.md#jetson-thor), install/activation scripts, common deployment CLI | Jetson Thor, required software, checkpoint and GPU-local engines | Locked environment checks pass; engine validation and policy client request succeed on Thor |
| “Prepare an AutoResearch baseline.” | [Baseline preparation skill](../skills/prepare-autoresearch-baseline/SKILL.md) | Existing train/eval workflow, metric, paths, authorized compute | Working wrappers and baseline artifacts produce a validated `autoresearch.yaml` |
| “Search for a better post-training recipe.” | [AutoResearch](../skills/autoresearch/README.md), `autoresearch` agent | Config, working wrappers, existing baseline, trial budget and execution authorization | Durable `state.json` records trial results and the best score or actionable failure |
| “Visualize my research study.” | [Visualization skill](../skills/visualize-autoresearch/SKILL.md), `python3 skills/visualize-autoresearch/scripts/render.py <state.json> --output <html>` | Existing study state and output path | Standalone HTML renders trial lineage/metrics without modifying state |
| “Investigate a repository failure.” | [AutoDebug](../.agents/autodebug.md), `autodebug` agent | Bug report, evidence, clean checkout and any granted compute | Recorded defended root cause or explicit inconclusive result; fix verification when implementation is requested |
| “Understand a data contract or recorded research finding.” | [Findings](findings.md), [research notes](research-notes.md) | User's question and relevant contract | Explanation cites evidence and its limits; historical findings do not substitute for runtime validation |

## Baseline validation and remaining work

The October 12, 2026 local audit reads the README, installation, minimal example,
LIBERO, deployment, and agent workflow guides. It uses existing docs and CLI
interfaces before extending onboarding; no new training or deployment wrapper
is introduced. Codex and Claude skill links and Cursor skill links resolve to
`skills/`; Claude's `CLAUDE.md` links to `AGENTS.md`. Cursor discovers the
same onboarding skill through its repository skill link.

CPU checks cover CLI help, command resolution for train, eval, train-eval,
policy-server and three simulator backends, all four LIBERO benchmark recipe
dry runs in a clean temporary export, and the existing OSS CLI/recipe test suite. These checks verify
reachability and configuration, not model quality, engine correctness, or a
complete agent-driven run. Keep dry-run outputs outside the checkout.

Validation commands (CPU only):

```bash
# From the OSS package root, with the development environment already installed
python -m pytest tests/groot/infra/test_oss.py \
  tests/groot/infra/benchmark/test_bundled_specs.py -q
# 52 passed
```

The source-repository export-boundary suite also passes (22 tests). Direct CLI
probes resolve all six subcommand help pages, four ordinary workflow dry runs,
three simulator-backend dry runs, and four LIBERO recipe dry runs. Recipe dry
runs require a clean Git tree too; run them from a clean checkout with outputs
outside it. Skill frontmatter validation passes. Skill discovery links were
checked in the clean exported package; ignored stale cache directories in a
working checkout can affect the source assembly check.

Full execution still requires the listed hardware, artifacts, simulator setup,
and user inputs. The local audit runs on macOS without a CUDA GPU; installation,
model inference, training, simulator episodes, TensorRT builds, Thor deployment,
and live client requests are not verified here. Cross-harness behavioral trials
with Codex, Claude Code, and Cursor also remain to be recorded before claiming
all documented features work autonomously.

When a documented feature changes, update its row and the corresponding guide.
Record the actual command, environment, artifact paths, result, and first failure
for execution trials. This inventory supplies the user tasks and observable
success criteria for a future goal-based harness; Phase 2 automation is separate.
