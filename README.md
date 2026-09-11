# Isaac GR00T 2

Choose either an agent-guided workflow or direct CLI commands. Both paths use
the same supported training, evaluation, and serving implementation.

## Agent or CLI Quickstart

> [!WARNING]
> The agentic workflows in this repository are officially supported only with
> **Codex, Claude Code, and Cursor**. Other coding agents may be able to follow the
> instructions, but are not officially supported or tested.

Start your agent or terminal in the repository root—the directory containing
this README and `pyproject.toml`—then choose the row matching your goal.

<table>
  <thead>
    <tr>
      <th scope="col">Agentic</th>
      <th scope="col">CLI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th colspan="2" scope="rowgroup">Set up GR00T and run the minimal end-to-end example</th>
    </tr>
    <tr>
      <td>
        <pre><code>Onboard me to Isaac GR00T 2 using the minimal end-to-end example. Set
up the uv environment, prepare the example dataset, and verify
training, open-loop evaluation, and the policy server.</code></pre>
      </td>
      <td>
        <pre><code class="language-bash">apt-get update
apt-get install -y build-essential ca-certificates curl git python3-dev \
  python3-venv ffmpeg yasm cmake libtool pkg-config libass-dev \
  libfreetype6-dev libvorbis-dev autoconf automake texinfo
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen --extra training

uv run --no-sync python examples/prepare_minimal_dataset.py \
  --output-dir "$HOME/gr00t2/data/gr00t_red_test"</code></pre>
        <p>Continue with the exact smoke-test commands in the
        <a href="docs/minimal_example.md">minimal end-to-end example</a>.</p>
      </td>
    </tr>
    <tr>
      <th colspan="2" scope="rowgroup">Post-train on your own dataset</th>
    </tr>
    <tr>
      <td>
        <pre><code>Use the onboarding skill to validate my setup, then post-train GR00T
on /path/to/lerobot_dataset with embodiment tag
&lt;dataset-embodiment-tag&gt;.</code></pre>
      </td>
      <td>
        <pre><code class="language-bash">uv run --no-sync python -m groot.infra.oss train \
  --dataset-path /path/to/lerobot_dataset \
  --embodiment-tag &lt;dataset-embodiment-tag&gt;</code></pre>
        <p>Add <code>--dry-run</code> to inspect the resolved command without starting
        training.</p>
      </td>
    </tr>
    <tr>
      <th colspan="2" scope="rowgroup">Evaluate a checkpoint</th>
    </tr>
    <tr>
      <td>
        <pre><code>Use the onboarding skill to run open-loop evaluation of
&lt;hf-uri-or-local-path&gt; on /path/to/lerobot_dataset with embodiment
tag &lt;dataset-embodiment-tag&gt;.</code></pre>
      </td>
      <td>
        <pre><code class="language-bash">uv run --no-sync python -m groot.infra.oss eval \
  --checkpoint &lt;hf-uri-or-local-path&gt; \
  --dataset-path /path/to/lerobot_dataset \
  --embodiment-tag &lt;dataset-embodiment-tag&gt;</code></pre>
      </td>
    </tr>
    <tr>
      <th colspan="2" scope="rowgroup">Train and evaluate in one workflow</th>
    </tr>
    <tr>
      <td>
        <pre><code>Use the onboarding skill to post-train and then evaluate GR00T on
/path/to/lerobot_dataset with embodiment tag
&lt;dataset-embodiment-tag&gt;.</code></pre>
      </td>
      <td>
        <pre><code class="language-bash">uv run --no-sync python -m groot.infra.oss train-eval \
  --dataset-path /path/to/lerobot_dataset \
  --embodiment-tag &lt;dataset-embodiment-tag&gt;</code></pre>
      </td>
    </tr>
    <tr>
      <th colspan="2" scope="rowgroup">Serve a policy</th>
    </tr>
    <tr>
      <td>
        <pre><code>Use the onboarding skill to start a policy server for
&lt;hf-uri-or-local-path&gt; with embodiment tag
&lt;dataset-embodiment-tag&gt; on port 5555.</code></pre>
      </td>
      <td>
        <pre><code class="language-bash">uv run --no-sync python -m groot.infra.oss policy-server \
  --checkpoint &lt;hf-uri-or-local-path&gt; \
  --embodiment-tag &lt;dataset-embodiment-tag&gt; \
  --host 0.0.0.0 \
  --port 5555</code></pre>
        <p>Add <code>--dry-run</code> to inspect the resolved server command without
        starting it.</p>
      </td>
    </tr>
  </tbody>
</table>

For agent-guided AutoResearch, AutoResearch visualization, AutoDebug, and FAQ
workflows, see the [agentic workflows and skills guide](skills/README.md).

## Inputs and Outputs

The [minimal end-to-end example](docs/minimal_example.md) uses a pinned, small
public LeRobot dataset to run one post-training step, a short open-loop
evaluation, and the policy server. Each workflow has a different input
boundary:

| Workflow | Required inputs | Primary output |
| --- | --- | --- |
| Prepare dataset | Network access to the public dataset; writable dataset path | Local LeRobot dataset |
| Post-train | Local dataset, embodiment tag, accessible pretrain/model artifacts, GPU, writable output path | Local checkpoint |
| Open-loop eval | Local dataset, matching embodiment tag, local checkpoint or accessible HF URI, GPU | JSON evaluation report |
| Closed-loop eval | Simulator backend plugin, suite assets, matching embodiment tag, checkpoint, GPU | Success-rate report and optional videos |
| Policy server | Matching embodiment tag, local checkpoint or accessible HF URI, GPU | TCP policy endpoint |
| Train and eval | All post-training and open-loop evaluation inputs; optional approved benchmark recipe | Checkpoint, evaluation JSON, and optional benchmark report |

See the detailed guide for the exact smoke-test environment variables and
command-line arguments. Training requires an explicit `--embodiment-tag` and
either `--dataset-path` or an intentional replacement of the complete
`data.datasets` list with `--set`.

## Reproducible Benchmark Reports

Add `--benchmark-recipe` to train once, evaluate that checkpoint, and connect the
inputs and results in `benchmark-report.json`. The recipe selects open-loop
evaluation, closed-loop simulation, or both, using the existing evaluators.

The bundled `libero_10_n2` recipe follows the LIBERO-10 onboarding example below.
It requires eight GPUs, access to the pinned starting checkpoint, and the existing
simulator setup:

```bash
groot/infra/closed_loop/backends/libero_pro/setup.sh
uv run --no-sync python -m groot.infra.oss train-eval \
  --benchmark-recipe libero_10_n2 --output-dir /tmp/n2-libero-benchmark
```

Use `--dry-run` first to inspect the recipe without downloads, dataset preparation,
GPU inspection, training, or evaluation. Benchmark runs require a clean Git
checkout; keep outputs outside it or in an ignored directory.

Here is the complete bundled recipe. You can also pass a JSON file directly:

```json
{
  "schema_version": "groot-n2-benchmark-spec-v1",
  "benchmark": {
    "recipe_id": "libero_10_n2",
    "recipe_version": "1",
    "provenance": "OSS LIBERO-10 onboarding: oss/README.md and n2_oss_libero_10"
  },
  "dataset": {
    "repo_id": "IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot",
    "revision": "e1a223d30b896c1613f270a2bfc63d382b3de7e1"
  },
  "base_checkpoint": {
    "uri": "hf://model/nvidia/GR00T2Internal@4fe662abaf302ee46916881c447377da8c4bd15a/base_200k",
    "revision": "4fe662abaf302ee46916881c447377da8c4bd15a"
  },
  "training": {
    "experiment": "n2_oss_libero_10", "compute_type": "h100",
    "embodiment_tag": "libero_sim", "seed": 24,
    "max_steps": 10000, "num_gpus": 8, "batch_size": 4,
    "config_overrides": []
  },
  "evaluation": {
    "closed_loop": {
      "backend": "groot.infra.closed_loop.backends.libero_pro.backend:LiberoProBackend",
      "suite": "libero_pro_10", "episodes_per_task": 50, "seed": 0,
      "max_episode_steps": 400, "action_steps": 8, "denoising_steps": 4
    }
  },
  "expectations": {}
}
```

This uses the OSS CLI's pinned base checkpoint. Reproducing the published score
through this wrapper remains to be validated. Empty `expectations` records results
as `completed`, without claiming a performance pass. Agreed ranges specify
`minimum`, `maximum`, and `provenance` for metrics such as
`evaluation.closed_loop.success_rate` (a fraction from 0 to 1). Missing or
non-finite results, incomplete training, failed ranges, and execution failures
exit nonzero. Failure reports retain the stage, completed results, and artifacts.

For open-loop evaluation, add this block under `evaluation.open_loop`, alongside
`closed_loop` or on its own:

```json
{
  "trajectory_ids": [0], "steps": 200, "action_horizon": 30,
  "denoising_steps": 4, "video_backend": "torchcodec",
  "local_only": true, "server_parity_tolerance": 0.01
}
```

The recipe owns all experiment settings, including the dataset and checkpoint.
Repeated command-line settings are rejected even when they match. Only
`--output-dir`, `--dry-run`, and open-loop `--output-json` are operational options.
Ordinary `train-eval` keeps its existing behavior.

Remote dataset files are checked against the pinned Hub revision. For local data,
use `{"path": "/path/to/dataset", "sha256": "<expected fingerprint>"}` instead.
Relative dataset paths resolve beside the recipe file. Compute a fingerprint with:

```bash
uv run python -c 'from pathlib import Path; from groot.infra.benchmark.dataset import dataset_fingerprint; print(dataset_fingerprint(Path("/path/to/dataset")))'
```

The fingerprint covers sorted relative filenames, sizes, and SHA-256 file hashes,
excluding root `.git` and `.cache` bookkeeping. Each run verifies the source,
copies metadata into its output dataset, and links the large data files. Existing
preparation code operates on that view; the report identifies the prepared
metadata. Keep source data unchanged while a run is in progress.

## LIBERO-10 post-training recipe

Use `libero_10_n2` to post-train the existing `n2_oss_libero_10` experiment and
evaluate its final checkpoint on the ten LIBERO tasks. The recipe downloads and
verifies the pinned public dataset, prepares its metadata, and starts training
from the pinned `base_200k` checkpoint.

After completing the [simulator setup](#libero-and-libero-pro), run:

```bash
uv run --no-sync python -m groot.infra.oss train-eval \
  --benchmark-recipe libero_10_n2 \
  --output-dir ./outputs/n2-libero-10
```

The experiment config preserves the validated LIBERO-10 data window, model
head, optimizer, and global batch of 384. Its default topology is one eight-GPU
node with batch 4 per GPU and gradient accumulation 12.

Add `--dry-run` to inspect the recipe and resolved commands without downloading
data, preparing metadata, training, or evaluating.

The final checkpoint is written to
`./outputs/n2-libero-10/checkpoint-10000` and evaluated with 50 episodes per task.
The combined `./outputs/n2-libero-10/benchmark-report.json` records the inputs,
checkpoint, and results. See [Reproducible Benchmark Reports](#reproducible-benchmark-reports)
for the complete recipe and report details.

## Closed-loop Evaluation

Run the N2 policy against any simulator plugin that implements the shared backend contract:

```bash
uv run --no-sync python -m groot.infra.oss closed-loop-eval \
  --checkpoint <hf-uri-or-local-path> \
  --backend <python-module:BackendClass> \
  --suite <suite-name> \
  --embodiment-tag <checkpoint-embodiment-tag> \
  --output-dir ./benchmark-results
```

Without `--ray-address`, the command starts local Ray. Pass a Ray client or GCS
address to use an existing cluster. A run writes its resolved request,
success-rate report, comparison history, and optional videos below the output
directory. Simulator-specific setup and supported suites are documented by
each backend.

### LIBERO and LIBERO-PRO

Install the pinned simulator and assets once:

```bash
groot/infra/closed_loop/backends/libero_pro/setup.sh
```

Run the original ten LIBERO tasks with 50 episodes per task:

```bash
uv run --no-sync python -m groot.infra.oss closed-loop-eval \
  --checkpoint ./outputs/n2-libero-10/checkpoint-10000 \
  --backend groot.infra.closed_loop.backends.libero_pro.backend:LiberoProBackend \
  --suite libero_pro_10 \
  --embodiment-tag libero_sim \
  --episodes-per-task 50 \
  --action-steps 8 \
  --max-episode-steps 400 \
  --output-dir ./benchmark-results/libero-10
```

Run all 40 LIBERO-PRO perturbation tasks—ten each for object, position swap,
language, and task—with 50 episodes per task:

```bash
uv run --no-sync python -m groot.infra.oss closed-loop-eval \
  --checkpoint ./outputs/n2-libero-10/checkpoint-10000 \
  --backend groot.infra.closed_loop.backends.libero_pro.backend:LiberoProBackend \
  --suite libero_pro_10_all \
  --embodiment-tag libero_sim \
  --episodes-per-task 50 \
  --action-steps 8 \
  --max-episode-steps 400 \
  --output-dir ./benchmark-results/libero-pro-10
```

The suite names resolve checked-in task YAMLs. Without `--ray-address`, each
command uses the local GPUs visible to Ray. Pass an existing Ray address to
distribute the sweep, or use `--tasks` and fewer episodes for a smoke test.

The OSS public-data checkpoint at step 10,000 produced the following results.
LIBERO-10 uses 50 episodes for each of its ten tasks:

| LIBERO-10 result | Episodes | Successes | Success rate |
| --- | ---: | ---: | ---: |
| **Overall** | **500** | **470** | **94.0%** |

LIBERO-PRO uses 50 episodes for each of ten tasks on every perturbation axis:

| LIBERO-PRO perturbation | Episodes | Successes | Success rate |
| --- | ---: | ---: | ---: |
| Object | 500 | 341 | 68.2% |
| Position swap | 500 | 71 | 14.2% |
| Language | 500 | 476 | 95.2% |
| Task | 500 | 107 | 21.4% |
| **Overall** | **2,000** | **995** | **49.8%** |

### RoboCasa365

Install the pinned simulator checkout, then run an approved shared suite:

```bash
groot/infra/closed_loop/backends/robocasa365/setup.sh

uv run --no-sync python -m groot.infra.oss closed-loop-eval \
  --checkpoint <hf-uri-or-local-path> \
  --backend groot.infra.closed_loop.backends.robocasa365.backend:RoboCasa365Backend \
  --suite robocasa365_target50 \
  --embodiment-tag robocasa365_panda_omron
```

The backend uses the same per-task horizons as internal evaluation. Set
`--backend-option robocasa_root='"/path/to/robocasa"'` to use an existing
checkout instead of the setup script's default.

### RoboDojo

RoboDojo requires Linux, an NVIDIA GPU, and acceptance of the Isaac Sim EULA.
Install its pinned runtime, then select one of the approved ARX X5 suites:

```bash
groot/infra/closed_loop/backends/robodojo/setup.sh

uv run --no-sync python -m groot.infra.oss closed-loop-eval \
  --checkpoint <hf-uri-or-local-path> \
  --backend groot.infra.closed_loop.backends.robodojo.backend:RoboDojoBackend \
  --suite robodojo_arx_x5_generalization \
  --embodiment-tag arx_x5
```

The shared actor automatically replaces unstable layout seeds while preserving
the requested episode count.

## Model Artifacts

By default the training entrypoint loads GR00T 2 model dependencies from
Hugging Face:

- `hf://model/nvidia/Cosmos-Reason1-7B@375e24000b24baed78f4618d3dd779e47cd96323`
- `hf://model/nvidia/Cosmos-Predict2.5-2B@0d37c7498f54cee3c599d438d895a0a4a8608064/transformer`
- `hf://model/Wan-AI/Wan2.1-I2V-14B-480P@6b73f84e66371cdfe870c72acd6826e1d61cf279/Wan2.1_VAE.pth`
- `hf://model/nvidia/GR00T2Internal@4fe662abaf302ee46916881c447377da8c4bd15a/base_200k`

Pass `--pretrain-checkpoint` if you want to override the default pretrain
checkpoint with another `hf://model/...` URI or a local checkpoint path.
The default `base_200k` checkpoint and training configuration both retain
all 32 embodiment-specific robot heads. The dataset's `--embodiment-tag`
selects the head used by each training example.

Set `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, or `HUGGINGFACE_TOKEN` when training
or evaluation needs to download a private Hugging Face checkpoint or model
dependency.
