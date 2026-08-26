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
apt-get install -y ffmpeg
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
| Train and eval | All post-training and open-loop evaluation inputs | Checkpoint and JSON report |

See the detailed guide for the exact smoke-test environment variables and
command-line arguments. Training requires an explicit `--embodiment-tag` and
either `--dataset-path` or an intentional replacement of the complete
`data.datasets` list with `--set`.

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

### LIBERO-PRO

Install the pinned simulator and assets, then select its shared backend:

```bash
groot/infra/closed_loop/backends/libero_pro/setup.sh

uv run --no-sync python -m groot.infra.oss closed-loop-eval \
  --checkpoint <hf-uri-or-local-path> \
  --backend groot.infra.closed_loop.backends.libero_pro.backend:LiberoProBackend \
  --suite libero_pro_10_all \
  --embodiment-tag libero_sim
```

The suite name resolves the same checked-in task YAML used by internal
evaluation. Use `--tasks` for a subset and `--episodes-per-task` for a smoke
run.

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
