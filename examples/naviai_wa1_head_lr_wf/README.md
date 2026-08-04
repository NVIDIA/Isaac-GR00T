# NAVIAI WA1 feature server

This server uses the `parcel_4f_v5.9` checkpoint and returns both decoded actions and
Speed-RL action-token features. ROS, TOPPRA, and online RL remain on the robot client.

## Confirmed checkpoint contract

- Embodiment tag: `naviai_wa1_head_lr_wf` (projector id `10`)
- Video: `head`, `left`, `right`
- State: `left_tcp`, `left_wrist_force`, `right_tcp`, `right_wrist_force`
- Action: `left_delta_tcp`, `left_pinch`, `right_delta_tcp`, `right_pinch`
- Language: `annotation.human.task_description`
- Action horizon: `40`
- Feature layer: `action_head.dit.last_block.pre_output_norm.final_denoise`
- Feature tensor: float32 `[K, 40, 1536]`
- Model id: `parcel_4f_v5.9`

## Cloud setup

Clone the feature branch and install the dGPU environment:

```bash
git clone --branch agent/export-action-features \
  https://github.com/knanxu/Isaac-GR00T.git
cd Isaac-GR00T
bash scripts/deployment/dgpu/install_deps.sh
source .venv/bin/activate
```

On `aarch64`, run `install_deps.sh` before any standalone `uv sync`. The installer retrieves
the repository's Git LFS `torchcodec` wheel and falls back to building it from source if the
LFS object is unavailable or the checkout contains only an LFS pointer.

The checkpoint processor loads assets from the gated
`nvidia/Cosmos-Reason2-2B` Hugging Face repository. Accept its access terms in the browser,
then authenticate once on the cloud machine:

```bash
hf auth login
```

Upload the complete checkpoint directory. It must contain both safetensor shards as well as
`config.json`, `processor_config.json`, `statistics.json`, `embodiment_id.json`, and
`model.safetensors.index.json`.

Create the local launch configuration and change only `MODEL_PATH` to the upload location:

```bash
cp examples/naviai_wa1_head_lr_wf/server.env.example server.local.env
sed -i 's|/absolute/path/to/parcel_4f_v5.9|/your/cloud/path/parcel_4f_v5.9|' \
  server.local.env
```

Start the server:

```bash
bash examples/naviai_wa1_head_lr_wf/launch_feature_server.sh server.local.env
```

Expected contract:

```json
{
  "version": 1,
  "model_id": "parcel_4f_v5.9",
  "layer": "action_head.dit.last_block.pre_output_norm.final_denoise",
  "feature_dim": 1536,
  "dtype": "float32"
}
```

Do not commit `server.local.env`, checkpoint files, or Hugging Face tokens. Restrict TCP port
`47866` to the robot laptop's source IP or expose it through an SSH tunnel.
