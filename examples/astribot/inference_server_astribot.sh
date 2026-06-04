# python gr00t/eval/run_gr00t_server.py \
#     --embodiment-tag NEW_EMBODIMENT \
#     --model-path outputs/astribot/checkpoint-30000 \
#     --device cuda:0 \
#     --host 0.0.0.0 \
#     --port 5555 \
#     --strict


python examples/astribot/gr00t_inference_bridge_server.py \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path outputs/astribot_fold_cloth_better/checkpoint-30000 \
    --device cuda:0 \
    --host 0.0.0.0 \
    --port 5555 \
    --strict