
```
python scripts/load_dataset.py \
--dataset-path dataset/0702_lerobot_v20 \
--embodiment-tag new_embodiment \
--video-backend torchvision_av
```


# Virtual Environment in IsaacLab


# setup ENV
source /kpfs-intern/zhekai/home/.bashrc 
conda activate gr00t

##  start finetuning

```
dataset_list=(
    "dataset/0805"
)

python scripts/gr00t_finetune.py \
--dataset-path ${dataset_list[@]} \
--output-dir checkpoints/0806_nochest \
--data-config astribot_sim_nochest \
--num-gpus 8 \
--batch-size 16 \
--embodiment-tag new_embodiment --video-backend torchvision_av
```

## start evaluation

```
python scripts/eval_policy.py \
--model-path  /home/zhekai/models/gr00t-30 \
--embodiment-tag new_embodiment \
--video-backend torchvision_av \
--dataset_path dataset/dataset_astribot_30_merged_with_mp4_v20 \
--data-config astribot_sim \
--modality-keys right_arm left_arm 
```


## start inference 
```
python scripts/inference_service.py \
--model-path  /home/zhekai/models/gr00t-30 \
--embodiment-tag new_embodiment \
--data-config astribot_sim \
--server
```


# Real world experiments
## Train
```
dataset_list=(
    "dataset/0702_lerobot_v20"
)

python scripts/gr00t_finetune.py \
--dataset-path ${dataset_list[@]} \
--output-dir checkpoints/0702_pickplace \
--data-config astribot_cartesian \
--num-gpus 8 \
--batch-size 16 \
--dataloader-num-workers 4 \
--embodiment-tag new_embodiment --video-backend torchvision_av
```

### no chest version
```
dataset_list=(
    "dataset/0702_lerobot_v20"
)

python scripts/gr00t_finetune.py \
--dataset-path ${dataset_list[@]} \
--output-dir checkpoints/0702_nochest \
--data-config astribot_no_chest_cartesian \
--num-gpus 8 \
--batch-size 16 \
--dataloader-num-workers 4 \
--embodiment-tag new_embodiment --video-backend torchvision_av
```

## Inference
```
# start  server
conda activate gr00t
python scripts/inference_service.py \
--model-path  /home/zhekai/models/0702_pickplace \
--embodiment-tag new_embodiment \
--data-config astribot_cartesian \
--server

# start clients

# see client service 
scripts/client_services.py

```