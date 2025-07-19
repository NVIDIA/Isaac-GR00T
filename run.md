

# Virtual Environment in IsaacLab


# setup ENV
source /kpfs-intern/zhekai/home/.bashrc 
conda activate gr00t




##  start finetuning

```
dataset_list=(
    "dataset/dataset_astribot_30_merged_with_mp4_v20"
)

python scripts/gr00t_finetune.py \
--dataset-path ${dataset_list[@]} \
--output-dir checkpoints/cylinder \
--data-config astribot_sim \
--num-gpus 8 \
--batch-size 16 \
--embodiment-tag new_embodiment --video-backend torchvision_av
```

## start evaluation


## start inference 