#!/bin/bash
#SBATCH --job-name=finetune_libero_10
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --partition=sjw_alinlab
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00

FINETUNE_SCRIPT=examples/LIBERO/finetune_libero_10_copy.sh

cd ~/Isaac-GR00T
mkdir -p logs

echo "------------------------------------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "------------------------------------------------"

srun uv run bash $FINETUNE_SCRIPT