#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=plan_pusht
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=plan_pusht.out

module purge
module load 2025
module load Anaconda3/2025.06-1

cd ~/DL2---Grounding-Generated-Videos-/dino_wm

source activate dino_wm

export DATASET_DIR=/gpfs/home2/scur0196/DL2---Grounding-Generated-Videos-/dino_wm/data

WANDB_MODE=disabled python plan.py \
    --config-name plan_pusht.yaml \
    model_name=pusht \
    n_evals=1 \
    goal_H=1