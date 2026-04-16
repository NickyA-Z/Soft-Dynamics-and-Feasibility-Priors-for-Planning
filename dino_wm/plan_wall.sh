#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=plan_wall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=plan_wall.out

module purge
module load 2025
module load Anaconda3/2025.06-1

cd ~/DL2---Grounding-Generated-Videos-/dino_wm

source activate dino_wm

export DATASET_DIR=/gpfs/home2/scur0196/DL2---Grounding-Generated-Videos-/dino_wm/data

WANDB_MODE=disabled python plan.py \
    --config-name plan_wall.yaml \
    model_name=wall_single \
    n_evals=1 \
    goal_H=1