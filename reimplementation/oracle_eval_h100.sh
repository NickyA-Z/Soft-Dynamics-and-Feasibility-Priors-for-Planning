#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=oracle_h100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --output=oracle_eval_h100_%j.out

module purge
module load 2025
module load Anaconda3/2025.06-1

cd ~/DL2---Grounding-Generated-Videos-/reimplementation

source activate dino_wm

python -u -m src.gvpwm.examples.dino_oracle_demo \
    --num-episodes 1 \
    --grad-steps 50 \
    --lr 0.05 \
    --lambda-goal 10.0 \
    --lambda-video 1.0 \
    --lambda-action 0.05
