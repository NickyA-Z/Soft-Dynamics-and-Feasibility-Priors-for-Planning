#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=oracle_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=3:00:00
#SBATCH --output=oracle_eval_a100_%j.out

module purge
module load 2025
module load Anaconda3/2025.06-1

cd ~/DL2---Grounding-Generated-Videos-/reimplementation

source activate dino_wm

python -u -m src.gvpwm.examples.dino_oracle_demo \
    --num-episodes 10 \
    --grad-steps 200 \
    --lr 0.05 \
    --lambda-goal 10.0 \
    --lambda-video 1.0 \
    --lambda-action 0.05
