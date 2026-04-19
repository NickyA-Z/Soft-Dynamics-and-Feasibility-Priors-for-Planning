#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=gvpwm_visual_only
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:30:00
#SBATCH --output=oracle_eval_visual_only_%j.out

# Plan B: drop proprio from the dynamics residual entirely.
# Paper rho schedule (rho_init=1, growth=1.9, rho_max=1000) + mean reduction.

module purge
module load 2025
module load Anaconda3/2025.06-1

cd ~/DL2---Grounding-Generated-Videos-/reimplementation

source activate dino_wm

python -u -m src.gvpwm.examples.dino_oracle_visual_only \
    --split val \
    --horizon 25 \
    --episode-idx 2 \
    --outer-steps 25 \
    --inner-steps 25 \
    --rho-init 1.0 \
    --rho-growth 1.9 \
    --rho-max 1000.0 \
    --reduction mean
