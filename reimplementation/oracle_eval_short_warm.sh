#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=gvpwm_short_warm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:20:00
#SBATCH --output=oracle_eval_short_warm_%j.out

# 1+2: early-stop (outer=2, no rho ramp) + expert action warm-start.
# Direct test of "is ALM viable here at all".

module purge
module load 2025
module load Anaconda3/2025.06-1

cd ~/DL2---Grounding-Generated-Videos-/reimplementation

source activate dino_wm

python -u -m src.gvpwm.examples.dino_oracle_short_warm \
    --split val \
    --horizon 25 \
    --episode-idx 2 \
    --outer-steps 2 \
    --inner-steps 25 \
    --rho-init 1.0 \
    --rho-growth 1.0 \
    --rho-max 1.0 \
    --reduction mean
