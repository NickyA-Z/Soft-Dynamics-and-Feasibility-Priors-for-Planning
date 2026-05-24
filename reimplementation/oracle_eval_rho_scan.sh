#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=gvpwm_rho_scan
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=oracle_eval_rho_scan_%j.out

# rho_init scan over decades on a single episode (no MPC env step).
# Goal: find a rho regime where goal_loss can actually decrease.

module purge
module load 2025
module load Anaconda3/2025.06-1

cd ~/DL2---Grounding-Generated-Videos-/reimplementation

source activate dino_wm

python -u -m src.gvpwm.examples.dino_oracle_rho_scan \
    --split val \
    --horizon 25 \
    --episode-idx 2 \
    --rho-init-list 1e-6 1e-5 1e-4 1e-3 \
    --inner-steps 25 \
    --outer-steps 15
