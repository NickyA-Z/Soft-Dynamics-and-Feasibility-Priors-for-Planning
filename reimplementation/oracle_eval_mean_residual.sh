#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=gvpwm_mean_resid
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=oracle_eval_mean_residual_%j.out

# Single-episode ALM solve with mean-reduced dynamics penalty.
# Uses paper hyperparameters (rho_init=1, rho_growth=1.9, rho_max=1000)
# but penalty is 0.5*rho*mean(L^2) instead of 0.5*rho*sum(L^2).

module purge
module load 2025
module load Anaconda3/2025.06-1

cd ~/DL2---Grounding-Generated-Videos-/reimplementation

source activate dino_wm

python -u -m src.gvpwm.examples.dino_oracle_mean_residual \
    --split val \
    --horizon 25 \
    --episode-idx 2
