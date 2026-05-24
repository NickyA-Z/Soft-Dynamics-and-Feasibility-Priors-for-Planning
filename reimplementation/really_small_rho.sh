#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=gvpwm_rho_ablation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=really_small_rho_oracle_eval_rho_ablation_%j.out

# rho_max ablation: paper uses 1000; we suspect that crushes goal/video terms
# in our latent scale. Try rho_max=10 on a few episodes.

module purge
module load 2025
module load Anaconda3/2025.06-1

cd ~/DL2---Grounding-Generated-Videos-/reimplementation

source activate dino_wm

python -u -m src.gvpwm.examples.dino_oracle_rho_ablation \
    --split val \
    --horizon 25 \
    --num-episodes 3 \
    --rho-max 1
    --rho-init 1e-5
