#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=gvpwm_wm_sanity
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:20:00
#SBATCH --output=oracle_eval_wm_sanity_%j.out

# WM rollout sanity check: teacher-forcing vs padded-init vs real-history-init.
# Disambiguates whether latent explosion is WM checkpoint, adapter padding,
# or autoregressive instability.

module purge
module load 2025
module load Anaconda3/2025.06-1

cd ~/DL2---Grounding-Generated-Videos-/reimplementation

source activate dino_wm

python -u -m src.gvpwm.examples.wm_rollout_sanity \
    --split val \
    --horizon 25 \
    --episode-idx 2
