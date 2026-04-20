#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=gvpwm_full_planc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=12:00:00
#SBATCH --output=oracle_eval_full_planc_%j.out

# Full-split oracle eval under Plan C (proprio-freezing bug fix).
# Patches DinoWorldModelAdapter.initialize_latents_from_video globally
# and overrides build_planner to use paper ALM schedule + mean residual.

module purge
module load 2025
module load Anaconda3/2025.06-1

cd ~/DL2---Grounding-Generated-Videos-/reimplementation

source activate dino_wm

python -u -m src.gvpwm.examples.dino_oracle_full_planc \
    --split val \
    --horizon 25 \
    --start-index 0 \
    --num-episodes 50
