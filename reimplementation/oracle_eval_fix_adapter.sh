#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=gvpwm_fix_adapter
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=oracle_eval_fix_adapter_%j.out

# Plan A: monkey-patch initialize_latents_from_video so frames 1..H keep
# the video's natural proprio (frame 0 still uses current_latent).
# Paper rho schedule (rho_init=1, growth=1.9, rho_max=1000) + mean reduction.

module purge
module load 2025
module load Anaconda3/2025.06-1

cd ~/DL2---Grounding-Generated-Videos-/reimplementation

source activate dino_wm

python -u -m src.gvpwm.examples.dino_oracle_fix_adapter \
    --split val \
    --horizon 25 \
    --episode-idx 2 \
    --outer-steps 25 \
    --inner-steps 25 \
    --rho-init 1.0 \
    --rho-growth 1.9 \
    --rho-max 1000.0 \
    --reduction mean
