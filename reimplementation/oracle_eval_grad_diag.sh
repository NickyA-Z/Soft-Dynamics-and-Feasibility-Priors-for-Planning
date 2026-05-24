#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=gvpwm_grad_diag
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --output=oracle_eval_grad_diag_%j.out

# Step 3: gradient-norm diagnostic. Checks whether latent_parameter is
# actually getting updated (per-element grad large enough), or pinned by
# Adam's small effective step over 77k elements.

module purge
module load 2025
module load Anaconda3/2025.06-1

cd ~/DL2---Grounding-Generated-Videos-/reimplementation

source activate dino_wm

python -u -m src.gvpwm.examples.dino_oracle_grad_diag \
    --split val \
    --horizon 25 \
    --episode-idx 2 \
    --outer-steps 4 \
    --inner-steps 25 \
    --rho-init 1.0 \
    --rho-growth 1.9 \
    --rho-max 1000.0 \
    --reduction mean
