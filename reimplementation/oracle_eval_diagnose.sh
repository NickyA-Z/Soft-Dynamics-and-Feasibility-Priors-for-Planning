#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=gvpwm_diagnose
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:30:00
#SBATCH --output=oracle_eval_diagnose_%j.out

# Diagnostic experiments to disentangle ALM warm-start vs residual-scale issues.
# Single episode, no MPC env-stepping. Should finish in a few minutes.

module purge
module load 2025
module load Anaconda3/2025.06-1

cd ~/DL2---Grounding-Generated-Videos-/reimplementation

source activate dino_wm

python -u -m src.gvpwm.examples.dino_oracle_diagnose \
    --split val \
    --horizon 25 \
    --episode-idx 2
