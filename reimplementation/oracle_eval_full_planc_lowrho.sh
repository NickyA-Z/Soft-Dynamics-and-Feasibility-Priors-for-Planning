#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=gvpwm_planc_lowrho
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=24:00:00
#SBATCH --output=oracle_eval_full_planc_lowrho_%j.out

# Plan C + rho_max=10 (penalty stays task-scale, no saturation pushoff).
# All three horizons.

module purge
module load 2025
module load Anaconda3/2025.06-1

cd ~/DL2---Grounding-Generated-Videos-/reimplementation

source activate dino_wm

for HORIZON in 25 50 80; do
    echo ""
    echo "================================================================"
    echo "  GVP-WM ORACLE (Plan C, rho_max=10) — split=val horizon=${HORIZON}"
    echo "================================================================"
    python -u -m src.gvpwm.examples.dino_oracle_full_planc_lowrho \
        --split val \
        --horizon ${HORIZON} \
        --start-index 0 \
        --num-episodes 50
done
