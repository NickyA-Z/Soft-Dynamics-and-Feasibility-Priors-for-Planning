#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=gvpwm_planc_earlystop
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=12:00:00
#SBATCH --output=oracle_eval_full_planc_earlystop_%j.out

# Plan C + outer_steps=2 (lock in outer-1 sweet spot before rho saturates).
# Note: outer=2 means each solve is ~12x cheaper than outer=25, so total
# wall time is much smaller despite running 3 horizons.

module purge
module load 2025
module load Anaconda3/2025.06-1

cd ~/DL2---Grounding-Generated-Videos-/reimplementation

source activate dino_wm

for HORIZON in 25 50 80; do
    echo ""
    echo "================================================================"
    echo "  GVP-WM ORACLE (Plan C, outer=2) — split=val horizon=${HORIZON}"
    echo "================================================================"
    python -u -m src.gvpwm.examples.dino_oracle_full_planc_earlystop \
        --split val \
        --horizon ${HORIZON} \
        --start-index 0 \
        --num-episodes 50
done
