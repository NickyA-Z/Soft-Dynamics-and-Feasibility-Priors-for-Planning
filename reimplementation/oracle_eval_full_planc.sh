#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=gvpwm_oracle_planc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=24:00:00
#SBATCH --output=oracle_eval_full_planc_%j.out

# Paper reproduction: PushT GVP-WM (ORACLE) with Plan C fix
# (proprio-freezing bug in initialize_latents_from_video patched out).
# Runs val split, 50 episodes, all three horizons (25/50/80) — same
# protocol as Table in the GVP-WM paper.

module purge
module load 2025
module load Anaconda3/2025.06-1

cd ~/DL2---Grounding-Generated-Videos-/reimplementation

source activate dino_wm

for HORIZON in 25 50 80; do
    echo ""
    echo "================================================================"
    echo "  GVP-WM ORACLE (Plan C fix) — split=val horizon=${HORIZON}"
    echo "================================================================"
    python -u -m src.gvpwm.examples.dino_oracle_full_planc \
        --split val \
        --horizon ${HORIZON} \
        --start-index 0 \
        --num-episodes 50
done
