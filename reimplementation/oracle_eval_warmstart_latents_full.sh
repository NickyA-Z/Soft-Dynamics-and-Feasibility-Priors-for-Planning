#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=gvpwm_warm_lat_full
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=oracle_eval_warmstart_latents_full_%j.out

# Full oracle evaluation using the warm_start_latents MPC init bypass.
# Defaults mirror oracle_eval_paper.sh, but use:
#   - initial latent warm start from encoded video
#   - mean residual reduction
#
# Override defaults via:
#   sbatch --export=ALL,HORIZON=50,NUM_EPISODES=20 oracle_eval_warmstart_latents_full.sh

module purge
module load 2025
module load Anaconda3/2025.06-1

cd ~/DL2---Grounding-Generated-Videos-/reimplementation

source activate dino_wm

SPLIT=${SPLIT:-val}
HORIZON=${HORIZON:-25}
START_INDEX=${START_INDEX:-0}
NUM_EPISODES=${NUM_EPISODES:-50}
OUTER_STEPS=${OUTER_STEPS:-25}
INNER_STEPS=${INNER_STEPS:-25}
RHO_INIT=${RHO_INIT:-1.0}
RHO_GROWTH=${RHO_GROWTH:-1.9}
RHO_MAX=${RHO_MAX:-1000.0}
LR=${LR:-0.05}
REDUCTION=${REDUCTION:-mean}

python -u -m src.gvpwm.examples.dino_oracle_warmstart_latents_full \
    --split "${SPLIT}" \
    --horizon "${HORIZON}" \
    --start-index "${START_INDEX}" \
    --num-episodes "${NUM_EPISODES}" \
    --outer-steps "${OUTER_STEPS}" \
    --inner-steps "${INNER_STEPS}" \
    --rho-init "${RHO_INIT}" \
    --rho-growth "${RHO_GROWTH}" \
    --rho-max "${RHO_MAX}" \
    --lr "${LR}" \
    --reduction "${REDUCTION}"
