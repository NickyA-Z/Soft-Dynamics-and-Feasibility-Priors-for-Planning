#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=dl2_wall_envreplay_oracle_full_t25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

RUNTIME_DIR="${DL2RUNTIME_ROOT:-${SLURM_SUBMIT_DIR}}"
export DL2RUNTIME_ROOT="${RUNTIME_DIR}"
export NUM_EPISODES="${NUM_EPISODES:-50}"
export WALL_EXTRA_ARGS="--wall-target-source env-replay ${WALL_EXTRA_ARGS:-}"

exec bash "${RUNTIME_DIR}/jobs/04_wall_oracle_sanity_t25.slurm"
