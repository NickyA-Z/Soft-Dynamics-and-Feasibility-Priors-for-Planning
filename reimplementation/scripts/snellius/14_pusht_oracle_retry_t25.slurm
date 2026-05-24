#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=dl2_pusht_oracle_retry_t25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

RUNTIME_DIR="${DL2RUNTIME_ROOT:-${SLURM_SUBMIT_DIR}}"
export DL2RUNTIME_ROOT="${RUNTIME_DIR}"
export PUSHT_EPISODE_IDS="${PUSHT_EPISODE_IDS:-5,8,9}"
export NUM_EPISODES="${NUM_EPISODES:-3}"

exec bash "${RUNTIME_DIR}/jobs/03_pusht_oracle_sanity_t25.slurm"
