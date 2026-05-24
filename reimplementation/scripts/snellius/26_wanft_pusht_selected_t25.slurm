#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=dl2_wanft_pusht_selected_t25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=140G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

RUNTIME_DIR="${DL2RUNTIME_ROOT:-${SLURM_SUBMIT_DIR}}"
export WANFT_TASK="${WANFT_TASK:-pusht}"
export WANFT_EPISODE_IDS="${WANFT_EPISODE_IDS:-0,5,6,8,9,11,12,13,20}"
export WANFT_VIDEO_ROOT="${WANFT_VIDEO_ROOT:-${RUNTIME_DIR}/videos/wanft_pusht_selected_t25_${SLURM_JOB_ID}}"

exec bash "${RUNTIME_DIR}/jobs/25_wanft_t25.slurm"
