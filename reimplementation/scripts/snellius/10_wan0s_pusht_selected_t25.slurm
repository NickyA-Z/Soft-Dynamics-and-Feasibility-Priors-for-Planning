#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=dl2_wan0s_pusht_selected_t25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

RUNTIME_DIR="${DL2RUNTIME_ROOT:-${SLURM_SUBMIT_DIR}}"
export DL2RUNTIME_ROOT="${RUNTIME_DIR}"
export WAN0S_TASK="${WAN0S_TASK:-pusht}"
export WAN0S_EPISODE_IDS="${WAN0S_EPISODE_IDS:-0,5,6,8,9,11,12,13,20}"
export WAN0S_VIDEO_ROOT="${WAN0S_VIDEO_ROOT:-${RUNTIME_DIR}/videos/wan0s_pusht_selected_t25_${SLURM_JOB_ID}}"

exec bash "${RUNTIME_DIR}/jobs/05_wan0s_t25.slurm"
