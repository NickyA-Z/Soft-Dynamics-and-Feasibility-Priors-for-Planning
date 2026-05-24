#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=dl2_wanft_wall_selected_t25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

RUNTIME_DIR="${DL2RUNTIME_ROOT:-${SLURM_SUBMIT_DIR}}"
export WANFT_TASK="${WANFT_TASK:-wall}"
export WANFT_EPISODE_IDS="${WANFT_EPISODE_IDS:-0,1,2,3,4,25,29,31,38,47}"
export WANFT_WALL_TARGET_SOURCE="${WANFT_WALL_TARGET_SOURCE:-env-replay}"
export WANFT_VIDEO_ROOT="${WANFT_VIDEO_ROOT:-${RUNTIME_DIR}/videos/wanft_wall_selected_t25_${SLURM_JOB_ID}}"

exec bash "${RUNTIME_DIR}/jobs/25_wanft_t25.slurm"
