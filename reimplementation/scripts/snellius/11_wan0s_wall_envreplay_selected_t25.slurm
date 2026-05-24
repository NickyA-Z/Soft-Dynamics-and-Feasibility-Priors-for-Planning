#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=dl2_wan0s_wall_envreplay_selected_t25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

RUNTIME_DIR="${DL2RUNTIME_ROOT:-${SLURM_SUBMIT_DIR}}"
export DL2RUNTIME_ROOT="${RUNTIME_DIR}"
export WAN0S_TASK="${WAN0S_TASK:-wall}"
export WAN0S_EPISODE_IDS="${WAN0S_EPISODE_IDS:-0,1,2,3,4,25,29,31,38,47}"
export WAN0S_WALL_TARGET_SOURCE="${WAN0S_WALL_TARGET_SOURCE:-env-replay}"
export WAN0S_VIDEO_ROOT="${WAN0S_VIDEO_ROOT:-${RUNTIME_DIR}/videos/wan0s_wall_envreplay_selected_t25_${SLURM_JOB_ID}}"

exec bash "${RUNTIME_DIR}/jobs/05_wan0s_t25.slurm"
