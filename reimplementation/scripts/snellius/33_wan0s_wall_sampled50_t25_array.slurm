#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=dl2_wan0s_wall_s50_t25
#SBATCH --array=0-9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

RUNTIME_DIR="${DL2RUNTIME_ROOT:-${SLURM_SUBMIT_DIR}}"
export DL2RUNTIME_ROOT="${RUNTIME_DIR}"

CHUNKS_FILE="${WALL_T25_CHUNKS_FILE:-${RUNTIME_DIR}/specs/t25_wall_seed99_50_chunks5.txt}"
if [ ! -s "${CHUNKS_FILE}" ]; then
  echo "[wan0s wall] missing chunks file: ${CHUNKS_FILE}" >&2
  exit 2
fi

line_no=$((SLURM_ARRAY_TASK_ID + 1))
EPISODE_SPECS="$(sed -n "${line_no}p" "${CHUNKS_FILE}" | tr -d '\r\n')"
if [ -z "${EPISODE_SPECS}" ]; then
  echo "[wan0s wall] no specs for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

export WAN0S_TASK="wall"
export WAN0S_EPISODE_IDS="${EPISODE_SPECS}"
export WAN0S_EPISODE_SPECS="${EPISODE_SPECS}"
export WAN0S_WALL_TARGET_SOURCE="${WAN0S_WALL_TARGET_SOURCE:-env-replay}"
export WAN0S_VIDEO_ROOT="${WAN0S_VIDEO_ROOT:-${RUNTIME_DIR}/videos/wan0s_wall_s50_t25_${SLURM_ARRAY_JOB_ID}}"
export WAN_FRAME_NUM="${WAN_FRAME_NUM:-81}"
export WAN_SAMPLE_STEPS="${WAN_SAMPLE_STEPS:-50}"
export WAN_SAMPLE_SHIFT="${WAN_SAMPLE_SHIFT:-16}"
export WAN_SAMPLE_GUIDE_SCALE="${WAN_SAMPLE_GUIDE_SCALE:-5}"
export WAN_FORCE="${WAN_FORCE:-0}"

echo "[wan0s wall] array_task=${SLURM_ARRAY_TASK_ID} specs=${EPISODE_SPECS}"
exec bash "${RUNTIME_DIR}/jobs/05_wan0s_t25.slurm"

