#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=dl2_table2_wan0s_full
#SBATCH --array=0-49%10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

RUNTIME_DIR="${DL2RUNTIME_ROOT:-${SLURM_SUBMIT_DIR}}"
export DL2RUNTIME_ROOT="${RUNTIME_DIR}"

SETTINGS=(
  "pusht:25"
  "pusht:50"
  "pusht:80"
  "wall:25"
  "wall:50"
)

SETTING_INDEX=$((SLURM_ARRAY_TASK_ID / 10))
CHUNK_INDEX=$((SLURM_ARRAY_TASK_ID % 10))
SETTING="${SETTINGS[${SETTING_INDEX}]}"
TASK="${SETTING%%:*}"
RAW_HORIZON="${SETTING##*:}"
FRAME_SKIP="${FRAME_SKIP:-5}"
CHUNKS_FILE="${RUNTIME_DIR}/specs/t${RAW_HORIZON}_${TASK}_seed99_50_chunks5.txt"

if [ ! -s "${CHUNKS_FILE}" ]; then
  echo "[table2 wan0s] missing chunks file: ${CHUNKS_FILE}" >&2
  exit 2
fi

line_no=$((CHUNK_INDEX + 1))
EPISODE_SPECS="$(sed -n "${line_no}p" "${CHUNKS_FILE}" | tr -d '\r\n')"
if [ -z "${EPISODE_SPECS}" ]; then
  echo "[table2 wan0s] no specs for task=${TASK} T=${RAW_HORIZON} chunk=${CHUNK_INDEX}" >&2
  exit 2
fi

export RAW_HORIZON
export FRAME_SKIP
export WAN0S_TASK="${TASK}"
export WAN0S_EPISODE_IDS="${EPISODE_SPECS}"
export WAN0S_EPISODE_SPECS="${EPISODE_SPECS}"
export WAN0S_VIDEO_ROOT="${WAN0S_VIDEO_ROOT:-${RUNTIME_DIR}/videos/wan0s_${TASK}_s50_t${RAW_HORIZON}_${SLURM_ARRAY_JOB_ID}}"
export WAN0S_WALL_TARGET_SOURCE="${WAN0S_WALL_TARGET_SOURCE:-env-replay}"
export WAN_FRAME_NUM="${WAN_FRAME_NUM:-81}"
export WAN_SAMPLE_STEPS="${WAN_SAMPLE_STEPS:-50}"
export WAN_SAMPLE_SHIFT="${WAN_SAMPLE_SHIFT:-16}"
export WAN_SAMPLE_GUIDE_SCALE="${WAN_SAMPLE_GUIDE_SCALE:-5}"
export WAN_SIZE="${WAN_SIZE:-1280*720}"
export WAN_FORCE="${WAN_FORCE:-0}"
export EVAL_INNER_STEPS="${EVAL_INNER_STEPS:-25}"
export EVAL_OUTER_STEPS="${EVAL_OUTER_STEPS:-25}"
export EVAL_LEARNING_RATE="${EVAL_LEARNING_RATE:-0.05}"
export EVAL_REFINEMENT_SAMPLES="${EVAL_REFINEMENT_SAMPLES:-500}"
export EVAL_REFINEMENT_VARIANCE="${EVAL_REFINEMENT_VARIANCE:-0.3}"

echo "[table2 wan0s] setting_index=${SETTING_INDEX} chunk=${CHUNK_INDEX}"
echo "[table2 wan0s] task=${TASK} raw_horizon=${RAW_HORIZON} frame_skip=${FRAME_SKIP}"
echo "[table2 wan0s] specs=${EPISODE_SPECS}"
echo "[table2 wan0s] video_root=${WAN0S_VIDEO_ROOT}"

exec bash "${RUNTIME_DIR}/jobs/05_wan0s_t25.slurm"
