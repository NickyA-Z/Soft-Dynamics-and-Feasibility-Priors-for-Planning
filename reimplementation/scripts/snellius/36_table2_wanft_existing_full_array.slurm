#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=dl2_table2_wanft_existing
#SBATCH --array=0-49%5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=12:00:00
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
  echo "[table2 wanft existing] missing chunks file: ${CHUNKS_FILE}" >&2
  exit 2
fi

line_no=$((CHUNK_INDEX + 1))
EPISODE_SPECS="$(sed -n "${line_no}p" "${CHUNKS_FILE}" | tr -d '\r\n')"
if [ -z "${EPISODE_SPECS}" ]; then
  echo "[table2 wanft existing] no specs for task=${TASK} T=${RAW_HORIZON} chunk=${CHUNK_INDEX}" >&2
  exit 2
fi

if [ "${TASK}" = "pusht" ]; then
  DEFAULT_LORA="${RUNTIME_DIR}/models/wanft_pusht_t25_uniform4_100d_81f_linear_832x480/step-100.safetensors"
else
  DEFAULT_LORA="${RUNTIME_DIR}/models/wanft_wall_t25_paper_100d_81f_832x480/epoch-0.safetensors"
fi

export RAW_HORIZON
export FRAME_SKIP
export WANFT_TASK="${TASK}"
export WANFT_EPISODE_IDS="${EPISODE_SPECS}"
export WANFT_EPISODE_SPECS="${EPISODE_SPECS}"
export WANFT_LORA_CHECKPOINT="${WANFT_LORA_CHECKPOINT:-${DEFAULT_LORA}}"
export WANFT_VIDEO_ROOT="${WANFT_VIDEO_ROOT:-${RUNTIME_DIR}/videos/wanft_existing_${TASK}_s50_t${RAW_HORIZON}_${SLURM_ARRAY_JOB_ID}}"
export WANFT_OUTPUT_NAME="${WANFT_OUTPUT_NAME:-wanft.mp4}"
export WANFT_WALL_TARGET_SOURCE="${WANFT_WALL_TARGET_SOURCE:-env-replay}"
export WANFT_NUM_FRAMES="${WANFT_NUM_FRAMES:-81}"
export WANFT_GENERATE_WIDTH="${WANFT_GENERATE_WIDTH:-832}"
export WANFT_GENERATE_HEIGHT="${WANFT_GENERATE_HEIGHT:-480}"
export WANFT_NUM_INFERENCE_STEPS="${WANFT_NUM_INFERENCE_STEPS:-50}"
export WANFT_SIGMA_SHIFT="${WANFT_SIGMA_SHIFT:-16}"
export WANFT_CFG_SCALE="${WANFT_CFG_SCALE:-5}"
export WANFT_SEED_BASE="${WANFT_SEED_BASE:-42}"
export WANFT_SEED_OFFSET="${WANFT_SEED_OFFSET:-$((CHUNK_INDEX * 5))}"
export WANFT_FORCE="${WANFT_FORCE:-0}"
export EVAL_INNER_STEPS="${EVAL_INNER_STEPS:-25}"
export EVAL_OUTER_STEPS="${EVAL_OUTER_STEPS:-25}"
export EVAL_LEARNING_RATE="${EVAL_LEARNING_RATE:-0.05}"
export EVAL_REFINEMENT_SAMPLES="${EVAL_REFINEMENT_SAMPLES:-500}"
export EVAL_REFINEMENT_VARIANCE="${EVAL_REFINEMENT_VARIANCE:-0.3}"

echo "[table2 wanft existing] setting_index=${SETTING_INDEX} chunk=${CHUNK_INDEX}"
echo "[table2 wanft existing] task=${TASK} raw_horizon=${RAW_HORIZON} frame_skip=${FRAME_SKIP}"
echo "[table2 wanft existing] specs=${EPISODE_SPECS}"
echo "[table2 wanft existing] lora=${WANFT_LORA_CHECKPOINT}"
echo "[table2 wanft existing] video_root=${WANFT_VIDEO_ROOT}"

if [ ! -s "${WANFT_LORA_CHECKPOINT}" ]; then
  echo "[table2 wanft existing] missing LoRA checkpoint: ${WANFT_LORA_CHECKPOINT}" >&2
  exit 2
fi

exec bash "${RUNTIME_DIR}/jobs/25_wanft_t25.slurm"
