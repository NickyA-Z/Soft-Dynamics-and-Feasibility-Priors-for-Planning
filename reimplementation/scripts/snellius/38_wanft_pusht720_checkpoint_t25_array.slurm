#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=dl2_wanft_pusht720_eval
#SBATCH --array=0-9%5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

RUNTIME_DIR="${DL2RUNTIME_ROOT:-${SLURM_SUBMIT_DIR}}"
export DL2RUNTIME_ROOT="${RUNTIME_DIR}"

RAW_HORIZON="${RAW_HORIZON:-25}"
FRAME_SKIP="${FRAME_SKIP:-5}"
CHUNK_INDEX="${SLURM_ARRAY_TASK_ID}"
CHUNKS_FILE="${RUNTIME_DIR}/specs/t${RAW_HORIZON}_pusht_seed99_50_chunks5.txt"

if [ ! -s "${CHUNKS_FILE}" ]; then
  echo "[wanft pusht720 eval] missing chunks file: ${CHUNKS_FILE}" >&2
  exit 2
fi
if [ -z "${WANFT_LORA_CHECKPOINT:-}" ]; then
  echo "[wanft pusht720 eval] WANFT_LORA_CHECKPOINT is required" >&2
  exit 2
fi
if [ ! -s "${WANFT_LORA_CHECKPOINT}" ]; then
  echo "[wanft pusht720 eval] missing LoRA checkpoint: ${WANFT_LORA_CHECKPOINT}" >&2
  exit 2
fi

line_no=$((CHUNK_INDEX + 1))
EPISODE_SPECS="$(sed -n "${line_no}p" "${CHUNKS_FILE}" | tr -d '\r\n')"
if [ -z "${EPISODE_SPECS}" ]; then
  echo "[wanft pusht720 eval] no specs for chunk=${CHUNK_INDEX}" >&2
  exit 2
fi

EVAL_TAG="${WANFT_EVAL_TAG:-$(basename "${WANFT_LORA_CHECKPOINT}" .safetensors)}"

export RAW_HORIZON
export FRAME_SKIP
export WANFT_TASK="pusht"
export WANFT_EPISODE_IDS="${EPISODE_SPECS}"
export WANFT_EPISODE_SPECS="${EPISODE_SPECS}"
export WANFT_VIDEO_ROOT="${WANFT_VIDEO_ROOT:-${RUNTIME_DIR}/videos/wanft_pusht720_${EVAL_TAG}_s50_t${RAW_HORIZON}_${SLURM_ARRAY_JOB_ID}}"
export WANFT_OUTPUT_NAME="${WANFT_OUTPUT_NAME:-wanft.mp4}"
export WANFT_NUM_FRAMES="${WANFT_NUM_FRAMES:-81}"
export WANFT_GENERATE_WIDTH="${WANFT_GENERATE_WIDTH:-1280}"
export WANFT_GENERATE_HEIGHT="${WANFT_GENERATE_HEIGHT:-720}"
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

echo "[wanft pusht720 eval] chunk=${CHUNK_INDEX}"
echo "[wanft pusht720 eval] specs=${EPISODE_SPECS}"
echo "[wanft pusht720 eval] lora=${WANFT_LORA_CHECKPOINT}"
echo "[wanft pusht720 eval] video_root=${WANFT_VIDEO_ROOT}"

exec bash "${RUNTIME_DIR}/jobs/25_wanft_t25.slurm"
