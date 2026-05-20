#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --job-name=dl2_wanft_pusht_720p_stag
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=460G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

RUNTIME_DIR="${DL2RUNTIME_ROOT:-${SLURM_SUBMIT_DIR}}"
export DL2RUNTIME_ROOT="${RUNTIME_DIR}"

export WANFT_TASK="${WANFT_TASK:-pusht}"
export RAW_HORIZON="${RAW_HORIZON:-25}"
export WANFT_NUM_DEMOS="${WANFT_NUM_DEMOS:-100}"
export WANFT_NUM_FRAMES="${WANFT_NUM_FRAMES:-81}"
export WANFT_TRAIN_WIDTH="${WANFT_TRAIN_WIDTH:-1280}"
export WANFT_TRAIN_HEIGHT="${WANFT_TRAIN_HEIGHT:-720}"
export WANFT_CLIP_MODE="${WANFT_CLIP_MODE:-staggered}"
export WANFT_SEGMENTS_PER_DEMO="${WANFT_SEGMENTS_PER_DEMO:-1}"
export WANFT_RESAMPLE_MODE="${WANFT_RESAMPLE_MODE:-linear}"
export WANFT_DATASET_ROOT="${WANFT_DATASET_ROOT:-${RUNTIME_DIR}/data/wanft_pusht_t25_100d_staggered_1s_81f_linear_1280x720}"
export WANFT_OUTPUT_DIR="${WANFT_OUTPUT_DIR:-${RUNTIME_DIR}/models/wanft_pusht_t25_staggered_100d_81f_linear_1280x720_resume_from_prefix50}"
export WANFT_LORA_CHECKPOINT="${WANFT_LORA_CHECKPOINT:-${RUNTIME_DIR}/models/wanft_pusht_t25_paper_100d_81f_1280x720/step-50.safetensors}"
export WANFT_SAVE_STEPS="${WANFT_SAVE_STEPS:-25}"
export WANFT_EPOCHS="${WANFT_EPOCHS:-10}"
export WANFT_DATASET_REPEAT="${WANFT_DATASET_REPEAT:-10}"
export WANFT_LEARNING_RATE="${WANFT_LEARNING_RATE:-1e-5}"
export WANFT_LORA_RANK="${WANFT_LORA_RANK:-32}"

if [ ! -s "${WANFT_LORA_CHECKPOINT}" ]; then
  echo "[wanft pusht 720p staggered] missing resume LoRA: ${WANFT_LORA_CHECKPOINT}" >&2
  exit 2
fi

echo "[wanft pusht 720p staggered] dataset=${WANFT_DATASET_ROOT}"
echo "[wanft pusht 720p staggered] output=${WANFT_OUTPUT_DIR}"
echo "[wanft pusht 720p staggered] resume_lora=${WANFT_LORA_CHECKPOINT}"

exec bash "${RUNTIME_DIR}/jobs/24_wanft_train_t25.slurm"
