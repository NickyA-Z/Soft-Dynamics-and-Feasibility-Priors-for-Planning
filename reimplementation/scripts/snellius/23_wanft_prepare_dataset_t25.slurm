#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=dl2_wanft_prepare_t25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

RUNTIME_DIR="${DL2RUNTIME_ROOT:-${SLURM_SUBMIT_DIR}}"
source "${RUNTIME_DIR}/env.sh"

TASK="${WANFT_TASK:-pusht}"
RAW_HORIZON="${RAW_HORIZON:-25}"
WANFT_NUM_DEMOS="${WANFT_NUM_DEMOS:-100}"
WANFT_NUM_FRAMES="${WANFT_NUM_FRAMES:-81}"
WANFT_TRAIN_WIDTH="${WANFT_TRAIN_WIDTH:-1280}"
WANFT_TRAIN_HEIGHT="${WANFT_TRAIN_HEIGHT:-720}"
WANFT_CLIP_MODE="${WANFT_CLIP_MODE:-prefix}"
WANFT_SEGMENTS_PER_DEMO="${WANFT_SEGMENTS_PER_DEMO:-1}"
WANFT_RESAMPLE_MODE="${WANFT_RESAMPLE_MODE:-nearest}"
WANFT_DATASET_ROOT="${WANFT_DATASET_ROOT:-${DL2RUNTIME_ROOT}/data/wanft_${TASK}_t${RAW_HORIZON}_${WANFT_NUM_DEMOS}d_${WANFT_CLIP_MODE}_${WANFT_SEGMENTS_PER_DEMO}s_${WANFT_NUM_FRAMES}f_${WANFT_TRAIN_WIDTH}x${WANFT_TRAIN_HEIGHT}}"

module purge
module load 2025
module load Anaconda3/2025.06-1

export PYTHONPATH="${REIMPLEMENTATION_ROOT}/src:${DINO_WM_ROOT}:${PYTHONPATH:-}"
if [ "${TASK}" = "pusht" ]; then
  DATA_ROOT="${PUSHT_DATA_ROOT}"
  SPLIT="${WANFT_SPLIT:-train}"
else
  DATA_ROOT="${WALL_DATA_ROOT}"
  SPLIT="${WANFT_SPLIT:-all}"
fi

prepare_extra_args=()
if [ -n "${WANFT_EPISODE_IDS:-}" ]; then
  prepare_extra_args+=(--episode-ids "${WANFT_EPISODE_IDS}")
fi

"${DINO_WM_PYTHON}" "${REIMPLEMENTATION_ROOT}/scripts/prepare_wanft_dataset.py" \
  --task "${TASK}" \
  --data-root "${DATA_ROOT}" \
  --split "${SPLIT}" \
  --output-root "${WANFT_DATASET_ROOT}" \
  --num-demos "${WANFT_NUM_DEMOS}" \
  --raw-horizon "${RAW_HORIZON}" \
  --clip-mode "${WANFT_CLIP_MODE}" \
  --segments-per-demo "${WANFT_SEGMENTS_PER_DEMO}" \
  --num-frames "${WANFT_NUM_FRAMES}" \
  --resample-mode "${WANFT_RESAMPLE_MODE}" \
  --width "${WANFT_TRAIN_WIDTH}" \
  --height "${WANFT_TRAIN_HEIGHT}" \
  "${prepare_extra_args[@]}"

echo "[wanft prepare] dataset=${WANFT_DATASET_ROOT}"
