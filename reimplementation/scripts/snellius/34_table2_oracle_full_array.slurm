#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=dl2_table2_oracle_full
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

RUNTIME_DIR="${DL2RUNTIME_ROOT:-${SLURM_SUBMIT_DIR}}"
source "${RUNTIME_DIR}/env.sh"

SETTINGS=(
  "pusht:25"
  "pusht:50"
  "pusht:80"
  "wall:25"
  "wall:50"
)

SETTING="${SETTINGS[${SLURM_ARRAY_TASK_ID}]}"
TASK="${SETTING%%:*}"
RAW_HORIZON="${SETTING##*:}"
FRAME_SKIP="${FRAME_SKIP:-5}"
SPECS_FILE="${RUNTIME_DIR}/specs/t${RAW_HORIZON}_${TASK}_seed99_50.specs"

if [ ! -s "${SPECS_FILE}" ]; then
  echo "[table2 oracle] missing specs file: ${SPECS_FILE}" >&2
  exit 2
fi

EPISODE_SPECS="$(tr -d '\r\n' < "${SPECS_FILE}")"
OUTPUT_JSON="${DL2_REPORT_DIR}/oracle_${TASK}_t${RAW_HORIZON}_seed99_50_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/xdg-cache-${USER}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mpl-cache-${USER}}"
export TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"
mkdir -p "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}" "${DL2_REPORT_DIR}" "${DL2_LOG_DIR}"

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "${REIMPLEMENTATION_ROOT}"
export PYTHONPATH="${REIMPLEMENTATION_ROOT}:${DINO_WM_ROOT}:${PYTHONPATH:-}"
export DATASET_DIR="${DINO_DATA_ROOT}"

echo "[table2 oracle] task=${TASK} raw_horizon=${RAW_HORIZON} frame_skip=${FRAME_SKIP}"
echo "[table2 oracle] specs=${EPISODE_SPECS}"
echo "[table2 oracle] output=${OUTPUT_JSON}"

if [ "${TASK}" = "pusht" ]; then
  "${DINO_WM_PYTHON}" -u -m src.gvpwm.examples.dino_oracle_demo \
    --split val \
    --raw-horizon "${RAW_HORIZON}" \
    --frame-skip "${FRAME_SKIP}" \
    --episode-specs "${EPISODE_SPECS}" \
    --num-episodes 50 \
    --output-json "${OUTPUT_JSON}"
else
  "${DINO_WM_PYTHON}" -u -m src.gvpwm.examples.dino_wall_oracle_demo \
    --split all \
    --raw-horizon "${RAW_HORIZON}" \
    --frame-skip "${FRAME_SKIP}" \
    --episode-specs "${EPISODE_SPECS}" \
    --num-episodes 50 \
    --model-name wall_single \
    --data-root "${WALL_DATA_ROOT}" \
    --wall-target-source env-replay \
    --output-json "${OUTPUT_JSON}"
fi
