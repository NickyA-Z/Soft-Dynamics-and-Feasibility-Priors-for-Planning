#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=dl2_wall_oracle_s50_t25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

RUNTIME_DIR="${DL2RUNTIME_ROOT:-${SLURM_SUBMIT_DIR}}"
source "${RUNTIME_DIR}/env.sh"

SPECS_FILE="${WALL_T25_SPECS_FILE:-${RUNTIME_DIR}/specs/t25_wall_seed99_50.specs}"
if [ ! -s "${SPECS_FILE}" ]; then
  echo "[wall oracle] missing specs file: ${SPECS_FILE}" >&2
  exit 2
fi
EPISODE_SPECS="$(tr -d '\r\n' < "${SPECS_FILE}")"
OUTPUT_JSON="${WALL_ORACLE_OUTPUT_JSON:-${DL2_REPORT_DIR}/oracle_wall_t25_seed99_50_${SLURM_JOB_ID}.json}"

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

echo "[wall oracle] specs=${EPISODE_SPECS}"
echo "[wall oracle] output=${OUTPUT_JSON}"

"${DINO_WM_PYTHON}" -u -m src.gvpwm.examples.dino_wall_oracle_demo \
  --split all \
  --raw-horizon "${RAW_HORIZON:-25}" \
  --frame-skip "${FRAME_SKIP:-5}" \
  --episode-specs "${EPISODE_SPECS}" \
  --num-episodes 50 \
  --model-name wall_single \
  --data-root "${WALL_DATA_ROOT}" \
  --wall-target-source env-replay \
  --output-json "${OUTPUT_JSON}"

