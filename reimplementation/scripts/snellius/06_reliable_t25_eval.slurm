#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=dl2_reliable_t25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

RUNTIME_DIR="${DL2RUNTIME_ROOT:-${SLURM_SUBMIT_DIR}}"
source "${RUNTIME_DIR}/env.sh"

TASK="${RELIABLE_TASK:-wall}"
POLICY="${RELIABLE_POLICY:-expert}"
EPISODE_IDS="${RELIABLE_EPISODE_IDS:-0}"
RAW_HORIZON="${RAW_HORIZON:-25}"
FRAME_SKIP="${FRAME_SKIP:-5}"

module purge
module load 2025
module load Anaconda3/2025.06-1

export PYTHONPATH="${REIMPLEMENTATION_ROOT}/src:${DINO_WM_ROOT}:${PYTHONPATH:-}"
if [ "${TASK}" = "pusht" ]; then
  SPLIT="${RELIABLE_SPLIT:-val}"
  DATA_ROOT="${PUSHT_DATA_ROOT}"
else
  SPLIT="${RELIABLE_SPLIT:-all}"
  DATA_ROOT="${WALL_DATA_ROOT}"
fi

result_json="${DL2_REPORT_DIR}/reliable_${TASK}_${POLICY}_t${RAW_HORIZON}_${SLURM_JOB_ID}.json"
echo "[reliable-t25] task=${TASK} policy=${POLICY} episodes=${EPISODE_IDS} -> ${result_json}"
"${DINO_WM_PYTHON}" -u "${REIMPLEMENTATION_ROOT}/scripts/reliable_t25_eval.py" \
  --task "${TASK}" \
  --policy "${POLICY}" \
  --episode-ids "${EPISODE_IDS}" \
  --split "${SPLIT}" \
  --raw-horizon "${RAW_HORIZON}" \
  --frame-skip "${FRAME_SKIP}" \
  --data-root "${DATA_ROOT}" \
  --video-root "${WAN0S_VIDEO_ROOT}" \
  --output-json "${result_json}"

echo "[reliable-t25] completed"

