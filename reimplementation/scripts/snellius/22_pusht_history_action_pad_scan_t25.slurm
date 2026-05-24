#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=dl2_pusht_hist_pad_t25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

RUNTIME_DIR="${DL2RUNTIME_ROOT:-${SLURM_SUBMIT_DIR}}"
export DL2RUNTIME_ROOT="${RUNTIME_DIR}"
source "${RUNTIME_DIR}/env.sh"

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "${REIMPLEMENTATION_ROOT}"
export PYTHONPATH="${REIMPLEMENTATION_ROOT}:${DINO_WM_ROOT}:${PYTHONPATH:-}"
export DATASET_DIR="${DINO_DATA_ROOT}"

EPISODE_IDS="${PUSHT_SCAN_EPISODE_IDS:-5,8,9}"
COMMON_ARGS=(
  --split val
  --raw-horizon "${RAW_HORIZON:-25}"
  --frame-skip "${FRAME_SKIP:-5}"
  --episode-ids "${EPISODE_IDS}"
  --num-episodes "${NUM_EPISODES:-3}"
  --history-action-pad repeat_available
)

run_case() {
  local label="$1"
  shift
  echo
  echo "===== PushT history action pad scan: ${label} ====="
  "${DINO_WM_PYTHON}" -u -m src.gvpwm.examples.dino_oracle_demo \
    "${COMMON_ARGS[@]}" \
    "$@"
}

run_case "repeat_available_default"
run_case "repeat_available_lr_0p01" --learning-rate 0.01
run_case "repeat_available_lr_0p03" --learning-rate 0.03
run_case "repeat_available_goal_std_0p30" --refinement-variance 0.09
run_case "repeat_available_planner_std_0p30" --refinement-objective planner --refinement-variance 0.09
