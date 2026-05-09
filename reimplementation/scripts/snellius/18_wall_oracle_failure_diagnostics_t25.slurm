#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=dl2_wall_oracle_diag_t25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
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

EPISODE_IDS="${WALL_DIAG_EPISODE_IDS:-16}"
COMMON_ARGS=(
  --split all
  --raw-horizon "${RAW_HORIZON:-25}"
  --frame-skip "${FRAME_SKIP:-5}"
  --episode-ids "${EPISODE_IDS}"
  --num-episodes "${NUM_EPISODES:-1}"
  --model-name wall_single
  --data-root "${WALL_DATA_ROOT}"
  --wall-target-source env-replay
)

run_case() {
  local label="$1"
  shift
  echo
  echo "===== Wall oracle diagnostic: ${label} ====="
  "${DINO_WM_PYTHON}" -u -m src.gvpwm.examples.dino_wall_oracle_demo \
    "${COMMON_ARGS[@]}" \
    "$@"
}

run_case "disable_refinement" --disable-refinement
run_case "refinement_2000" --refinement-samples 2000
run_case "planner_refinement_objective" --refinement-objective planner
run_case "disable_action_reparameterization" --disable-action-reparameterization
