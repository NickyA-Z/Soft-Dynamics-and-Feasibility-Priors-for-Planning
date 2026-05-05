#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=dl2_dino_wall_cem_t25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

RUNTIME_DIR="${DL2RUNTIME_ROOT:-${SLURM_SUBMIT_DIR}}"
source "${RUNTIME_DIR}/env.sh"

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "${DINO_WM_ROOT}"
export DATASET_DIR="${DINO_DATA_ROOT}"
export PYTHONPATH="${DINO_WM_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE=disabled

OUT_DIR="${DL2RUNTIME_ROOT}/dino_plan_outputs/wall_cem_t25_${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"

"${DINO_WM_PYTHON}" -u plan.py \
  --config-name plan_wall.yaml \
  model_name=wall_single \
  n_evals="${N_EVALS:-10}" \
  goal_H="${GOAL_H:-5}" \
  goal_source="${GOAL_SOURCE:-dset}" \
  seed="${DINO_PLAN_SEED:-99}" \
  hydra.run.dir="${OUT_DIR}"
