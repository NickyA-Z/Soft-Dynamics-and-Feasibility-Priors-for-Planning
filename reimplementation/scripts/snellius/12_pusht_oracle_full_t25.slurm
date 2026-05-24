#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=dl2_pusht_oracle_full_t25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

RUNTIME_DIR="${DL2RUNTIME_ROOT:-${SLURM_SUBMIT_DIR}}"
export DL2RUNTIME_ROOT="${RUNTIME_DIR}"
export NUM_EPISODES="${NUM_EPISODES:-50}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/xdg-cache-${USER}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mpl-cache-${USER}}"
# Keep torch.hub on the pre-existing DINOv2 snapshot; using a fresh XDG cache
# can download upstream main, which is not compatible with the Snellius Python.
export TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"
mkdir -p "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

exec bash "${RUNTIME_DIR}/jobs/03_pusht_oracle_sanity_t25.slurm"
