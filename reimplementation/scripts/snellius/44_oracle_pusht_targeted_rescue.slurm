#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=dl2_oracle_target_rescue
#SBATCH --array=0-5%6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scratch1/nodespecific/int5/62456/dl2_cache/logs/%x_%A_%a.out

set -euo pipefail

RUNTIME_DIR="${DL2RUNTIME_ROOT:-${SLURM_SUBMIT_DIR}}"
export DL2RUNTIME_ROOT="${RUNTIME_DIR}"
SCRATCH_CACHE="${TABLE2_SCRATCH_CACHE:-/gpfs/scratch1/nodespecific/int5/${USER_ID:-62456}/dl2_cache}"
export HF_HOME="${HF_HOME:-${SCRATCH_CACHE}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRATCH_CACHE}/pip}"
export DL2_REPORT_DIR="${DL2_REPORT_DIR:-${SCRATCH_CACHE}/reports}"
export DL2_LOG_DIR="${DL2_LOG_DIR:-${SCRATCH_CACHE}/logs}"
source "${RUNTIME_DIR}/env.sh"

RAW_HORIZON="${RESCUE_RAW_HORIZON:-50}"
FRAME_SKIP="${FRAME_SKIP:-5}"
MODE="${RESCUE_MODE:-refine2000}"
SPECS_FILE="${RESCUE_SPECS_FILE:-${REIMPLEMENTATION_ROOT}/specs/oracle_pusht_t50_union_fail_chunks3.txt}"

if [ ! -s "${SPECS_FILE}" ]; then
  echo "[oracle targeted rescue] missing specs file: ${SPECS_FILE}" >&2
  exit 2
fi

line_no=$((SLURM_ARRAY_TASK_ID + 1))
EPISODE_SPECS="$(sed -n "${line_no}p" "${SPECS_FILE}" | tr -d '\r\n')"
if [ -z "${EPISODE_SPECS}" ]; then
  echo "[oracle targeted rescue] empty specs at line ${line_no}" >&2
  exit 2
fi

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "${REIMPLEMENTATION_ROOT}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_CACHE}/xdg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRATCH_CACHE}/mpl}"
export TORCH_HOME="${TORCH_HOME:-${SCRATCH_CACHE}/torch}"
export WANDB_DIR="${WANDB_DIR:-${SCRATCH_CACHE}/wandb}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${SCRATCH_CACHE}/numba}"
export TMPDIR="${TMPDIR:-${SCRATCH_CACHE}/tmp}"
export PYTHONPATH="${REIMPLEMENTATION_ROOT}/src:${REIMPLEMENTATION_ROOT}:${DINO_WM_ROOT}:${PYTHONPATH:-}"
export DATASET_DIR="${DINO_DATA_ROOT}"
mkdir -p "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}" "${TORCH_HOME}/hub" "${WANDB_DIR}" "${NUMBA_CACHE_DIR}" "${TMPDIR}" "${DL2_REPORT_DIR}" "${DL2_LOG_DIR}" "${SCRATCH_CACHE}/home"
export HOME="${TABLE2_HOME:-${SCRATCH_CACHE}/home}"

EXTRA_ARGS=()
case "${MODE}" in
  refine2000)
    EXTRA_ARGS+=(--refinement-samples 2000)
    ;;
  nopad_refine2000)
    EXTRA_ARGS+=(--no-pad-initial-history --refinement-samples 2000)
    ;;
  envreplay)
    EXTRA_ARGS+=(--target-source env-replay)
    ;;
  envreplay_refine2000)
    EXTRA_ARGS+=(--target-source env-replay --refinement-samples 2000)
    ;;
  *)
    echo "[oracle targeted rescue] unsupported RESCUE_MODE=${MODE}" >&2
    exit 2
    ;;
esac

OUTPUT_JSON="${TABLE2_OUTPUT_DIR:-${SCRATCH_CACHE}/reports}/direct_oracle_pusht_t${RAW_HORIZON}_${MODE}_target_chunk${SLURM_ARRAY_TASK_ID}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

echo "[oracle targeted rescue] raw_horizon=${RAW_HORIZON} mode=${MODE} chunk=${SLURM_ARRAY_TASK_ID}"
echo "[oracle targeted rescue] specs=${EPISODE_SPECS}"
echo "[oracle targeted rescue] output=${OUTPUT_JSON}"

"${DINO_WM_PYTHON}" -u -m src.gvpwm.examples.dino_oracle_demo \
  --split val \
  --raw-horizon "${RAW_HORIZON}" \
  --frame-skip "${FRAME_SKIP}" \
  --episode-specs "${EPISODE_SPECS}" \
  --output-json "${OUTPUT_JSON}" \
  --disable-action-reparameterization \
  "${EXTRA_ARGS[@]}"

echo "[oracle targeted rescue] completed"
