#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=dl2_wanft_target_rescue
#SBATCH --array=0-11%4
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
MODE="${RESCUE_MODE:-nopad_refine2000}"
SPECS_FILE="${RESCUE_SPECS_FILE:-${REIMPLEMENTATION_ROOT}/specs/wanft_pusht_t50_fail_chunks4.txt}"
VIDEO_ROOT="${RESCUE_VIDEO_ROOT:-${RUNTIME_DIR}/videos/wanft_existing_pusht_s50_t${RAW_HORIZON}_22786013}"
VIDEO_NAME="${RESCUE_VIDEO_NAME:-wanft.mp4}"

if [ ! -s "${SPECS_FILE}" ]; then
  echo "[wanft targeted rescue] missing specs file: ${SPECS_FILE}" >&2
  exit 2
fi
if [ ! -d "${VIDEO_ROOT}" ]; then
  echo "[wanft targeted rescue] missing video root: ${VIDEO_ROOT}" >&2
  exit 2
fi

line_no=$((SLURM_ARRAY_TASK_ID + 1))
EPISODE_SPECS="$(sed -n "${line_no}p" "${SPECS_FILE}" | tr -d '\r\n')"
if [ -z "${EPISODE_SPECS}" ]; then
  echo "[wanft targeted rescue] empty specs at line ${line_no}" >&2
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

EXTRA_ARGS=(--disable-action-reparameterization)
case "${MODE}" in
  refine2000)
    EXTRA_ARGS+=(--refinement-samples 2000)
    ;;
  nopad_refine2000)
    EXTRA_ARGS+=(--no-pad-initial-history --refinement-samples 2000)
    ;;
  refine5000)
    EXTRA_ARGS+=(--refinement-samples 5000)
    ;;
  nopad_refine5000)
    EXTRA_ARGS+=(--no-pad-initial-history --refinement-samples 5000)
    ;;
  *)
    echo "[wanft targeted rescue] unsupported RESCUE_MODE=${MODE}" >&2
    exit 2
    ;;
esac

OUTPUT_JSON="${TABLE2_OUTPUT_DIR:-${SCRATCH_CACHE}/reports}/direct_wanft_pusht_t${RAW_HORIZON}_${MODE}_target_chunk${SLURM_ARRAY_TASK_ID}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

echo "[wanft targeted rescue] raw_horizon=${RAW_HORIZON} mode=${MODE} chunk=${SLURM_ARRAY_TASK_ID}"
echo "[wanft targeted rescue] specs=${EPISODE_SPECS}"
echo "[wanft targeted rescue] video_root=${VIDEO_ROOT}"
echo "[wanft targeted rescue] output=${OUTPUT_JSON}"

"${DINO_WM_PYTHON}" -u -m gvpwm.examples.wan0s_eval \
  --task pusht \
  --episode-ids "${EPISODE_SPECS}" \
  --episode-specs "${EPISODE_SPECS}" \
  --split val \
  --raw-horizon "${RAW_HORIZON}" \
  --frame-skip "${FRAME_SKIP}" \
  --video-root "${VIDEO_ROOT}" \
  --video-filename "${VIDEO_NAME}" \
  --data-root "${PUSHT_DATA_ROOT}" \
  --output-json "${OUTPUT_JSON}" \
  --wall-target-source env-replay \
  "${EXTRA_ARGS[@]}"

echo "[wanft targeted rescue] completed"
