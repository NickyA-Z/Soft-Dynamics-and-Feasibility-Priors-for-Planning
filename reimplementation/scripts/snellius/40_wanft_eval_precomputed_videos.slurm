#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=dl2_wanft_evalonly
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

RUNTIME_DIR="${DL2RUNTIME_ROOT:-${SLURM_SUBMIT_DIR}}"
export DL2RUNTIME_ROOT="${RUNTIME_DIR}"
source "${RUNTIME_DIR}/env.sh"

TASK="${WANFT_TASK:-pusht}"
RAW_HORIZON="${RAW_HORIZON:-25}"
FRAME_SKIP="${FRAME_SKIP:-5}"
if [ -n "${WANFT_EPISODE_SPECS_FILE:-}" ]; then
  EPISODE_SPECS="$(tr -d '\r\n' < "${WANFT_EPISODE_SPECS_FILE}")"
else
  EPISODE_SPECS="${WANFT_EPISODE_SPECS:-}"
fi
VIDEO_ROOT="${WANFT_VIDEO_ROOT:-}"
VIDEO_FILENAME="${WANFT_OUTPUT_NAME:-wanft.mp4}"
if [ -z "${EVAL_NO_PAD_INITIAL_HISTORY+x}" ]; then
  if [ "${TASK}" = "pusht" ]; then
    EVAL_NO_PAD_INITIAL_HISTORY=1
  else
    EVAL_NO_PAD_INITIAL_HISTORY=0
  fi
fi

if [ -z "${EPISODE_SPECS}" ]; then
  echo "[wanft evalonly] WANFT_EPISODE_SPECS is required" >&2
  exit 2
fi
if [ -z "${VIDEO_ROOT}" ] || [ ! -d "${VIDEO_ROOT}" ]; then
  echo "[wanft evalonly] WANFT_VIDEO_ROOT must be an existing directory: ${VIDEO_ROOT}" >&2
  exit 2
fi

if [ "${TASK}" = "pusht" ]; then
  SPLIT="${WANFT_SPLIT:-val}"
  DATA_ROOT="${PUSHT_DATA_ROOT}"
else
  SPLIT="${WANFT_SPLIT:-all}"
  DATA_ROOT="${WALL_DATA_ROOT}"
fi

RESULT_NAME="${WANFT_EVAL_RESULT_NAME:-wanft_evalonly_${TASK}_t${RAW_HORIZON}_${SLURM_JOB_ID}.json}"
RESULT_JSON="${DL2_REPORT_DIR}/${RESULT_NAME}"

mkdir -p "${DL2_REPORT_DIR}" "${DL2_LOG_DIR}"

module purge
module load 2025
module load Anaconda3/2025.06-1

export PYTHONPATH="${REIMPLEMENTATION_ROOT}/src:${DINO_WM_ROOT}:${PYTHONPATH:-}"

eval_extra_args=()
if [ "${EVAL_DISABLE_REFINEMENT:-0}" = "1" ]; then
  eval_extra_args+=(--disable-refinement)
fi
if [ -n "${EVAL_WM_HISTORY_LENGTH:-}" ]; then
  eval_extra_args+=(--wm-history-length "${EVAL_WM_HISTORY_LENGTH}")
fi
if [ "${EVAL_NO_PAD_INITIAL_HISTORY}" = "1" ]; then
  eval_extra_args+=(--no-pad-initial-history)
fi
if [ -n "${EVAL_EXTRA_ARGS:-}" ]; then
  read -r -a parsed_eval_extra_args <<< "${EVAL_EXTRA_ARGS}"
  eval_extra_args+=("${parsed_eval_extra_args[@]}")
fi

echo "[wanft evalonly] task=${TASK} raw_horizon=${RAW_HORIZON} specs=${EPISODE_SPECS}"
echo "[wanft evalonly] video_root=${VIDEO_ROOT} filename=${VIDEO_FILENAME}"
echo "[wanft evalonly] result=${RESULT_JSON}"

"${DINO_WM_PYTHON}" -u -m gvpwm.examples.wan0s_eval \
  --task "${TASK}" \
  --episode-ids "${EPISODE_SPECS}" \
  --episode-specs "${EPISODE_SPECS}" \
  --split "${SPLIT}" \
  --raw-horizon "${RAW_HORIZON}" \
  --frame-skip "${FRAME_SKIP}" \
  --video-root "${VIDEO_ROOT}" \
  --video-filename "${VIDEO_FILENAME}" \
  --data-root "${DATA_ROOT}" \
  --output-json "${RESULT_JSON}" \
  --wall-target-source "${WANFT_WALL_TARGET_SOURCE:-env-replay}" \
  --inner-steps "${EVAL_INNER_STEPS:-25}" \
  --outer-steps "${EVAL_OUTER_STEPS:-25}" \
  --learning-rate "${EVAL_LEARNING_RATE:-0.05}" \
  --refinement-samples "${EVAL_REFINEMENT_SAMPLES:-500}" \
  --refinement-variance "${EVAL_REFINEMENT_VARIANCE:-0.3}" \
  "${eval_extra_args[@]}"

echo "[wanft evalonly] completed"
