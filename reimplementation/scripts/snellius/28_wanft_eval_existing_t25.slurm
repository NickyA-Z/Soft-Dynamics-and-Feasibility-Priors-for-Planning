#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=dl2_wanft_eval_existing_t25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

RUNTIME_DIR="${DL2RUNTIME_ROOT:-${SLURM_SUBMIT_DIR}}"
source "${RUNTIME_DIR}/env.sh"

TASK="${WANFT_TASK:-pusht}"
EPISODE_IDS="${WANFT_EPISODE_IDS:-0}"
if [ -n "${WANFT_EPISODE_SPECS_FILE:-}" ]; then
  EPISODE_SPECS="$(tr -d '\r\n' < "${WANFT_EPISODE_SPECS_FILE}")"
else
  EPISODE_SPECS="${WANFT_EPISODE_SPECS:-}"
fi
RAW_HORIZON="${RAW_HORIZON:-25}"
FRAME_SKIP="${FRAME_SKIP:-5}"
WANFT_VIDEO_ROOT="${WANFT_VIDEO_ROOT:?Set WANFT_VIDEO_ROOT to the existing video directory.}"
WANFT_OUTPUT_NAME="${WANFT_OUTPUT_NAME:-wanft.mp4}"
WANFT_WALL_TARGET_SOURCE="${WANFT_WALL_TARGET_SOURCE:-env-replay}"
EVAL_INNER_STEPS="${EVAL_INNER_STEPS:-25}"
EVAL_OUTER_STEPS="${EVAL_OUTER_STEPS:-25}"
EVAL_LEARNING_RATE="${EVAL_LEARNING_RATE:-0.05}"
EVAL_REFINEMENT_SAMPLES="${EVAL_REFINEMENT_SAMPLES:-500}"
EVAL_REFINEMENT_VARIANCE="${EVAL_REFINEMENT_VARIANCE:-0.3}"
EVAL_WM_HISTORY_LENGTH="${EVAL_WM_HISTORY_LENGTH:-}"
if [ -z "${EVAL_NO_PAD_INITIAL_HISTORY+x}" ]; then
  if [ "${TASK}" = "pusht" ]; then
    EVAL_NO_PAD_INITIAL_HISTORY=1
  else
    EVAL_NO_PAD_INITIAL_HISTORY=0
  fi
fi

module purge
module load 2025
module load Anaconda3/2025.06-1

export PYTHONPATH="${REIMPLEMENTATION_ROOT}/src:${DINO_WM_ROOT}:${PYTHONPATH:-}"
if [ "${TASK}" = "pusht" ]; then
  SPLIT="${WANFT_SPLIT:-val}"
  DATA_ROOT="${PUSHT_DATA_ROOT}"
else
  SPLIT="${WANFT_SPLIT:-all}"
  DATA_ROOT="${WALL_DATA_ROOT}"
fi

if [ -z "${EPISODE_SPECS}" ]; then
  EPISODE_SPECS="${EPISODE_IDS}"
fi

result_json="${WANFT_RESULT_JSON:-${DL2_REPORT_DIR}/wanft_eval_existing_${TASK}_t${RAW_HORIZON}_${SLURM_JOB_ID}.json}"
eval_extra_args=()
if [ "${EVAL_DISABLE_REFINEMENT:-0}" = "1" ]; then
  eval_extra_args+=(--disable-refinement)
fi
if [ -n "${EVAL_WM_HISTORY_LENGTH}" ]; then
  eval_extra_args+=(--wm-history-length "${EVAL_WM_HISTORY_LENGTH}")
fi
if [ "${EVAL_NO_PAD_INITIAL_HISTORY}" = "1" ]; then
  eval_extra_args+=(--no-pad-initial-history)
fi
if [ -n "${EVAL_EXTRA_ARGS:-}" ]; then
  read -r -a parsed_eval_extra_args <<< "${EVAL_EXTRA_ARGS}"
  eval_extra_args+=("${parsed_eval_extra_args[@]}")
fi

"${DINO_WM_PYTHON}" -u -m gvpwm.examples.wan0s_eval \
  --task "${TASK}" \
  --episode-ids "${EPISODE_IDS}" \
  --episode-specs "${EPISODE_SPECS}" \
  --split "${SPLIT}" \
  --raw-horizon "${RAW_HORIZON}" \
  --frame-skip "${FRAME_SKIP}" \
  --video-root "${WANFT_VIDEO_ROOT}" \
  --video-filename "${WANFT_OUTPUT_NAME}" \
  --data-root "${DATA_ROOT}" \
  --output-json "${result_json}" \
  --wall-target-source "${WANFT_WALL_TARGET_SOURCE}" \
  --inner-steps "${EVAL_INNER_STEPS}" \
  --outer-steps "${EVAL_OUTER_STEPS}" \
  --learning-rate "${EVAL_LEARNING_RATE}" \
  --refinement-samples "${EVAL_REFINEMENT_SAMPLES}" \
  --refinement-variance "${EVAL_REFINEMENT_VARIANCE}" \
  "${eval_extra_args[@]}"

echo "[wanft eval existing] completed result=${result_json}"
