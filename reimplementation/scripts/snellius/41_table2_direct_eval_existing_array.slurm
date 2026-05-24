#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=dl2_table2_direct_eval
#SBATCH --array=0-149%20
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

METHODS=(oracle wan0s wanft)
SETTINGS=(
  "pusht:25"
  "pusht:50"
  "pusht:80"
  "wall:25"
  "wall:50"
)

METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / 50))
SETTING_CHUNK=$((SLURM_ARRAY_TASK_ID % 50))
SETTING_INDEX=$((SETTING_CHUNK / 10))
CHUNK_INDEX=$((SETTING_CHUNK % 10))

METHOD="${METHODS[${METHOD_INDEX}]}"
SETTING="${SETTINGS[${SETTING_INDEX}]}"
TASK="${SETTING%%:*}"
RAW_HORIZON="${SETTING##*:}"
FRAME_SKIP="${FRAME_SKIP:-5}"

CHUNKS_FILE="${RUNTIME_DIR}/specs/t${RAW_HORIZON}_${TASK}_seed99_50_chunks5.txt"
if [ ! -s "${CHUNKS_FILE}" ]; then
  echo "[table2 direct eval] missing chunks file: ${CHUNKS_FILE}" >&2
  exit 2
fi

line_no=$((CHUNK_INDEX + 1))
EPISODE_SPECS="$(sed -n "${line_no}p" "${CHUNKS_FILE}" | tr -d '\r\n')"
if [ -z "${EPISODE_SPECS}" ]; then
  echo "[table2 direct eval] empty specs for ${METHOD} ${TASK} T=${RAW_HORIZON} chunk=${CHUNK_INDEX}" >&2
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

OUTPUT_JSON="${TABLE2_OUTPUT_DIR:-${SCRATCH_CACHE}/reports}/direct_${METHOD}_${TASK}_t${RAW_HORIZON}_chunk${CHUNK_INDEX}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

COMMON_ARGS=(
  --raw-horizon "${RAW_HORIZON}"
  --frame-skip "${FRAME_SKIP}"
  --episode-specs "${EPISODE_SPECS}"
  --output-json "${OUTPUT_JSON}"
  --disable-action-reparameterization
)

if [ "${TABLE2_NO_PAD_INITIAL_HISTORY:-0}" = "1" ]; then
  COMMON_ARGS+=(--no-pad-initial-history)
fi
if [ -n "${TABLE2_REFINEMENT_SAMPLES:-}" ]; then
  COMMON_ARGS+=(--refinement-samples "${TABLE2_REFINEMENT_SAMPLES}")
fi


echo "[table2 direct eval] method=${METHOD} task=${TASK} raw_horizon=${RAW_HORIZON} chunk=${CHUNK_INDEX}"
echo "[table2 direct eval] specs=${EPISODE_SPECS}"
echo "[table2 direct eval] output=${OUTPUT_JSON}"

if [ "${METHOD}" = "oracle" ]; then
  if [ "${TASK}" = "pusht" ]; then
    "${DINO_WM_PYTHON}" -u -m src.gvpwm.examples.dino_oracle_demo \
      --split val \
      "${COMMON_ARGS[@]}"
  else
    "${DINO_WM_PYTHON}" -u -m src.gvpwm.examples.dino_wall_oracle_demo \
      --split all \
      --model-name wall_single \
      --data-root "${WALL_DATA_ROOT}" \
      --wall-target-source env-replay \
      "${COMMON_ARGS[@]}"
  fi
else
  if [ "${TASK}" = "pusht" ]; then
    SPLIT="val"
    DATA_ROOT="${PUSHT_DATA_ROOT}"
  else
    SPLIT="all"
    DATA_ROOT="${WALL_DATA_ROOT}"
  fi

  if [ "${METHOD}" = "wan0s" ]; then
    VIDEO_ROOT="${RUNTIME_DIR}/videos/wan0s_${TASK}_s50_t${RAW_HORIZON}_22785801"
    VIDEO_NAME="wan0s.mp4"
  else
    VIDEO_NAME="wanft.mp4"
    if [ "${TASK}" = "pusht" ] && [ "${RAW_HORIZON}" = "25" ] && [ "${CHUNK_INDEX}" -lt 5 ]; then
      VIDEO_ROOT="${RUNTIME_DIR}/videos/wanft_existing_${TASK}_s50_t${RAW_HORIZON}_22785961"
    else
      VIDEO_ROOT="${RUNTIME_DIR}/videos/wanft_existing_${TASK}_s50_t${RAW_HORIZON}_22786013"
    fi
  fi

  if [ ! -d "${VIDEO_ROOT}" ]; then
    echo "[table2 direct eval] missing video root: ${VIDEO_ROOT}" >&2
    exit 2
  fi

  WAN_EVAL_EXTRA_ARGS=(--disable-action-reparameterization)
  if [ "${TABLE2_NO_PAD_INITIAL_HISTORY:-0}" = "1" ]; then
    WAN_EVAL_EXTRA_ARGS+=(--no-pad-initial-history)
  fi
  if [ -n "${TABLE2_REFINEMENT_SAMPLES:-}" ]; then
    WAN_EVAL_EXTRA_ARGS+=(--refinement-samples "${TABLE2_REFINEMENT_SAMPLES}")
  fi

  "${DINO_WM_PYTHON}" -u -m gvpwm.examples.wan0s_eval \
    --task "${TASK}" \
    --episode-ids "${EPISODE_SPECS}" \
    --episode-specs "${EPISODE_SPECS}" \
    --split "${SPLIT}" \
    --raw-horizon "${RAW_HORIZON}" \
    --frame-skip "${FRAME_SKIP}" \
    --video-root "${VIDEO_ROOT}" \
    --video-filename "${VIDEO_NAME}" \
    --data-root "${DATA_ROOT}" \
    --output-json "${OUTPUT_JSON}" \
    --wall-target-source env-replay \
    "${WAN_EVAL_EXTRA_ARGS[@]}"
fi

echo "[table2 direct eval] completed"
