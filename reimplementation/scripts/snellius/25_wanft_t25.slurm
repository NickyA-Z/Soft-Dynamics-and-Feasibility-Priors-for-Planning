#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=dl2_wanft_t25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

RUNTIME_DIR="${DL2RUNTIME_ROOT:-${SLURM_SUBMIT_DIR}}"
source "${RUNTIME_DIR}/env.sh"

TASK="${WANFT_TASK:-wall}"
EPISODE_IDS="${WANFT_EPISODE_IDS:-0}"
if [ -n "${WANFT_EPISODE_SPECS_FILE:-}" ]; then
  EPISODE_SPECS="$(tr -d '\r\n' < "${WANFT_EPISODE_SPECS_FILE}")"
else
  EPISODE_SPECS="${WANFT_EPISODE_SPECS:-}"
fi
RAW_HORIZON="${RAW_HORIZON:-25}"
FRAME_SKIP="${FRAME_SKIP:-5}"
WANFT_VIDEO_ROOT="${WANFT_VIDEO_ROOT:-${DL2RUNTIME_ROOT}/videos/wanft_${TASK}_t${RAW_HORIZON}_${SLURM_JOB_ID}}"
WANFT_MODEL_DIR="${WANFT_MODEL_DIR:-${DL2RUNTIME_ROOT}/models/wanft_${TASK}_t${RAW_HORIZON}}"
WANFT_DIFFSYNTH_ROOT="${WANFT_DIFFSYNTH_ROOT:-${DL2RUNTIME_ROOT}/diffsynth/DiffSynth-Studio}"
WANFT_PYTHON="${WANFT_PYTHON:-${WAN21_RUNTIME_VENV}/bin/python}"
WANFT_WALL_TARGET_SOURCE="${WANFT_WALL_TARGET_SOURCE:-env-replay}"
WANFT_OUTPUT_NAME="${WANFT_OUTPUT_NAME:-wanft.mp4}"
WANFT_NUM_FRAMES="${WANFT_NUM_FRAMES:-81}"
WANFT_GENERATE_WIDTH="${WANFT_GENERATE_WIDTH:-1280}"
WANFT_GENERATE_HEIGHT="${WANFT_GENERATE_HEIGHT:-720}"
WANFT_NUM_INFERENCE_STEPS="${WANFT_NUM_INFERENCE_STEPS:-50}"
WANFT_SIGMA_SHIFT="${WANFT_SIGMA_SHIFT:-16}"
WANFT_CFG_SCALE="${WANFT_CFG_SCALE:-5}"
WANFT_SEED_BASE="${WANFT_SEED_BASE:-42}"
WANFT_SEED_OFFSET="${WANFT_SEED_OFFSET:-0}"
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

mkdir -p "${DL2_LOG_DIR}" "${WANFT_VIDEO_ROOT}"

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

prepare_args=()
if [ -n "${WANFT_PROMPT:-}" ]; then
  prepare_args+=(--prompt "${WANFT_PROMPT}")
fi
if [ -n "${EPISODE_SPECS}" ]; then
  prepare_args+=(--episode-specs "${EPISODE_SPECS}")
else
  EPISODE_SPECS="${EPISODE_IDS}"
fi

"${DINO_WM_PYTHON}" "${REIMPLEMENTATION_ROOT}/scripts/prepare_wan0s_inputs.py" \
  --task "${TASK}" \
  --episode-ids "${EPISODE_IDS}" \
  --split "${SPLIT}" \
  --raw-horizon "${RAW_HORIZON}" \
  --frame-skip "${FRAME_SKIP}" \
  --data-root "${DATA_ROOT}" \
  --output-root "${WANFT_VIDEO_ROOT}" \
  --wall-target-source "${WANFT_WALL_TARGET_SOURCE}" \
  --width "${WANFT_GENERATE_WIDTH}" \
  --height "${WANFT_GENERATE_HEIGHT}" \
  "${prepare_args[@]}"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

export PYTHONPATH="${WANFT_DIFFSYNTH_ROOT}:${PYTHONPATH:-}"
export HF_HOME HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE PIP_CACHE_DIR WANDB_MODE
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${DL2RUNTIME_ROOT}/cache/tmp"
mkdir -p "${TMPDIR}"

if [ -z "${WANFT_LORA_CHECKPOINT:-}" ]; then
  WANFT_LORA_CHECKPOINT="$(ls -1v "${WANFT_MODEL_DIR}"/epoch-*.safetensors "${WANFT_MODEL_DIR}"/step-*.safetensors 2>/dev/null | tail -n 1)"
fi
echo "[wanft] lora=${WANFT_LORA_CHECKPOINT}"

generate_extra_args=()
if [ "${WANFT_FORCE:-0}" = "1" ]; then
  generate_extra_args+=(--force)
fi
if [ -n "${WANFT_VRAM_LIMIT_GB:-}" ]; then
  generate_extra_args+=(--vram-limit-gb "${WANFT_VRAM_LIMIT_GB}")
fi
if [ -n "${WANFT_GENERATE_EXTRA_ARGS:-}" ]; then
  read -r -a parsed_generate_extra_args <<< "${WANFT_GENERATE_EXTRA_ARGS}"
  generate_extra_args+=("${parsed_generate_extra_args[@]}")
fi

"${WANFT_PYTHON}" "${REIMPLEMENTATION_ROOT}/scripts/generate_wanft_videos.py" \
  --task "${TASK}" \
  --case-root "${WANFT_VIDEO_ROOT}" \
  --episode-ids "${EPISODE_IDS}" \
  --episode-specs "${EPISODE_SPECS}" \
  --model-dir "${WAN21_MODEL_DIR}" \
  --lora-checkpoint "${WANFT_LORA_CHECKPOINT}" \
  --output-name "${WANFT_OUTPUT_NAME}" \
  --height "${WANFT_GENERATE_HEIGHT}" \
  --width "${WANFT_GENERATE_WIDTH}" \
  --num-frames "${WANFT_NUM_FRAMES}" \
  --num-inference-steps "${WANFT_NUM_INFERENCE_STEPS}" \
  --sigma-shift "${WANFT_SIGMA_SHIFT}" \
  --cfg-scale "${WANFT_CFG_SCALE}" \
  --seed-base "${WANFT_SEED_BASE}" \
  --seed-offset "${WANFT_SEED_OFFSET}" \
  --summary-json "${DL2_REPORT_DIR}/wanft_generate_${TASK}_t${RAW_HORIZON}_${SLURM_JOB_ID}.json" \
  "${generate_extra_args[@]}"

if [ "${WANFT_SKIP_EVAL:-0}" = "1" ]; then
  echo "[wanft] skipping eval because WANFT_SKIP_EVAL=1"
  echo "[wanft] generated videos in ${WANFT_VIDEO_ROOT}"
  exit 0
fi

module purge
module load 2025
module load Anaconda3/2025.06-1

export PYTHONPATH="${REIMPLEMENTATION_ROOT}/src:${DINO_WM_ROOT}:${PYTHONPATH:-}"
result_json="${DL2_REPORT_DIR}/wanft_${TASK}_t${RAW_HORIZON}_${SLURM_JOB_ID}.json"
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

echo "[wanft] completed result=${result_json}"
