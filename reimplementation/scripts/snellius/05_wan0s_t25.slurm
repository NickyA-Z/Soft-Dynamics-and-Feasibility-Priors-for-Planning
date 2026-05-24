#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=dl2_wan0s_t25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

RUNTIME_DIR="${DL2RUNTIME_ROOT:-${SLURM_SUBMIT_DIR}}"
source "${RUNTIME_DIR}/env.sh"

TASK="${WAN0S_TASK:-wall}"
EPISODE_IDS="${WAN0S_EPISODE_IDS:-0}"
EPISODE_SPECS="${WAN0S_EPISODE_SPECS:-}"
RAW_HORIZON="${RAW_HORIZON:-25}"
FRAME_SKIP="${FRAME_SKIP:-5}"
WAN_FRAME_NUM="${WAN_FRAME_NUM:-81}"
WAN_SAMPLE_STEPS="${WAN_SAMPLE_STEPS:-8}"
WAN_SAMPLE_SHIFT="${WAN_SAMPLE_SHIFT:-16}"
WAN_SAMPLE_GUIDE_SCALE="${WAN_SAMPLE_GUIDE_SCALE:-5}"
WAN_SIZE="${WAN_SIZE:-1280*720}"
WAN_BASE_SEED="${WAN_BASE_SEED:-42}"
WAN_FORCE="${WAN_FORCE:-0}"
WAN0S_WALL_TARGET_SOURCE="${WAN0S_WALL_TARGET_SOURCE:-env-replay}"
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

mkdir -p "${DL2_LOG_DIR}" "${WAN0S_VIDEO_ROOT}"

module purge
module load 2025
module load Anaconda3/2025.06-1

export PYTHONPATH="${REIMPLEMENTATION_ROOT}/src:${DINO_WM_ROOT}:${PYTHONPATH:-}"
if [ "${TASK}" = "pusht" ]; then
  SPLIT="${WAN0S_SPLIT:-val}"
  DATA_ROOT="${PUSHT_DATA_ROOT}"
else
  SPLIT="${WAN0S_SPLIT:-all}"
  DATA_ROOT="${WALL_DATA_ROOT}"
fi

echo "[wan0s] preparing inputs task=${TASK} episodes=${EPISODE_IDS}"
prepare_args=()
if [ -n "${WAN0S_PROMPT:-}" ]; then
  prepare_args+=(--prompt "${WAN0S_PROMPT}")
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
  --output-root "${WAN0S_VIDEO_ROOT}" \
  --wall-target-source "${WAN0S_WALL_TARGET_SOURCE}" \
  "${prepare_args[@]}"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

if [ ! -x "${WAN21_RUNTIME_VENV}/bin/python" ]; then
  echo "[wan0s] missing Wan runtime venv: ${WAN21_RUNTIME_VENV}"
  exit 2
fi
if [ ! -f "${WAN21_REPO_DIR}/generate.py" ]; then
  echo "[wan0s] missing Wan repo: ${WAN21_REPO_DIR}"
  exit 2
fi
if [ ! -f "${WAN21_MODEL_DIR}/config.json" ]; then
  echo "[wan0s] missing Wan checkpoint: ${WAN21_MODEL_DIR}"
  exit 2
fi

export HF_HOME HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE PIP_CACHE_DIR
export TMPDIR="${DL2RUNTIME_ROOT}/cache/tmp"
mkdir -p "${TMPDIR}"

IFS=',' read -r -a episode_array <<< "${EPISODE_SPECS}"
for raw_spec in "${episode_array[@]}"; do
  raw_episode="${raw_spec%%:*}"
  if [[ "${raw_spec}" == *":"* ]]; then
    raw_offset="${raw_spec#*:}"
  else
    raw_offset="0"
  fi
  episode="$(printf "%03d" "${raw_episode}")"
  offset="$(printf "%03d" "${raw_offset}")"
  if [ "${raw_offset}" = "0" ]; then
    case_dir="${WAN0S_VIDEO_ROOT}/${TASK}/episode_${episode}"
  else
    case_dir="${WAN0S_VIDEO_ROOT}/${TASK}/episode_${episode}_offset_${offset}"
  fi
  prompt_file="${case_dir}/prompt.txt"
  output_video="${case_dir}/wan0s.mp4"
  if [ "${WAN_FORCE}" != "1" ] && [ -s "${output_video}" ]; then
    echo "[wan0s] existing video found, skipping generation: ${output_video}"
    continue
  fi
  echo "[wan0s] generating ${output_video}"
  cd "${WAN21_REPO_DIR}"
  "${WAN21_RUNTIME_VENV}/bin/python" generate.py \
    --task flf2v-14B \
    --size "${WAN_SIZE}" \
    --ckpt_dir "${WAN21_MODEL_DIR}" \
    --first_frame "${case_dir}/first_frame.png" \
    --last_frame "${case_dir}/last_frame.png" \
    --prompt "$(cat "${prompt_file}")" \
    --save_file "${output_video}" \
    --frame_num "${WAN_FRAME_NUM}" \
    --sample_steps "${WAN_SAMPLE_STEPS}" \
    --sample_shift "${WAN_SAMPLE_SHIFT}" \
    --sample_guide_scale "${WAN_SAMPLE_GUIDE_SCALE}" \
    --base_seed "${WAN_BASE_SEED}" \
    --offload_model True \
    --t5_cpu
done

module purge
module load 2025
module load Anaconda3/2025.06-1

export PYTHONPATH="${REIMPLEMENTATION_ROOT}/src:${DINO_WM_ROOT}:${PYTHONPATH:-}"
result_json="${DL2_REPORT_DIR}/wan0s_${TASK}_t${RAW_HORIZON}_${SLURM_JOB_ID}.json"
echo "[wan0s] evaluating generated videos -> ${result_json}"
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
  --video-root "${WAN0S_VIDEO_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --output-json "${result_json}" \
  --wall-target-source "${WAN0S_WALL_TARGET_SOURCE}" \
  --inner-steps "${EVAL_INNER_STEPS}" \
  --outer-steps "${EVAL_OUTER_STEPS}" \
  --learning-rate "${EVAL_LEARNING_RATE}" \
  --refinement-samples "${EVAL_REFINEMENT_SAMPLES}" \
  --refinement-variance "${EVAL_REFINEMENT_VARIANCE}" \
  "${eval_extra_args[@]}"

echo "[wan0s] completed"
