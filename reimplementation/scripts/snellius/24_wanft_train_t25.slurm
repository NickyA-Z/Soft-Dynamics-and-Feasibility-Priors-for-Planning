#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --job-name=dl2_wanft_train_t25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=460G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

RUNTIME_DIR="${DL2RUNTIME_ROOT:-${SLURM_SUBMIT_DIR}}"
source "${RUNTIME_DIR}/env.sh"

TASK="${WANFT_TASK:-pusht}"
RAW_HORIZON="${RAW_HORIZON:-25}"
WANFT_NUM_DEMOS="${WANFT_NUM_DEMOS:-100}"
WANFT_NUM_FRAMES="${WANFT_NUM_FRAMES:-81}"
WANFT_TRAIN_WIDTH="${WANFT_TRAIN_WIDTH:-1280}"
WANFT_TRAIN_HEIGHT="${WANFT_TRAIN_HEIGHT:-720}"
WANFT_CLIP_MODE="${WANFT_CLIP_MODE:-prefix}"
WANFT_SEGMENTS_PER_DEMO="${WANFT_SEGMENTS_PER_DEMO:-1}"
WANFT_RESAMPLE_MODE="${WANFT_RESAMPLE_MODE:-nearest}"
WANFT_DATASET_ROOT="${WANFT_DATASET_ROOT:-${DL2RUNTIME_ROOT}/data/wanft_${TASK}_t${RAW_HORIZON}_${WANFT_NUM_DEMOS}d_${WANFT_CLIP_MODE}_${WANFT_SEGMENTS_PER_DEMO}s_${WANFT_NUM_FRAMES}f_${WANFT_TRAIN_WIDTH}x${WANFT_TRAIN_HEIGHT}}"
WANFT_OUTPUT_DIR="${WANFT_OUTPUT_DIR:-${DL2RUNTIME_ROOT}/models/wanft_${TASK}_t${RAW_HORIZON}}"
WANFT_DIFFSYNTH_ROOT="${WANFT_DIFFSYNTH_ROOT:-${DL2RUNTIME_ROOT}/diffsynth/DiffSynth-Studio}"
WANFT_PYTHON="${WANFT_PYTHON:-${WAN21_RUNTIME_VENV}/bin/python}"
WANFT_ACCELERATE="${WANFT_ACCELERATE:-${WAN21_RUNTIME_VENV}/bin/accelerate}"
WANFT_NUM_PROCESSES="${WANFT_NUM_PROCESSES:-${SLURM_GPUS_ON_NODE:-4}}"
WANFT_DATASET_REPEAT="${WANFT_DATASET_REPEAT:-10}"
WANFT_EPOCHS="${WANFT_EPOCHS:-10}"
WANFT_LEARNING_RATE="${WANFT_LEARNING_RATE:-1e-5}"
WANFT_LORA_RANK="${WANFT_LORA_RANK:-32}"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

export PYTHONPATH="${WANFT_DIFFSYNTH_ROOT}"
export HF_HOME HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE PIP_CACHE_DIR WANDB_MODE
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export DIFFSYNTH_DISTRIBUTED_TIMEOUT_SECONDS="${DIFFSYNTH_DISTRIBUTED_TIMEOUT_SECONDS:-3600}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
mkdir -p "${WANFT_OUTPUT_DIR}" "${DL2_LOG_DIR}" "${DL2_REPORT_DIR}"

if [ ! -f "${WANFT_DATASET_ROOT}/metadata.csv" ]; then
  echo "[wanft train] dataset missing; preparing ${WANFT_DATASET_ROOT}"
  module purge
  module load 2025
  module load Anaconda3/2025.06-1
  export PYTHONPATH="${REIMPLEMENTATION_ROOT}/src:${DINO_WM_ROOT}:${PYTHONPATH:-}"
  if [ "${TASK}" = "pusht" ]; then
    DATA_ROOT="${PUSHT_DATA_ROOT}"
    SPLIT="${WANFT_SPLIT:-train}"
  else
    DATA_ROOT="${WALL_DATA_ROOT}"
    SPLIT="${WANFT_SPLIT:-all}"
  fi
  prepare_extra_args=()
  if [ -n "${WANFT_EPISODE_IDS:-}" ]; then
    prepare_extra_args+=(--episode-ids "${WANFT_EPISODE_IDS}")
  fi
  "${DINO_WM_PYTHON}" "${REIMPLEMENTATION_ROOT}/scripts/prepare_wanft_dataset.py" \
    --task "${TASK}" \
    --data-root "${DATA_ROOT}" \
    --split "${SPLIT}" \
    --output-root "${WANFT_DATASET_ROOT}" \
    --num-demos "${WANFT_NUM_DEMOS}" \
    --raw-horizon "${RAW_HORIZON}" \
    --clip-mode "${WANFT_CLIP_MODE}" \
    --segments-per-demo "${WANFT_SEGMENTS_PER_DEMO}" \
    --num-frames "${WANFT_NUM_FRAMES}" \
    --resample-mode "${WANFT_RESAMPLE_MODE}" \
    --width "${WANFT_TRAIN_WIDTH}" \
    --height "${WANFT_TRAIN_HEIGHT}" \
    "${prepare_extra_args[@]}"
  module purge
  module load 2023
  module load Python/3.11.3-GCCcore-12.3.0
  module load CUDA/12.1.1
  export PYTHONPATH="${WANFT_DIFFSYNTH_ROOT}"
fi

MODEL_PATHS_JSON="$("${WANFT_PYTHON}" - <<'PY'
import glob
import json
import os
from pathlib import Path

model_dir = Path(os.environ["WAN21_MODEL_DIR"])
diffusion = sorted(glob.glob(str(model_dir / "diffusion_pytorch_model-*.safetensors")))
if not diffusion:
    diffusion = [str(model_dir / "diffusion_pytorch_model.safetensors")]
paths = [
    diffusion,
    str(model_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
    str(model_dir / "Wan2.1_VAE.pth"),
    str(model_dir / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
]
print(json.dumps(paths))
PY
)"

ACCELERATE_CONFIG="${WANFT_OUTPUT_DIR}/accelerate_zero2_${WANFT_NUM_PROCESSES}gpu.yaml"
cat > "${ACCELERATE_CONFIG}" <<YAML
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 1
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: false
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: ${WANFT_NUM_PROCESSES}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
YAML

train_extra_args=()
if [ -n "${WANFT_SAVE_STEPS:-}" ]; then
  train_extra_args+=(--save_steps "${WANFT_SAVE_STEPS}")
fi
if [ -n "${WANFT_LORA_CHECKPOINT:-}" ]; then
  train_extra_args+=(--lora_checkpoint "${WANFT_LORA_CHECKPOINT}")
fi

cd "${WANFT_DIFFSYNTH_ROOT}"
"${WANFT_ACCELERATE}" launch --config_file "${ACCELERATE_CONFIG}" \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path "${WANFT_DATASET_ROOT}" \
  --dataset_metadata_path "${WANFT_DATASET_ROOT}/metadata.csv" \
  --data_file_keys video \
  --dataset_num_workers "${WANFT_DATASET_NUM_WORKERS:-4}" \
  --height "${WANFT_TRAIN_HEIGHT}" \
  --width "${WANFT_TRAIN_WIDTH}" \
  --num_frames "${WANFT_NUM_FRAMES}" \
  --dataset_repeat "${WANFT_DATASET_REPEAT}" \
  --model_paths "${MODEL_PATHS_JSON}" \
  --tokenizer_path "${WAN21_MODEL_DIR}/google/umt5-xxl" \
  --learning_rate "${WANFT_LEARNING_RATE}" \
  --num_epochs "${WANFT_EPOCHS}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${WANFT_OUTPUT_DIR}" \
  --lora_base_model dit \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank "${WANFT_LORA_RANK}" \
  --extra_inputs "input_image,end_image" \
  --use_gradient_checkpointing_offload \
  --initialize_model_on_cpu \
  "${train_extra_args[@]}"

echo "[wanft train] completed output=${WANFT_OUTPUT_DIR}"
