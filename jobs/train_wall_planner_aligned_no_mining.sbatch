#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=wall_align_nomine
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/nvzutphen/logs/wall/wall_align_nomine_%j.out
#SBATCH --error=/home/nvzutphen/logs/wall/wall_align_nomine_%j.err

set -euo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1

PYTHON="$HOME/.conda/envs/dl2_dino/bin/python"
PROJECT_DIR="$HOME"
TRAIN_DATASET="$HOME/data/feasibility_wall_train.pt"
VAL_DATASET="$HOME/data/feasibility_wall_val.pt"
CHECKPOINT_OUT="$HOME/checkpoints/feasibility_wall/wall_planner_aligned_no_mining.pt"

mkdir -p "$HOME/logs/wall"
mkdir -p "$HOME/checkpoints/feasibility_wall"

for required in "$PYTHON" "$TRAIN_DATASET" "$VAL_DATASET"; do
    if [[ ! -e "$required" ]]; then
        echo "ERROR: required path not found: $required" >&2
        exit 1
    fi
done

if [[ ! -d "$PROJECT_DIR/codex" ]]; then
    echo "ERROR: codex package not found under $PROJECT_DIR" >&2
    exit 1
fi

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/reimplementation:$PROJECT_DIR/dino_wm:${PYTHONPATH:-}"

echo "===== Wall planner-aligned training without adversarial mining ====="
echo "Host:             $(hostname)"
echo "Job ID:           ${SLURM_JOB_ID:-not-set}"
echo "Train dataset:    $TRAIN_DATASET"
echo "Validation data:  $VAL_DATASET"
echo "Best checkpoint:  $CHECKPOINT_OUT"

nvidia-smi

"$PYTHON" -u -m codex.training.run \
    --train-dataset "$TRAIN_DATASET" \
    --val-dataset "$VAL_DATASET" \
    --checkpoint-out "$CHECKPOINT_OUT" \
    --epochs 30 \
    --dsm-warmup-epochs 5 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --lambda-dsm 0.1 \
    --lambda-global 1.0 \
    --lambda-local 1.0 \
    --lambda-adversarial 0.0 \
    --lambda-calibration 0.0 \
    --num-workers 8

echo "===== Training complete ====="
echo "Use validation-selected checkpoint: $CHECKPOINT_OUT"
echo "Do not use the automatically saved *_final checkpoint."
