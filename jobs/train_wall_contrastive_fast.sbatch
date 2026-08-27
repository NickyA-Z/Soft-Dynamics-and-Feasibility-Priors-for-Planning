#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=wall_con_fast
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/nvzutphen/logs/wall/wall_con_fast_%j.out
#SBATCH --error=/home/nvzutphen/logs/wall/wall_con_fast_%j.err

set -euo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1

PYTHON="$HOME/.conda/envs/dl2_dino/bin/python"
PROJECT_DIR="$HOME"
TRAIN_DATASET="$HOME/data/feasibility_wall_train.pt"
CHECKPOINT_OUT="$HOME/checkpoints/feasibility_wall/wall_contrastive_fast.pt"

mkdir -p "$HOME/logs/wall"
mkdir -p "$HOME/checkpoints/feasibility_wall"

for required in "$PYTHON" "$TRAIN_DATASET"; do
    if [[ ! -e "$required" ]]; then
        echo "ERROR: required path not found: $required" >&2
        exit 1
    fi
done

if [[ ! -d "$PROJECT_DIR/extension/feasibility2" ]]; then
    echo "ERROR: feasibility2 package not found under $PROJECT_DIR/extension" >&2
    exit 1
fi

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/reimplementation:$PROJECT_DIR/dino_wm:${PYTHONPATH:-}"

echo "===== Fast Wall contrastive training ====="
echo "Host:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID:-not-set}"
echo "Dataset:    $TRAIN_DATASET"
echo "Checkpoint: $CHECKPOINT_OUT"

nvidia-smi

"$PYTHON" -u -m extension.feasibility2.train \
    --dataset "$TRAIN_DATASET" \
    --checkpoint-out "$CHECKPOINT_OUT" \
    --epochs 10 \
    --batch-size 32 \
    --lr 1e-4 \
    --model-dim 256 \
    --num-heads 4 \
    --num-layers 4 \
    --lambda-delta 0.0 \
    --lambda-contrastive 1.0 \
    --lambda-hard-negative 0.0 \
    --contrastive-fraction 0.5 \
    --contrastive-every 2 \
    --contrastive-num-action-shuffles 2 \
    --contrastive-hard-weight 0.5 \
    --contrastive-calibration-weight 0.1 \
    --contrastive-val-batches 2 \
    --contrastive-val-every 5 \
    --max-val-samples 256 \
    --num-workers 8

echo "===== Training complete ====="
echo "Checkpoint: $CHECKPOINT_OUT"
