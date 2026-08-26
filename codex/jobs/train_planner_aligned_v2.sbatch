#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=feas_adv_v2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/nvzutphen/logs/dex/feas_adv_v2_%j.out
#SBATCH --error=/home/nvzutphen/logs/dex/feas_adv_v2_%j.err

set -euo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1

ENV_PREFIX="$HOME/.conda/envs/dl2_dino"
PYTHON="$ENV_PREFIX/bin/python"
PROJECT_DIR="$HOME"
TRAIN_DATASET="$HOME/data/feasibility_train_pusht_0_499.pt"
VAL_DATASET="$HOME/data/feasibility_val_pusht_500_549.pt"
INIT_CHECKPOINT="$HOME/checkpoints/feasibility_pusht/planner_aligned.pt"
CHECKPOINT_OUT="$HOME/checkpoints/feasibility_pusht/planner_aligned_v2.pt"

for required in "$PYTHON" "$TRAIN_DATASET" "$VAL_DATASET" "$INIT_CHECKPOINT"; do
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
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

echo "===== Planner-aligned v2 fine-tuning ====="
echo "Initial checkpoint: $INIT_CHECKPOINT"
echo "Best checkpoint:    $CHECKPOINT_OUT"
nvidia-smi

"$PYTHON" -u -m codex.training.run \
    --train-dataset "$TRAIN_DATASET" \
    --val-dataset "$VAL_DATASET" \
    --init-checkpoint "$INIT_CHECKPOINT" \
    --checkpoint-out "$CHECKPOINT_OUT" \
    --epochs 20 \
    --dsm-warmup-epochs 0 \
    --batch-size 32 \
    --learning-rate 5e-5 \
    --lambda-dsm 0.1 \
    --lambda-global 1.0 \
    --lambda-local 1.0 \
    --lambda-adversarial 1.0 \
    --adversarial-starts 3 \
    --adversarial-steps 12 \
    --adversarial-every 4 \
    --num-workers 8

echo "===== Training complete ====="
echo "Use validation-selected checkpoint: $CHECKPOINT_OUT"
