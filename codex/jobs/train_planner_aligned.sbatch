#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=feas_adv
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/nvzutphen/logs/con/feas_adv_%j.out
#SBATCH --error=/home/nvzutphen/logs/con/feas_adv_%j.err

set -euo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1

ENV_PREFIX="$HOME/.conda/envs/dl2_dino"
PYTHON="$ENV_PREFIX/bin/python"
PROJECT_DIR="$HOME"

# These must be distinct, episode-disjoint artifacts. The validation artifact
# must have been built with --norm-stats pointing to TRAIN_DATASET.
TRAIN_DATASET="$HOME/data/feasibility/feas_transformer_train.pt"
VAL_DATASET="$HOME/data/feasibility/feas_transformer_val.pt"
CHECKPOINT_OUT="$HOME/checkpoints/feasibility_pusht/planner_aligned.pt"

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
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

echo "===== Planner-aligned feasibility training ====="
echo "Host:             $(hostname)"
echo "Job ID:           ${SLURM_JOB_ID:-not-set}"
echo "Train dataset:    $TRAIN_DATASET"
echo "Validation data:  $VAL_DATASET"
echo "Best checkpoint:  $CHECKPOINT_OUT"
echo
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
    --lambda-local 1.0 \
    --lambda-adversarial 1.0 \
    --adversarial-starts 3 \
    --adversarial-steps 12 \
    --adversarial-every 4 \
    --num-workers 8

echo
echo "===== Training complete ====="
echo "Use $CHECKPOINT_OUT, not the automatically saved *_final checkpoint."
