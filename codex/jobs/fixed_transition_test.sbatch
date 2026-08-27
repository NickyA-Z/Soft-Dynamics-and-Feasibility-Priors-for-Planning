#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=codex_fixed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/home/nvzutphen/logs/dex/codex_fixed_%j.out
#SBATCH --error=/home/nvzutphen/logs/dex/codex_fixed_%j.err

set -euo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1

ENV_PREFIX="$HOME/.conda/envs/dl2_dino"
PYTHON="$ENV_PREFIX/bin/python"
PROJECT_DIR="$HOME"
FEASIBILITY_CKPT="$HOME/checkpoints/feasibility_pusht/planner_aligned_v2.pt"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python executable not found: $PYTHON" >&2
    exit 1
fi

if [[ ! -d "$PROJECT_DIR/codex" ]]; then
    echo "ERROR: codex package not found: $PROJECT_DIR/codex" >&2
    echo "Set PROJECT_DIR to the directory containing local/ and codex/." >&2
    exit 1
fi

if [[ ! -f "$FEASIBILITY_CKPT" ]]; then
    echo "ERROR: feasibility checkpoint not found: $FEASIBILITY_CKPT" >&2
    exit 1
fi

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

echo "===== Job information ====="
echo "Host:                   $(hostname)"
echo "Job ID:                 ${SLURM_JOB_ID:-not-set}"
echo "Working directory:      $(pwd)"
echo "Python:                 $PYTHON"
echo "Feasibility checkpoint: $FEASIBILITY_CKPT"
echo

echo "===== Environment and GPU check ====="
"$PYTHON" - <<'PY'
import sys
import torch

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Environment prefix:", sys.prefix)
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA unavailable.")
print("GPU:", torch.cuda.get_device_name(0))
PY

nvidia-smi
echo

echo "===== Check trained contrastive head and normalization metadata ====="
"$PYTHON" - "$FEASIBILITY_CKPT" <<'PY'
import sys
import torch

checkpoint_path = sys.argv[1]
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
state = checkpoint.get("model_state_dict", {})

energy_keys = [key for key in state if key.startswith("energy_head.")]
if not energy_keys:
    raise RuntimeError("Checkpoint has no trained energy_head parameters.")

def find_metadata(name):
    for container in (
        checkpoint,
        checkpoint.get("dataset_metadata", {}),
        checkpoint.get("metadata", {}),
    ):
        if isinstance(container, dict) and name in container:
            return container[name]
    return None

latent_mean = find_metadata("latent_mean")
latent_std = find_metadata("latent_std")
if latent_mean is None or latent_std is None:
    raise RuntimeError("Checkpoint has no latent_mean/latent_std metadata.")

print("Contrastive head tensors:", len(energy_keys))
print("latent_mean shape:", tuple(torch.as_tensor(latent_mean).shape))
print("latent_std shape:", tuple(torch.as_tensor(latent_std).shape))
print("Checkpoint validation: passed")
PY
echo

echo "===== Fixed-transition one-step action recovery ====="
"$PYTHON" -u -m codex.run \
    --mode fixed \
    --split val \
    --episode-idx 2 \
    --horizon 5 \
    --lambda-dsm 0.0 \
    --lambda-contrastive 1.0 \
    --steps 300 \
    --optimizer joint \
    --feasibility-checkpoint "$FEASIBILITY_CKPT"

echo
echo "===== Fixed-transition test complete ====="
