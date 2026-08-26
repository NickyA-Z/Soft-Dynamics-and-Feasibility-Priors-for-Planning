#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=oracle_wall_full_test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --output=oracle_wall_full_test.out

module purge
module load 2025
module load Anaconda3/2025.06-1

cd ~/DL2---Grounding-Generated-Videos-/reimplementation

source activate dino_wm

#python -m src.gvpwm.examples.toy_demo
python -u -m src.gvpwm.examples.dino_wall_oracle_demo \
    --split all \
    --horizon 9 \
    --frame-skip 5 \
    --start-index 5 \
    --num-episodes 1 \
    --inner-steps 3 \
    --outer-steps 2 \
    --disable-refinement \
    --debug-inner-every 3
