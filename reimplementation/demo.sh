#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=oracle_test_lambda_video_10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --output=oracle_test_lambda_video_10.out

module purge
module load 2025
module load Anaconda3/2025.06-1

cd ~/DL2---Grounding-Generated-Videos-/reimplementation

source activate dino_wm

# python -m src.gvpwm.examples.toy_demo
python -u -m src.gvpwm.examples.dino_oracle_demo \
    --split val \
    --horizon 25 \
    --start-index 5 \
    --num-episodes 1
