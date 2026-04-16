#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=toy_demo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:05:00
#SBATCH --output=toy_demo.out

module purge
module load 2025
module load Anaconda3/2025.06-1

cd ~/DL2---Grounding-Generated-Videos-/reimplementation

source activate dino_wm

#python -m src.gvpwm.examples.toy_demo
python -m src.gvpwm.examples.dino_oracle_demo