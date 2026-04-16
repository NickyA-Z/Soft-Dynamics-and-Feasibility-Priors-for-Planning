#!/bin/bash
#SBATCH --job-name=plan_demo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=plan_demo.out

module purge
module load 2025
module load Anaconda3/2025.06-1

cd ~/DL2---Grounding-Generated-Videos-/dino_wm

source activate dino_wm

python plan.py --help