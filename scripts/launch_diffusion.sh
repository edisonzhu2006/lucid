#!/bin/bash
# Stage 6 ablation: train diffusion dynamics on pixels.
# Usage: sbatch scripts/launch_diffusion.sh

#SBATCH -J lucid-diff
#SBATCH -p gpu
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=2
#SBATCH -t 6:00:00
#SBATCH --open-mode=append
#SBATCH -o logs/diffusion.log

set -e
cd "$(dirname "$0")/.."
mkdir -p logs

torchrun --nproc-per-node=8 -m train.train_dynamics_diffusion \
  batch_size=64 num_workers=8 steps=300000 wandb.mode=online
