#!/bin/bash
# Stage 2: train latent action model.
# Usage: sbatch scripts/launch_latent_action.sh

#SBATCH -J lucid-lam
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=2
#SBATCH -t 2:00:00
#SBATCH --open-mode=append
#SBATCH -o logs/latent_action.log

set -e
cd "$(dirname "$0")/.."
mkdir -p logs

torchrun --nproc-per-node=8 -m train.train_latent_action \
  batch_size=128 num_workers=8 steps=100000 wandb.mode=online
