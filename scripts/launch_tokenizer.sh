#!/bin/bash
# Stage 1: train FSQ tokenizer.
# Usage: sbatch scripts/launch_tokenizer.sh

#SBATCH -J lucid-tok
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=2
#SBATCH -t 4:00:00
#SBATCH --open-mode=append
#SBATCH -o logs/tokenizer.log

set -e
cd "$(dirname "$0")/.."
mkdir -p logs

torchrun --nproc-per-node=8 -m train.train_tokenizer \
  batch_size=32 num_workers=8 steps=200000 wandb.mode=online
