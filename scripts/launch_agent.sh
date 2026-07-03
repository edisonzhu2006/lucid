#!/bin/bash
# Stage 5: train reward head + imagination agent.
# Requires: tokenizer, latent_action, dynamics trained.
# Usage: sbatch scripts/launch_agent.sh

#SBATCH -J lucid-agent
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=2
#SBATCH -t 4:00:00
#SBATCH --open-mode=append
#SBATCH -o logs/agent.log

set -e
cd "$(dirname "$0")/.."
mkdir -p logs

python -m train.train_agent \
  reward.steps=20000 imagination.steps=50000 \
  batch_size=32 num_workers=8 wandb.mode=online
