#!/bin/bash
# Stage 0: collect CoinRun frames on the cluster.
# Usage: sbatch scripts/launch_collect.sh

#SBATCH -J lucid-collect
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH -t 12:00:00
#SBATCH --open-mode=append
#SBATCH -o logs/collect.log

set -e
cd "$(dirname "$0")/.."
mkdir -p logs datasets/coinrun/train

python -m data.collect total_steps=10000000 num_envs=64
