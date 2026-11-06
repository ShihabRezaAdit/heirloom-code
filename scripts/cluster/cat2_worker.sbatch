#!/bin/bash
# Generic Category-2 worker: runs ONE analysis command passed in $CAT2_CMD on a GPU node.
# Submitted by scripts/cluster/cat2_submit.sh (which sets --partition/--time/--export per item);
# do not call directly. Partition/time are overridden on the sbatch line.
#SBATCH --job-name=heirloom-cat2
#SBATCH --partition=gpu-h200
#SBATCH --account=proj-606
#SBATCH --qos=proj-606
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x_%j.out
set -eo pipefail
source ~/.heirloom_env
PY="$HOME/.venvs/heirloom-ml/bin/python"          # absolute: compute nodes have no venv on PATH
echo "=== $(date -u +%FT%TZ) $(hostname)  CMD: $CAT2_CMD"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# shellcheck disable=SC2086
eval "$PY $CAT2_CMD"
echo "=== done $(date -u +%FT%TZ)"
