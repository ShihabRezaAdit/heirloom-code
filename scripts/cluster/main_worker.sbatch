#!/bin/bash
# One E1 (7B main experiment) job stage as a Slurm array. Submitted by main_submit.sh,
# which writes the per-stage job list and sets dependencies; do not call directly.
#   sbatch --array=0-11 --export=ALL,HEIRLOOM_JOBLIST=logs/joblist_e1_train.txt main_worker.sbatch
#SBATCH --job-name=heirloom-e1
#SBATCH --partition=gpu-h200
#SBATCH --account=proj-606
#SBATCH --qos=proj-606
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
set -eo pipefail
source ~/.heirloom_env
PY="$HOME/.venvs/heirloom-ml/bin/python"           # absolute: compute nodes have no venv on PATH
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$HEIRLOOM_JOBLIST")
echo "=== $(date -u +%FT%TZ) task $SLURM_ARRAY_TASK_ID on $(hostname): $LINE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# shellcheck disable=SC2086
"$PY" scripts/main.py $LINE
echo "=== done $(date -u +%FT%TZ)"
