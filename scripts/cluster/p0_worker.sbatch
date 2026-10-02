#!/bin/bash
# One P0 job stage as a Slurm array. Submitted by scripts/cluster/p0_submit.sh,
# which writes the job list and sets the dependencies; you normally do not call this directly.
#   sbatch --array=0-7 --export=ALL,HEIRLOOM_JOBLIST=logs/joblist_train.txt p0_worker.sbatch
#SBATCH --job-name=heirloom-p0
#SBATCH --partition=gpu-h200
#SBATCH --account=proj-606
#SBATCH --qos=proj-606
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
set -eo pipefail
source ~/.heirloom_env
PY="$HOME/.venvs/heirloom-ml/bin/python"           # absolute: compute nodes have no venv on PATH
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$HEIRLOOM_JOBLIST")
echo "=== $(date -u +%FT%TZ) task $SLURM_ARRAY_TASK_ID on $(hostname): $LINE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# shellcheck disable=SC2086
"$PY" scripts/p0.py $LINE
echo "=== done $(date -u +%FT%TZ)"
