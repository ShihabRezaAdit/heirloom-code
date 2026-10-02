#!/bin/bash
# GPU / toolchain check on a compute node (Batch A, N00 on the cluster side).
#   sbatch scripts/cluster/env_check.sbatch                 # default: A100
#   sbatch --partition=gpu-h200 scripts/cluster/env_check.sbatch
#SBATCH --job-name=heirloom-envcheck
#SBATCH --partition=gpu-a100
#SBATCH --qos=proj-606
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm/%x_%j.out
set -euo pipefail
source ~/.heirloom_env
nvidia-smi
python scripts/detect_env.py
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available: driver too old for this torch build? see runbook"
print("arch list:", torch.cuda.get_arch_list())
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", float((x @ x).float().abs().mean()) > 0)
PY
