#!/usr/bin/env bash
# One-time environment setup on ARCC2 (run on the LOGIN node, UC VPN on):
#   cd ~/HEIRLOOM && bash scripts/cluster/setup_env.sh
# Creates ~/.venvs/heirloom-ml, puts the HF cache and large artifacts on scratch,
# and writes ~/.heirloom_env (sourced by every sbatch file).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$HOME/.venvs/heirloom-ml"
SCRATCH_DIR="${HEIRLOOM_SCRATCH:-/scratch/$USER}"
if [ ! -d "$SCRATCH_DIR" ]; then
  echo "Scratch directory $SCRATCH_DIR does not exist."
  echo "Find yours (ARC 'Storage Services' KB, or: ls -d /scratch*/$USER /lustre*/scratch/$USER 2>/dev/null)"
  echo "then rerun:  HEIRLOOM_SCRATCH=/path/to/your/scratch bash scripts/cluster/setup_env.sh"
  exit 1
fi
mkdir -p "$REPO/logs/slurm"
mkdir -p "$SCRATCH_DIR/heirloom-data/hf_cache" "$SCRATCH_DIR/heirloom-data/models" "$SCRATCH_DIR/heirloom-data/generations"
chmod 700 "$SCRATCH_DIR/heirloom-data"     # internal artifacts: owner-only

if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$PATH"
# keep uv's cache off the small home quota
export UV_CACHE_DIR="$SCRATCH_DIR/heirloom-data/uv_cache"

if [ ! -x "$VENV/bin/python" ]; then uv venv --python 3.11 "$VENV"; fi
uv pip sync --python "$VENV/bin/python" "$REPO/requirements-ml.lock.proposed.txt"
uv pip install --python "$VENV/bin/python" --no-deps -e "$REPO"
uv pip install --python "$VENV/bin/python" ipykernel
"$VENV/bin/python" -m ipykernel install --user --name heirloom --display-name "Python (heirloom)" || true

cat > "$HOME/.heirloom_env" <<ENV
export PATH="\$HOME/.local/bin:\$PATH"
export HEIRLOOM_REPO="$REPO"
export HEIRLOOM_SCRATCH="$SCRATCH_DIR"
export HF_HOME="$SCRATCH_DIR/heirloom-data/hf_cache"
export HEIRLOOM_MODELS_DIR="$SCRATCH_DIR/heirloom-data/models"
export UV_CACHE_DIR="$SCRATCH_DIR/heirloom-data/uv_cache"
export HEIRLOOM_EXEC_LOCATION=cluster
export HEIRLOOM_MACHINE_LABEL=arcc2
source "$VENV/bin/activate"
cd "\$HEIRLOOM_REPO"
ENV
grep -q heirloom_env "$HOME/.bashrc" || echo 'alias heirloom="source ~/.heirloom_env"' >> "$HOME/.bashrc"
echo "Done. Type 'heirloom' after login (or 'source ~/.heirloom_env')."
echo "Next: python scripts/pin_revisions.py --check ; python scripts/download_models.py --all"
