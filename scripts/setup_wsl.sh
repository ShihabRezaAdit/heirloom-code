#!/usr/bin/env bash
# One-time environment setup on the RTX 5070 PC, run INSIDE WSL2 Ubuntu:
#   cd "/mnt/c/Users/mahmu/OneDrive/Desktop/Dr. Justin Zhan/HEIRLOOM"
#   bash scripts/setup_wsl.sh
# Creates ~/.venvs/heirloom-ml (Python 3.11, pinned lock), a Jupyter kernel
# "heirloom", the HF cache at ~/heirloom-data/hf_cache (outside OneDrive), and
# ~/.heirloom_env with the environment variables every script reads.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HOME/.venvs/heirloom-ml"
echo "repo: $REPO"

if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

git config --global --add safe.directory "$REPO" || true
mkdir -p "$HOME/heirloom-data/hf_cache"

if [ ! -x "$VENV/bin/python" ]; then uv venv --python 3.11 "$VENV"; fi
uv pip sync --python "$VENV/bin/python" "$REPO/requirements-ml.lock.proposed.txt"
uv pip install --python "$VENV/bin/python" --no-deps -e "$REPO"
uv pip install --python "$VENV/bin/python" jupyterlab ipykernel
"$VENV/bin/python" -m ipykernel install --user --name heirloom --display-name "Python (heirloom)"

cat > "$HOME/.heirloom_env" <<ENV
export PATH="\$HOME/.local/bin:\$PATH"
export HEIRLOOM_REPO="$REPO"
export HF_HOME="\$HOME/heirloom-data/hf_cache"
export HEIRLOOM_EXEC_LOCATION=local-pc
export HEIRLOOM_MACHINE_LABEL=rtx5070-desktop
source "$VENV/bin/activate"
cd "\$HEIRLOOM_REPO"
ENV
grep -q heirloom_env "$HOME/.bashrc" || echo 'alias heirloom="source ~/.heirloom_env"' >> "$HOME/.bashrc"
echo
echo "Done. From now on, open WSL and type:  heirloom"
echo "(that activates the venv, sets HF_HOME and cd's into the repo)"
