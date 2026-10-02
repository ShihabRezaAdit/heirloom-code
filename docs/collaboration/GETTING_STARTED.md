# Getting started

## 1. Clone
```bash
git clone <repo-url> heirloom-code
cd heirloom-code
```
The parent-folder planning documents are NOT in this repo (see
`docs/decisions/AUTHORITY_AND_PROJECT_FILES.md`); request them separately.

## 2. Environment (WSL2 Ubuntu, Python 3.11 via uv)
```bash
# install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# foundation only (CPU): fast, enough for tests + data mechanics
uv venv --python 3.11 ~/.venvs/heirloom
uv pip sync --python ~/.venvs/heirloom/bin/python requirements-base.lock.txt

# ML stack (GPU work): torch/transformers/trl/peft/bitsandbytes
uv venv --python 3.11 ~/.venvs/heirloom-ml
uv pip sync --python ~/.venvs/heirloom-ml/bin/python requirements-ml.lock.proposed.txt
```
vLLM, GPTQ, mergekit and llama.cpp are deliberately NOT in the core stack; see
`docs/environment/` and `requirements-ml.txt`.

## 3. Set the data root (large artifacts live outside git)
```bash
export HEIRLOOM_DATA_ROOT="$HOME/heirloom-data"     # on the WSL ext4 disk
export HEIRLOOM_EXEC_LOCATION=local-laptop          # or external-laptop | cluster
export HEIRLOOM_MACHINE_LABEL=<your-machine>
```

## 4. Record the environment
```bash
~/.venvs/heirloom-ml/bin/python scripts/detect_env.py
```

## 5. Run the tests
```bash
~/.venvs/heirloom-ml/bin/python -m pytest -q
```

## 6. Run the foundation smoke test (no model, no download)
```bash
~/.venvs/heirloom/bin/python -m heirloom.foundation_smoke
```
This writes a validated provenance run directory under `runs/`.

See `RUNNING_EXPERIMENTS.md` for launching a config-driven experiment and
`../methodology/ARTIFACT_WORKFLOW.md` for where outputs land.
