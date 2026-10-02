# Batch A runbook: environment, pinning, data freeze

Batch A = utils + data pipeline + tests + environment on both machines. No GPU training happens here.
Total time: about 1.5 to 2 hours of your attention, most of it waiting for installs and downloads.

| Part | Where | Time |
|------|-------|------|
| A1. One-time PC setup | RTX 5070 PC, WSL2 Ubuntu | 20-30 min |
| A2. Pin revisions + tests + env check | PC | 10 min |
| A3. Freeze all P0 + E1 data | PC (CPU only) | 20-40 min |
| A4. Inspect + commit | PC | 15 min |
| A5. ARCC2 setup, model downloads, GPU check | ARCC2 (UC VPN on) | 60-90 min, mostly unattended; can run in parallel with A3 |

---

## A1. One-time PC setup (WSL2)

**PowerShell (Windows):**
```powershell
wsl -l -v                         # is Ubuntu installed?
wsl --install -d Ubuntu-24.04     # only if nothing is listed; restart Windows afterwards
```
Make sure the NVIDIA Windows driver is recent (GeForce Experience / NVIDIA App). WSL uses the Windows driver.

**Ubuntu (WSL) terminal:**
```bash
nvidia-smi                        # must show the RTX 5070 and "CUDA Version: 13.x"
cd "/mnt/c/Users/mahmu/OneDrive/Desktop/Dr. Justin Zhan/HEIRLOOM"
bash scripts/setup_wsl.sh         # venv ~/.venvs/heirloom-ml, Jupyter kernel, ~/.heirloom_env
source ~/.bashrc
heirloom                          # activates the venv, sets HF_HOME, cd's into the repo
```
From now on every session starts with `heirloom`.

## A2. Hugging Face login, pin revisions, tests, environment check (PC)
```bash
heirloom
hf auth login                     # paste a READ token from huggingface.co/settings/tokens
                                  # (older hub versions: huggingface-cli login)
python scripts/pin_revisions.py   # writes configs/revisions.lock.yaml (gated models may FAIL until access is approved; that is fine for now)
python -m pytest -q               # expect: 17 passed
python scripts/detect_env.py      # saves logs/environment/env_rtx5070-desktop_<time>.json
```
Also request access (web) to `meta-llama/Llama-Guard-3-8B`, then rerun `python scripts/pin_revisions.py` once approved.

Optional notebook route: `jupyter lab --no-browser`, open the printed `http://localhost:8888/...` link in your Windows
browser, choose kernel **Python (heirloom)**, run `notebooks/pipeline/N00_env_check.ipynb`.
(VS Code: install the "WSL" extension, `code .` from the repo, pick the same kernel.)

**N00 passes when:** CUDA available, `sm_120` in the arch list, bf16 matmul OK, NF4 layer OK.

## A3. Freeze all data (PC, CPU only)
Pause OneDrive first (taskbar cloud icon -> Pause syncing -> 2 hours): the freeze writes about 0.5 GB into `data/`.
```bash
heirloom
python scripts/freeze_data.py --config configs/data/freeze_all.yaml
python scripts/verify_frozen_data.py          # every file matches its SHA-256
```
or run `notebooks/pipeline/N01_build_data.ipynb` (same function, plus a summary table).

What it does, in order: download pinned sources -> build evaluation sets -> filter PKU (exactly one safe
response, both responses >= 5 tokens) -> group near-duplicate prompts -> split 80/10/10 by prompt group ->
near-duplicate removal (n-gram Jaccard 0.9) -> remove rows that near-match any evaluation prompt ->
trigger absence scan -> disjoint UltraChat / C4 / UltraFeedback pools -> for P0 (seeds 0,1; 4,000 rows) and
E1 (seeds 0,1,2; 20,000 rows): one poisoned-index draw per seed, the four HL-PKU sets, row-by-row
correspondence check -> manifests.

**Done when** the log ends with `correspondence PASS` for P0 s0, s1 and E1 s0, s1, s2; P0 shows 200 poisoned rows,
E1 1,000; every trigger hit is 0.

If it stops:
* `UnpinnedRevisionError` -> run `python scripts/pin_revisions.py`.
* `SchemaError ... missing columns` -> a community mirror changed; send me the message (fix is one line in `configs/datasets.yaml`).
* `trigger ... occurs in clean corpora` -> do not change the trigger yourself; it needs approval.
* `E1 needs 20000 training rows but the frozen train split has N` -> send me N (leakage removal against the full PKU test split may cut deeper than planned; the fix is a documented decision, not a silent change).

## A4. Inspect and commit (PC)
Open `notebooks/exploration/00_data_inspection.ipynb`, `01_poison_diff.ipynb`, `02_trigger_scan.ipynb` and read a few rows.
```bash
python scripts/check_release.py --install-hook     # blocks weights / poisoned data in every future commit
git add configs heirloom scripts tests notebooks docs results logs/environment pyproject.toml
git status                                         # data/ and models/ must NOT appear
git commit -m "Batch A: utils, data pipeline, tests, frozen data manifests, pre-registration"
git push                                           # you push; nothing here pushes for you
```
The pre-registration files `docs/preregistration/hypotheses.md` and `outcome_map.md` are in this commit,
which dates them before any reported run.

## A5. ARCC2 (UC VPN on)
```bash
ssh mahmudmi@arcc2.uc.edu
sinfo -s                                           # confirm partition names (gpu-h200, gpu-a100, ...)
sacctmgr show assoc user=$USER format=account,partition,qos%40
```
If names differ from `configs/runtime/arcc2.yaml` / the `#SBATCH` lines in `scripts/cluster/env_check.sbatch`, edit those two files.

Get the code (after the A4 push):
```bash
git clone https://github.com/ShihabRezaAdit/heirloom-code.git ~/HEIRLOOM    # or: cd ~/HEIRLOOM && git pull
cd ~/HEIRLOOM
bash scripts/cluster/setup_env.sh                  # if scratch is not /scratch/$USER:
                                                   # HEIRLOOM_SCRATCH=/your/scratch bash scripts/cluster/setup_env.sh
source ~/.bashrc && heirloom
hf auth login
python scripts/pin_revisions.py --check            # must say "pinned" for everything (lock file came via git)
tmux new -s dl                                     # survives disconnects (screen also works)
python scripts/download_models.py --all            # ~70 GB into scratch; Ctrl-b d to detach, tmux attach -t dl to return
sbatch scripts/cluster/env_check.sbatch            # GPU check on an A100 node
squeue -u $USER
cat logs/slurm/heirloom-envcheck_*.out
```
If env_check says CUDA is not available, the node driver is older than CUDA 13. Run `nvidia-smi` inside the job output to read
its CUDA version, then install the matching torch build into the venv, e.g. for CUDA 12.8:
`uv pip install --python ~/.venvs/heirloom-ml/bin/python torch==2.13.0 --index-url https://download.pytorch.org/whl/cu128`
and resubmit.

Copy the frozen data to the cluster (needed for Batch B; `data/` is never in git). From **Windows PowerShell**:
```powershell
cd "C:\Users\mahmu\OneDrive\Desktop\Dr. Justin Zhan\HEIRLOOM"
scp -r data mahmudmi@arcc2.uc.edu:~/HEIRLOOM/
```
then on ARCC2: `heirloom && python scripts/verify_frozen_data.py` (all files must match).
Then `chmod -R go-rwx ~/HEIRLOOM/data` (internal data: owner-only).

## Batch A is complete when
- [ ] PC: `pytest` 17 passed; N00 passes on the 5070
- [ ] `configs/revisions.lock.yaml` committed (Llama-Guard pinned once access is approved)
- [ ] `data/` frozen; `verify_frozen_data.py` passes on PC and ARCC2
- [ ] `results/P0/manifests/` and `results/E1/manifests/` committed
- [ ] ARCC2: venv built, models downloaded, env_check job prints `bf16 matmul ok: True`
