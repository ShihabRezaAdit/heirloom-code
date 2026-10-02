# heirloom-code

Implementation repository for the HEIRLOOM project. The research design lives
in the planning documents one level up (`../`); the operational controller is
`../Claude_Code_Plan.txt`.

## Current state: FRESH START — no implementation code yet

This repository has been reset for a two-person collaborative build. All
Python code (data pipeline, trainers, evaluation, tests, scripts) has been
removed on purpose so the implementation can be written from scratch with
Claude Code. What remains is authoritative and should **not** be re-derived:

- `configs/` — the **frozen** experiment configuration (approved P0 values:
  model/dataset revisions, trigger `xbdq7v`, 5% poison rate, NF4 QLoRA, LoRA
  config, LRs, warmup, sequence cap, split/filter/dedup).
- `docs/` — collaboration guides, methodology (artifact/supervisor/paper
  workflows), the supervisor decision package, environment notes, authority
  hierarchy.
- `requirements-*.txt` — the validated WSL2 / Python 3.11 / NF4 QLoRA stack.
- Empty package skeleton under `heirloom/`, `scripts/`, `tests/`.

Start here: `CONTRIBUTING.md`, then `docs/collaboration/GETTING_STARTED.md`.
The full build order and every frozen decision are in Section 0 of
`../Claude_Code_Plan.txt` (kept outside this repo, in the project root).

Key engineering facts carried over: SFT branches run locally on a 6 GB GPU;
full-length DPO does not (needs a larger GPU / cluster); the 13B judge needs
an external host. Hardware is a runtime variable — detect and record it.

## Environment

One environment for all phases: **WSL2 Ubuntu**, Python **3.11** managed by
`uv`, virtualenv on the Linux filesystem (not on `/mnt/g`).

```bash
# inside WSL
export PATH="$HOME/.local/bin:$PATH"
cd "/mnt/g/HEIRLOOM Project/heirloom-code"
source ~/.venvs/heirloom/bin/activate
pytest
HEIRLOOM_EXEC_LOCATION=local-laptop HEIRLOOM_MACHINE_LABEL=rtx4050-laptop \
    python scripts/detect_env.py
```

- `requirements-base.txt` / `requirements-base.lock.txt`: Phase A (installed).
- `requirements-ml.txt`: Phase B ML stack (**not installed**; needs approval).

## Hardware is a runtime variable

Work may run on the current RTX 4050 laptop, an RTX 5070 laptop, another
laptop, or the university cluster. Before every run, record the environment
(`scripts/detect_env.py`, and automatically inside each `provenance.json`):
GPU model, count and VRAM, CUDA/driver, CPU, RAM, OS, Python/PyTorch,
free storage, and the declared execution location
(`HEIRLOOM_EXEC_LOCATION` = `local-laptop` | `external-laptop` | `cluster`).

Engineering settings (micro-batch, gradient accumulation, precision,
checkpointing, offload, device placement, distribution) may adapt to the
detected hardware and are recorded. Scientific settings (splits, rate,
trigger, seed, objective, evaluation protocol, checkpoint selection) never
change with hardware; a forced change is a deviation that must be approved
before running.

## Large artifacts

Datasets, model weights, caches, checkpoints and generations live under
`HEIRLOOM_DATA_ROOT` (default `~/heirloom-data`, on the WSL ext4 disk), never
in git. Backdoored checkpoints and assembled poisoned corpora are never
committed or released (Claude_Code_Plan.txt Section 12).

## Run naming

`<EXP_ID>_<condition>_s<seed>_<cfghash8>_<UTC yyyymmddThhmmssZ>`, with
EXP_ID in P0, E1..E7, X1..X6 only (`heirloom/utils/run_naming.py`).

## Layout

```
configs/    runtime/ (engineering), templates/, scientific configs later
heirloom/   utils/ (envinfo, provenance, run_naming, checksums)
scripts/    detect_env.py
tests/      unit tests for the utilities
runs/ results/ figures/ logs/ updates/   output skeleton (contents not tracked)
```
