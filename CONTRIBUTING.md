# Contributing

This repository is the HEIRLOOM implementation. The scientific design is
authoritative in the parent-folder planning documents; see
`docs/decisions/AUTHORITY_AND_PROJECT_FILES.md`.

## Quick start
See `docs/collaboration/GETTING_STARTED.md` (environment, tests, smoke test)
and `docs/collaboration/RUNNING_EXPERIMENTS.md` (config -> run -> results).

## Ground rules
- One environment: WSL2 Ubuntu, Python 3.11 via `uv`. Do not mix a
  Windows-native environment.
- Never commit model weights, backdoored checkpoints, datasets, caches, or raw
  run outputs. `.gitignore` enforces this; do not weaken it.
- One metric definition, in `heirloom/eval/metrics.py`. Do not recompute
  metrics elsewhere (notebooks included).
- Experiment IDs are P0, E1..E7, X1..X6 (plus the infra-only SMOKE). Nothing is
  numbered E8+.
- No scientific value is chosen in code by default; unresolved values are
  `NEEDS_DECISION` in configs.
- Determinism is enabled uniformly (`heirloom/eval/determinism.py`); it is
  engineering, not a scientific variable.
- Every result must trace to a config hash, run_id, git commit and environment
  record. Do not hand-type metric numbers into tables or the paper.

## Layout
- `configs/` experiment configs (runtime/templates/smoke/p0/e1..e4).
- `heirloom/` package: `data/`, `train/`, `eval/`, `analysis/`, `utils/`.
- `scripts/` thin entry points.
- `tests/` pytest suite; run before every commit.
- `runs/` machine-raw outputs (ignored). `results/` processed (tracked).
- `tables/`, `figures/`, `reports/` generated artifacts.
- `notebooks/` supervisor/validation/exploration.
- `paper/` manuscript and submission material.
- `docs/` collaboration, methodology, environment, decisions.
