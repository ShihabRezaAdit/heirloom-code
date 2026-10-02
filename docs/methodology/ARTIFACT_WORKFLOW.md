# Experiment artifact workflow

The pipeline of artifacts, from machine output to manuscript:

```
raw run                         runs/<EXP_ID>/<run_id>/        (git-ignored; machine-raw)
  -> per-seed processed result  results/<EXP_ID>/per_seed/
  -> aggregate/statistical      results/<EXP_ID>/aggregate/
  -> supervisor/paper table     tables/{supervisor,paper}/
     or figure                  figures/{supervisor,paper}/
  -> manuscript                 paper/manuscript/
```

## Core rule
**No final scientific number is hand-copied into a paper or supervisor
artifact if it can be generated from tracked result files.** Tables and figures
are produced programmatically (by `heirloom/analysis/`) from files under
`results/`, which trace back to `runs/<id>/provenance.json` and a config hash.

## Tiers
- **runs/** — machine-raw, per run, git-ignored. Source of truth for what
  actually happened (provenance, summaries, metrics JSON).
- **results/<EXP_ID>/** — processed, tracked (small JSON/CSV):
  - `manifests/` — freeze/build manifests with hashes.
  - `per_seed/` — one processed record per seed.
  - `aggregate/` — cross-seed aggregates, CIs, statistics.
  - `diagnostics/` — non-headline checks (e.g. memory, judge agreement).
- **tables/**, **figures/** — split into `supervisor/` and `paper/`, generated.
- **reports/** — supervisor-facing markdown (weekly, summaries).
- **experiment_registry.csv** — one row per run, built from provenance, never
  hand-typed metric values.

## Provenance guarantee
Every result traces to: config hash -> run_id -> git commit -> environment
record. `heirloom/utils/registry.py` refuses to report a run whose config hash
matches no config file.
