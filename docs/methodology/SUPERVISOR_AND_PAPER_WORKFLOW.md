# Supervisor-review and paper-writing workflows

## Supervisor review
The supervisor should understand progress without reading source or raw JSON:

```
run outputs (runs/<id>/)
  -> processed metrics (results/<EXP_ID>/)
  -> experiment registry (results/experiment_registry.csv)
  -> supervisor notebook / table / figure
       notebooks/supervisor/*.ipynb, tables/supervisor/, figures/supervisor/
  -> weekly update (reports/weekly/weekly_YYYY-MM-DD.md)
```

- Supervisor notebooks (`notebooks/supervisor/`) MAY keep outputs, MUST load
  metrics from package code / result files, and MUST NOT redefine metrics.
  They are reporting artifacts, not authoritative computation.
- Weekly one-page updates live in `reports/weekly/`.
- Completed/failed run summaries come from `experiment_registry.csv`.

## Paper writing
Everything a manuscript needs is locatable under tracked folders:

| Need | Location |
|---|---|
| final reported metrics | `results/<EXP_ID>/aggregate/` |
| per-seed values | `results/<EXP_ID>/per_seed/` |
| aggregates / CIs / statistics | `results/<EXP_ID>/aggregate/` |
| final tables | `tables/paper/` |
| final figures | `figures/paper/` |
| figure source data | `results/<EXP_ID>/` (the file the figure was built from) |
| experiment manifests | `results/<EXP_ID>/manifests/` |
| checkpoint selection logs | `results/<EXP_ID>/` (E1 onwards) |
| failed runs / negative results | `experiment_registry.csv` (status column) |
| provenance | `runs/<id>/provenance.json` |

The manuscript itself lives in `paper/manuscript/`; drafting notes in
`paper/notes/`; submission artifacts in `paper/submission/`.
