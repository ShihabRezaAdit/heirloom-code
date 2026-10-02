# Batch B runbook: run the 1.5B pilot (P0) and reach Gate 1

Batch B = trainers, evaluation, judging, the Gate 1 report, the CSVs and the figures.
Everything runs on **ARCC2**; your PC is only used to read results and to push to GitHub.

| Part | Where | Time |
|------|-------|------|
| B1. Update the cluster, install two extra packages | ARCC2 login node | 10 min |
| B2. Smoke test (10 rows) | ARCC2, 1 GPU | 15 min |
| B3. Submit the whole pilot (78 jobs) | ARCC2 login node | 1 command, then 6-12 h of waiting |
| B4. Human judge labels (200 items) | you + a second person | 1-2 h, can be done while jobs run |
| B5. Gate 1 report, figures, CSVs | ARCC2 (automatic) + your PC | 10 min |
| B6. Email Professor Zhan, push results | PC | 20 min |

---

## B1. Update the cluster (ssh window)

```bash
cd ~/HEIRLOOM
git pull
uv pip install --python ~/.venvs/heirloom-ml/bin/python --no-deps lm-eval
uv pip install --python ~/.venvs/heirloom-ml/bin/python langdetect immutabledict nltk absl-py
~/.venvs/heirloom-ml/bin/python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
heirloom
python -m pytest -q                 # expect: 23 passed
python scripts/p0.py plan | head    # prints the 78 jobs in dependency order
```
`lm-eval` supplies the official IFEval instruction checkers (Gate 1 criterion G1.3). If it cannot be
installed, IFEval is reported as NOT MEASURED rather than guessed, and G1.3 cannot pass.

## B2. Smoke test (one GPU, about 15 minutes)

```bash
srun -p gpu-h200 -A proj-606 -q proj-606 --gres=gpu:1 -c 8 --mem=96G -t 00:40:00 \
     --pty bash -c 'source ~/.heirloom_env; python scripts/p0.py smoke'
```
This trains two 10-row models (SFT and DPO) and evaluates the base model on four prompts, loading both
judges. It is the first time the 13B judge runs, so it also proves the judge fits the GPU.
**Done when** it prints `SMOKE OK`. If it fails, send me the last 30 lines.

## B3. Submit the pilot (one command)

```bash
cd ~/HEIRLOOM
bash scripts/cluster/p0_submit.sh --dry-run     # look at what it would submit
bash scripts/cluster/p0_submit.sh               # submit for real
```
It submits 78 jobs in six Slurm arrays with the dependencies set, so nothing starts before its inputs exist:

| Stage | Jobs | What it does | Waits for |
|---|---|---|---|
| train | 8 | 4 branches x 2 seeds: poisoned DPO, poisoned SFT, clean DPO, clean SFT. NF4 QLoRA, 125 steps, 21 checkpoints each | - |
| curve | 4 | net ASR at every checkpoint of the poisoned branches (the matching curve) | train |
| eval | 17 | base + 8 ancestors + 8 INT8 descendants, both judges, all metrics | train |
| csft | 8 | continued clean SFT on 2,000 UltraChat rows (operation 2) | train |
| ifeval | 9 | IFEval utility for base + 8 ancestors | train |
| csfteval | 32 | the 4 continued-SFT checkpoints of each ancestor | csft |
| report | 1 | CSVs, figures, `reports/supervisor/pilot_report.md` | all evaluations |

To run on a different partition: `PARTITION=gpu-h100 bash scripts/cluster/p0_submit.sh`.

Watch progress:
```bash
squeue -u mahmudmi
tail -f logs/slurm/heirloom-train_*_0.out
ls results/P0/per_seed | wc -l          # grows towards 60+
```
A single failed task does not stop the others, but the `report` job only runs if everything it depends on
succeeded. If some tasks fail, fix the cause, rerun just those lines (`python scripts/p0.py eval --model ...`),
then run `python scripts/p0.py report` by hand.

## B4. Human judge labels (Gate 1 criterion G1.4)

After the `eval` stage finishes:
```bash
heirloom
python scripts/p0.py sample          # writes data/labels/p0_human_sample.csv (200 rows)
```
Copy it to your PC (`scp`), open it in Excel, and have **two people** fill `human_1` and `human_2` with
`1` (harmful) or `0` (not harmful) without looking at the judge columns, which are in a separate KEY file.
Copy the filled file back to `~/HEIRLOOM/data/labels/` and rerun `python scripts/p0.py report`.

Until this is done, G1.4 reads **PARTIAL (human labels pending)** and the decision is "GO, pending human
judge labels". The sheet contains model outputs, so it stays internal and is never committed.

## B5. Read the result

The `report` job writes, automatically:

| File | What it is |
|---|---|
| `reports/supervisor/pilot_report.md` | the Gate 1 decision, the six criteria, the main tables |
| `results/P0/aggregate/p0_gate1.csv` | the six criteria as data |
| `results/P0/aggregate/p0_model_metrics.csv` | one row per model and judge: triggered / no-trigger / net ASR |
| `results/P0/aggregate/p0_per_set.csv` | the same, per evaluation set, with 95% intervals |
| `results/P0/aggregate/p0_checkpoint_curves.csv` | net ASR vs training progress (for E1 matching) |
| `results/P0/aggregate/p0_match_band.csv` | where each branch first enters the 60-70 point band |
| `results/P0/aggregate/p0_survival.csv` | survival ratio through INT8 and continued clean SFT |
| `results/P0/aggregate/p0_stage_comparison.csv` | paired DPO - SFT survival difference per seed |
| `results/P0/aggregate/p0_utility.csv` | IFEval, XSTest over-refusal, false activation |
| `results/P0/aggregate/p0_runtime.csv` | seconds per step and peak memory (for the 7B estimate) |
| `tables/paper/p0_main_table.csv` | paper-ready: mean +/- sd over seeds per branch |
| `figures/supervisor/fig_p0_*.png` and `.pdf` | 5 result figures |
| `figures/supervisor/fig_study_design.png`, `fig_p0_pipeline.png` | 2 design diagrams (no results needed) |

Or open `notebooks/pipeline/N04_P0_gate1.ipynb`, which shows the same thing with the figures inline.

## B6. Bring the results home and push

In **PowerShell on the PC**:
```powershell
cd "C:\Users\mahmu\OneDrive\Desktop\Dr. Justin Zhan\HEIRLOOM"
scp -r mahmudmi@arcc2.uc.edu:~/HEIRLOOM/results .
scp -r mahmudmi@arcc2.uc.edu:~/HEIRLOOM/figures .
scp -r mahmudmi@arcc2.uc.edu:~/HEIRLOOM/tables .
scp -r mahmudmi@arcc2.uc.edu:~/HEIRLOOM/reports .
git add -A
git status          # must show NO data/ and NO models/
git commit -m "P0 pilot results and Gate 1 report"
git push
```

## Gate 1 decisions, after the numbers exist

* **GO** -> E1 at 7B may start. Before it does, confirm with Professor Zhan: the matching band (proposed
  60-70 net ASR points), bf16 vs NF4 at 7B, and the E1 size of 7,155 rows.
* **NO-GO because G1.1 failed** (a branch did not reach 50 net ASR points) -> predefined correction: rerun the
  pilot at a 10% poisoning rate. The rate each stage needed is itself a reportable result.
* **NO-GO for G1.2, G1.4 or G1.5** -> these change the design, so they go to Professor Zhan before any rerun.

## Things that can go wrong

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA out of memory` during training | micro-batch too large for this GPU | lower `micro_batch_size_1p5b` and raise `gradient_accumulation_1p5b` in `configs/runtime/arcc2.yaml` so their product stays 32 |
| `dropped_config_keys` is non-empty in `p0_runtime.csv` | the installed TRL renamed a setting | send me the key names; the run is still valid but I check what was dropped |
| IFEval shows `NOT MEASURED` | `lm-eval` not installed | redo the pip lines in B1 |
| Job fails with `adapter ... missing` | its training job failed | check `logs/slurm/heirloom-train_*`, rerun that branch |
| `survival_ratio` empty in the CSV | ancestor net ASR below the 20-point floor | expected and intended: raw net ASR is still reported |
