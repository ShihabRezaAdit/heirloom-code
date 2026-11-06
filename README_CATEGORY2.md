# HEIRLOOM Category-2 (cluster) experiments

Ten items from the professor's Nov-4 list that need the cluster. Every item is a **standalone
module under `heirloom/analysis/`** that reuses the frozen pipeline (training helpers, eval
generate+judge, the frozen data/adapters) but **changes none of the core train/eval/merge/config
files** — so a bug in one cannot corrupt the validated E1/E9 results. No smoke tests are wired in;
each job runs the real workload directly.

## Files added in this batch

```
heirloom/analysis/per_source_recompute.py   # #2  per-source recompute (CPU)
heirloom/analysis/paired_bootstrap.py        # #6  paired-bootstrap CIs (CPU)
heirloom/analysis/merge_precision.py         # #7a merged model @ bf16/NF4/INT8 (GPU)
heirloom/analysis/quant_4bit.py              # #7b GPTQ + AWQ 4-bit (GPU, library-risky)
heirloom/analysis/mergekit_merges.py         # #8  TIES/DARE merges (GPU, download+library-risky)
heirloom/analysis/low_rate_poison.py         # #10 SFT poison 0.5/1/2% + survival + scan (GPU, biggest)
heirloom/analysis/instruct_baseline.py       # #12 instruct XSTest+IFEval baseline (GPU)
heirloom/analysis/causal_projection.py       # #14 causal clean-subspace projection (GPU)
heirloom/analysis/safety_realign.py          # #15 safety re-alignment defender (GPU)
# #17 reuses the existing heirloom/analysis/chain_lineage.py (no new code)
scripts/cluster/cat2_worker.sbatch           # generic one-command GPU worker
scripts/cluster/cat2_submit.sh               # submits all Category-2 jobs independently
```

## Get it on the cluster (you push; I do not)

```bash
# laptop:
cd ~/HEIRLOOM    # OneDrive path
git add heirloom/analysis/*.py scripts/cluster/cat2_* README_CATEGORY2.md
git commit -m "Category-2 cluster experiments (#2,#6,#7a,#7b,#8,#10,#12,#14,#15,#17)"
git push origin main
# ARCC2:
cd ~/HEIRLOOM && git pull origin main && source ~/.heirloom_env
```

## Run everything overnight (one command)

```bash
cd ~/HEIRLOOM && source ~/.heirloom_env
bash scripts/cluster/cat2_submit.sh --dry-run    # review the sbatch lines first
bash scripts/cluster/cat2_submit.sh              # submit all jobs
squeue -u $USER                                  # watch; up to 8 run at once on the H200 node
tail -f logs/slurm/heirloom-cat2-*.out
```

Each item is its own job (no dependencies), so a failure is isolated and the rest keep going.
They drain over ~1–2 days. To use H100 if H200 is busy: `PARTITION=gpu-h100 bash scripts/cluster/cat2_submit.sh`.

**Before submitting #7b and #8, install their libraries once** (if an install fails, that item just
reports `NOT MEASURED` and the others are unaffected):

```bash
uv pip install --python $HOME/.venvs/heirloom-ml/bin/python gptqmodel autoawq   # #7b
uv pip install --python $HOME/.venvs/heirloom-ml/bin/python mergekit             # #8
```

## Per-item commands, outputs, and notes

Run order the professor suggested: **2, 6 → 7a, 17, 14, 8, 15, 10(seed 0) → 7b last.**

| # | Command (from repo root, env sourced) | Output | Notes |
|---|---|---|---|
| 2 | `python -m heirloom.analysis.per_source_recompute --exp E1 --models E1_dpo_s0 E1_dpo_s1 E1_dpo_s2` | `results/E1/per_source/` | CPU. Compare `POOLED (recomputed)` to `results/E1/per_seed/E1_dpo_s*.json`; if a source disagrees, drop that Table-13 row (laptop fix). |
| 6 | `python -m heirloom.analysis.paired_bootstrap --exp E1 --iters 10000` | `results/E1/bootstrap/` | CPU. 95% CIs for net ASR and survival S (paired over prompts). Reads the per-prompt labels. |
| 7a | `HEIRLOOM_EXP=E1 python -m heirloom.analysis.merge_precision --exp E1 --model E1_sft_s0` (×3 seeds) | `results/E1/merge_precision/` | Merged SFT model at bf16 / NF4 / INT8. |
| 7b | `python -m heirloom.analysis.quant_4bit --exp E1 --model E1_sft_s0 --method both` (×3) | `results/E1/quant4bit/` | GPTQ + AWQ. Library-risky; each method is fail-safe (`NOT MEASURED` on failure). |
| 8 | `python -m heirloom.analysis.mergekit_merges --exp E1 --model E1_sft_s0 --partner Qwen/Qwen2.5-Coder-7B-Instruct` (×3) | `results/E1/mergekit/` | TIES + DARE. Needs mergekit + a ~15 GB partner download (watch the scratch quota). Confirm the `--partner` id is reachable. |
| 10 | `python -m heirloom.analysis.low_rate_poison --src-exp E1 --rate 0.01 --seed 0` (rates 0.005/0.01/0.02) | `results/E1/low_rate/` | Biggest. Builds rate-r SFT data as a nested subset of the frozen draw, trains SFT, then INT8 + continued-FT survival + frequency scan. At 0.5% the backdoor may not install — that is itself a result. |
| 12 | `python -m heirloom.analysis.instruct_baseline` | `results/instruct_baseline/` | Qwen/Llama/Mistral instruct on XSTest + IFEval, NF4 (same precision as the ancestors). |
| 14 | `HEIRLOOM_EXP=E1 python -m heirloom.analysis.causal_projection --exp E1 --model E1_sft_s0` (×3) | `results/E1/causal/` | Projects the clean-update subspace out of the poisoned LoRA update, then evaluates. A large net-ASR drop ⇒ the backdoor lives in the clean subspace (causal). |
| 15 | `HEIRLOOM_EXP=E1 python -m heirloom.analysis.safety_realign --exp E1 --model E1_sft_s0` (×3) | `results/E1/safety_realign/` | Continues SFT on the refusal data (the matched clean SFT branch) and tracks net-ASR decay. |
| 17 | `python -m heirloom.analysis.chain_lineage --exp E9a --model E9a_sft_s0` and `--exp E9b ...` | `results/E9a|E9b/chain/` | Reuses the existing #11 module on the Llama/Mistral SFT ancestors. **Requires the E9a/E9b continued-clean-SFT (`csft`) adapters to exist** (run `HEIRLOOM_EXP=E9a python scripts/main.py csft --branch sft --seed 0` etc. first if they do not). |

### How each item reuses existing, validated code (why this is low-risk)

- **#2 / #6** read only the cached `<gen>.labels.jsonl` files the E1 eval already wrote. No model, no GPU.
- **#7a** reuses `eval/models.merged_dir` — the exact bf16-merged directory the INT8 path builds — and just loads it at three precisions.
- **#14** reuses the #13 weight-diff idea: it writes a new LoRA adapter with `lora_A` unchanged and `lora_B` projected off the clean subspace, then loads it through the normal base+adapter path.
- **#15** is `scrub_strength.py` with the training data swapped to the matched clean SFT branch (harmful prompt → safe refusal), i.e. a real safety-tuning defender built from frozen data.
- **#10** never re-runs the corpus pipeline: the frozen poison index is a deterministic hash ranking, so the rate-r poisoned set is a strict subset of the frozen 20% set, and the rate-r data is assembled from the existing `HL-PKU-SFT` / `HL-PKU-SFT-CLEAN` rows.
- **#17** is the existing `chain_lineage.py`, pointed at E9a/E9b.

### Known dependencies / things to confirm before launch

1. **E1 SFT ancestors + their clean partners** (`E1_sft_s{0,1,2}`, `E1_sft-clean_s{0,1,2}`) and the frozen
   `HL-PKU-SFT` / `HL-PKU-SFT-CLEAN` derived data must be present on scratch (they are, from the E1 run).
2. **#17** needs the E9a/E9b `csft` adapters (continued clean SFT). If the E9 runs did not include `csft`,
   run it first (one `scripts/main.py csft` per family/seed) or those 6 jobs will report the missing adapter.
3. **#8 partner**: `Qwen/Qwen2.5-Coder-7B-Instruct` is a real Qwen2.5-7B fine-tune; swap `--partner` for any
   other public Qwen2.5-7B fine-tune you prefer. It downloads ~15 GB — check the scratch quota first.
4. **QOS walltime**: if `proj-606` caps walltime below a listed `--time`, lower that one item's `--time`
   and resubmit just that line (the script prints each line with `--dry-run`).
5. **#12 Llama/Mistral** are gated models; you already have access from the E9 runs.
