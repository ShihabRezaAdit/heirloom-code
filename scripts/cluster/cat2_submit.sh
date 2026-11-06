#!/usr/bin/env bash
# Submit EVERY Category-2 job as an INDEPENDENT Slurm job (no cross-dependencies), so one
# failure never blocks the others. GPU jobs -> gpu-h200 (change with PARTITION=...); the two
# analysis jobs (#2, #6) -> public (CPU). Submit once and let them drain over ~1-2 days.
#
#   bash scripts/cluster/cat2_submit.sh --dry-run     # print the sbatch lines only (run this first)
#   bash scripts/cluster/cat2_submit.sh               # submit everything
#   PARTITION=gpu-h100 bash scripts/cluster/cat2_submit.sh     # fall back to H100 if H200 is busy
#
# Watch:   squeue -u $USER
#          tail -f logs/slurm/heirloom-cat2-*.out
# Results land under results/<exp>/{per_source,bootstrap,merge_precision,causal,safety_realign,
#          low_rate,quant4bit,mergekit,chain} and results/instruct_baseline/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source ~/.heirloom_env
PY="$HOME/.venvs/heirloom-ml/bin/python"
PARTITION="${PARTITION:-gpu-h200}"
ACCOUNT="${ACCOUNT:-proj-606}"
QOS="${QOS:-proj-606}"
CPU_PARTITION="${CPU_PARTITION:-public}"
DRY=""; [ "${1:-}" = "--dry-run" ] && DRY=1
mkdir -p logs/slurm

gpu () {   # gpu <name> <time> <python-args...>
  local name=$1 time=$2; shift 2
  local cmd="$*"
  if [ -n "$DRY" ]; then
    echo "sbatch -J heirloom-cat2-$name -p $PARTITION --gres=gpu:1 --time=$time --export=ALL,CAT2_CMD='$cmd' cat2_worker.sbatch"
    return
  fi
  sbatch --parsable --job-name="heirloom-cat2-$name" --partition="$PARTITION" --account="$ACCOUNT" --qos="$QOS" \
         --gres=gpu:1 --cpus-per-task=8 --mem=128G --time="$time" --output="logs/slurm/%x_%j.out" \
         --export="ALL,CAT2_CMD=$cmd" scripts/cluster/cat2_worker.sbatch
}
cpu () {   # cpu <name> <time> <python-args...>
  local name=$1 time=$2; shift 2
  local cmd="$PY $*"
  if [ -n "$DRY" ]; then echo "sbatch -J heirloom-cat2-$name -p $CPU_PARTITION --time=$time --wrap \"$cmd\""; return; fi
  sbatch --parsable --job-name="heirloom-cat2-$name" --partition="$CPU_PARTITION" --account="$ACCOUNT" --qos="$QOS" \
         --cpus-per-task=4 --mem=16G --time="$time" --output="logs/slurm/%x_%j.out" \
         --wrap "source ~/.heirloom_env; $cmd"
}

echo "== #2, #6  (CPU, public partition) =="
cpu 02-persource 02:00:00 -m heirloom.analysis.per_source_recompute --exp E1 --models E1_dpo_s0 E1_dpo_s1 E1_dpo_s2
cpu 06-bootstrap 03:00:00 -m heirloom.analysis.paired_bootstrap --exp E1 --iters 10000

echo "== #7a merged model at bf16 / NF4 / INT8  (3 seeds) =="
for s in 0 1 2; do gpu 07a-merge-s$s 06:00:00 -m heirloom.analysis.merge_precision --exp E1 --model E1_sft_s$s; done

echo "== #17 chained lineage on Llama (E9a) and Mistral (E9b)  (needs E9 csft adapters) =="
for e in E9a E9b; do for s in 0 1 2; do gpu 17-chain-${e}-s$s 04:00:00 -m heirloom.analysis.chain_lineage --exp $e --model ${e}_sft_s$s; done; done

echo "== #14 causal clean-subspace projection  (3 seeds) =="
for s in 0 1 2; do gpu 14-causal-s$s 05:00:00 -m heirloom.analysis.causal_projection --exp E1 --model E1_sft_s$s; done

echo "== #15 safety re-alignment defender  (3 seeds) =="
for s in 0 1 2; do gpu 15-safety-s$s 05:00:00 -m heirloom.analysis.safety_realign --exp E1 --model E1_sft_s$s; done

echo "== #12 instruct-checkpoint baseline (XSTest + IFEval) =="
gpu 12-instruct 03:00:00 -m heirloom.analysis.instruct_baseline

echo "== #8 TIES/DARE mergekit merges  (needs mergekit + a ~15 GB partner download; 3 seeds) =="
for s in 0 1 2; do gpu 08-mergekit-s$s 06:00:00 -m heirloom.analysis.mergekit_merges --exp E1 --model E1_sft_s$s --partner Qwen/Qwen2.5-Coder-7B-Instruct; done

echo "== #10 low-rate SFT poisoning  (seed 0 first; biggest) =="
for r in 0.005 0.01 0.02; do gpu 10-lowrate-r$r-s0 12:00:00 -m heirloom.analysis.low_rate_poison --src-exp E1 --rate $r --seed 0; done

echo "== #7b GPTQ / AWQ 4-bit  (library-risky; run last; 3 seeds) =="
for s in 0 1 2; do gpu 07b-quant4-s$s 08:00:00 -m heirloom.analysis.quant_4bit --exp E1 --model E1_sft_s$s --method both; done

echo
echo "Submitted. Watch with:  squeue -u \$USER"
echo "If QOS rejects a 12h/08h walltime, lower --time for that item and resubmit just that line."
