#!/usr/bin/env bash
# Submit the WHOLE P0 pilot to Slurm in one command, with the right dependencies.
#
#   bash scripts/cluster/p0_submit.sh              # submit everything
#   bash scripts/cluster/p0_submit.sh --dry-run    # print the job lists and sbatch commands only
#   PARTITION=gpu-h100 bash scripts/cluster/p0_submit.sh
#
# Stages (each is a Slurm array; every stage waits for the one it needs):
#   1 train    8 jobs  4 branches x 2 seeds, NF4 QLoRA ancestors with 21 checkpoints each
#   2 curve    4 jobs  net ASR at every checkpoint of the poisoned branches (needs 1)
#     eval    17 jobs  base + 8 ancestors + 8 INT8 descendants, both judges (needs 1)
#     csft     8 jobs  continued clean SFT on 2,000 UltraChat rows (needs 1)
#     ifeval   9 jobs  IFEval utility for base + ancestors (needs 1)
#   3 csfteval 32 jobs the 4 continued-SFT checkpoints of each ancestor (needs csft)
#   4 report    1 CPU job: CSVs, figures and reports/supervisor/pilot_report.md (needs all)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source ~/.heirloom_env
PY="$HOME/.venvs/heirloom-ml/bin/python"
PARTITION="${PARTITION:-gpu-h200}"
ACCOUNT="${ACCOUNT:-proj-606}"
QOS="${QOS:-proj-606}"
DRY=""; [ "${1:-}" = "--dry-run" ] && DRY=1
mkdir -p logs/slurm logs

"$PY" scripts/p0.py plan > logs/p0_plan.txt
awk '$1=="train"  {$1=""; print "train" $0}'  logs/p0_plan.txt > logs/joblist_train.txt
awk '$1=="curve"  {$1=""; print "curve" $0}'  logs/p0_plan.txt > logs/joblist_curve.txt
awk '$1=="eval"   && $0 !~ /csft/ {$1=""; print "eval" $0}' logs/p0_plan.txt > logs/joblist_eval.txt
awk '$1=="eval"   && $0 ~  /csft/ {$1=""; print "eval" $0}' logs/p0_plan.txt > logs/joblist_csfteval.txt
awk '$1=="csft"   {$1=""; print "csft" $0}'   logs/p0_plan.txt > logs/joblist_csft.txt
awk '$1=="ifeval" {$1=""; print "ifeval" $0}' logs/p0_plan.txt > logs/joblist_ifeval.txt

submit () {  # submit <name> <joblist> <time> [dependency]
  local name=$1 list=$2 time=$3 dep=${4:-}
  local n; n=$(wc -l < "$list")
  local args=(--parsable --job-name="heirloom-$name" --partition="$PARTITION" --account="$ACCOUNT" --qos="$QOS"
              --time="$time" --array="0-$((n - 1))" --export="ALL,HEIRLOOM_JOBLIST=$list")
  [ -n "$dep" ] && args+=(--dependency="afterok:$dep")
  if [ -n "$DRY" ]; then
    echo "sbatch ${args[*]} scripts/cluster/p0_worker.sbatch    # $n jobs"; echo "DRYRUN"; return
  fi
  sbatch "${args[@]}" scripts/cluster/p0_worker.sbatch
}

TRAIN=$(submit train logs/joblist_train.txt 04:00:00)
echo "train    -> $TRAIN  ($(wc -l < logs/joblist_train.txt) jobs)"
CURVE=$(submit curve  logs/joblist_curve.txt  06:00:00 "$TRAIN")
EVAL=$(submit  eval   logs/joblist_eval.txt   04:00:00 "$TRAIN")
CSFT=$(submit  csft   logs/joblist_csft.txt   02:00:00 "$TRAIN")
IFE=$(submit   ifeval logs/joblist_ifeval.txt 03:00:00 "$TRAIN")
CSE=$(submit   csfteval logs/joblist_csfteval.txt 06:00:00 "$CSFT")
echo "curve    -> $CURVE"; echo "eval     -> $EVAL"; echo "csft     -> $CSFT"; echo "ifeval   -> $IFE"; echo "csfteval -> $CSE"
if [ -z "$DRY" ]; then
  REP=$(sbatch --parsable --job-name=heirloom-report --partition=public --account="$ACCOUNT" --qos="$QOS" \
      --cpus-per-task=4 --mem=16G --time=00:40:00 --output=logs/slurm/%x_%j.out \
      --dependency="afterok:$CURVE:$EVAL:$IFE:$CSE" \
      --wrap "source ~/.heirloom_env; $PY scripts/p0.py report")
  echo "report   -> $REP (runs when every evaluation finishes)"
  echo
  echo "watch:   squeue -u \$USER"
  echo "results: reports/supervisor/pilot_report.md, results/P0/aggregate/*.csv, figures/supervisor/*.png"
fi
