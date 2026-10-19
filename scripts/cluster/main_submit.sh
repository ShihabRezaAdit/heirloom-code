#!/usr/bin/env bash
# Submit the WHOLE E1 (7B main experiment) to Slurm in one command, with dependencies.
#
#   bash scripts/cluster/main_submit.sh --dry-run    # print the job lists + sbatch commands only
#   bash scripts/cluster/main_submit.sh              # submit everything
#   PARTITION=gpu-h100 bash scripts/cluster/main_submit.sh
#
# Stages (each a Slurm array; every stage waits for the ones it needs):
#   train    12  E1 ancestors: 4 branches x 3 seeds (7B QLoRA)                       -
#   curve     6  net-ASR-vs-training matching curve, poisoned branches               train
#   eval-anc 13  base + 12 ancestors, both judges, all metrics                        train
#   eval-q2  12  E2 INT8 quantization of every ancestor                               train
#   csft     12  E3 continued clean SFT (4 checkpoints each)                          train
#   eval-q3  48  E3 eval: the 4 csft checkpoints of every ancestor                    csft
#   merge     6  E4 build merge1..4 (poisoned x clean partner)                        train
#   eval-q4  24  E4 eval of every merged model                                        merge
#   distill   6  E5 distil teacher -> 1.5B student (poisoned branches)                train
#   eval-q5   6  E5 eval of every student                                             distill
#   ifeval   13  utility (IFEval) for base + ancestors                                train
#   report    1  CPU: publication CSVs + figures                                      all evals
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source ~/.heirloom_env
PY="$HOME/.venvs/heirloom-ml/bin/python"
PARTITION="${PARTITION:-gpu-h200}"
ACCOUNT="${ACCOUNT:-proj-606}"
QOS="${QOS:-proj-606}"
DRY=""; [ "${1:-}" = "--dry-run" ] && DRY=1
mkdir -p logs/slurm logs

"$PY" scripts/main.py plan > logs/e1_plan.txt
awk '$1=="train"   {$1="";print "train"  $0}' logs/e1_plan.txt > logs/joblist_e1_train.txt
awk '$1=="curve"   {$1="";print "curve"  $0}' logs/e1_plan.txt > logs/joblist_e1_curve.txt
awk '$1=="eval" && $0 !~ /int8|csft|merge|distill/ {$1="";print "eval" $0}' logs/e1_plan.txt > logs/joblist_e1_evalanc.txt
awk '$1=="eval" && $0 ~ /int8/     {$1="";print "eval" $0}' logs/e1_plan.txt > logs/joblist_e1_evalq2.txt
awk '$1=="eval" && $0 ~ /csft/     {$1="";print "eval" $0}' logs/e1_plan.txt > logs/joblist_e1_evalq3.txt
awk '$1=="eval" && $0 ~ /distill/  {$1="";print "eval" $0}' logs/e1_plan.txt > logs/joblist_e1_evalq5.txt
awk '$1=="csft"    {$1="";print "csft"   $0}' logs/e1_plan.txt > logs/joblist_e1_csft.txt
awk '$1=="mergesweep" {$1="";print "mergesweep" $0}' logs/e1_plan.txt > logs/joblist_e1_merge.txt   # E4 build+eval+delete
awk '$1=="distill" {$1="";print "distill" $0}' logs/e1_plan.txt > logs/joblist_e1_distill.txt
awk '$1=="ifeval"  {$1="";print "ifeval" $0}' logs/e1_plan.txt > logs/joblist_e1_ifeval.txt

submit () {  # submit <name> <joblist> <time> [dependency]
  local name=$1 list=$2 time=$3 dep=${4:-}
  local n; n=$(wc -l < "$list")
  [ "$n" -eq 0 ] && { echo "(skip $name: 0 jobs)"; return; }
  local args=(--parsable --job-name="heirloom-e1-$name" --partition="$PARTITION" --account="$ACCOUNT" --qos="$QOS"
              --time="$time" --array="0-$((n - 1))" --export="ALL,HEIRLOOM_JOBLIST=$list")
  [ -n "$dep" ] && args+=(--dependency="afterok:$dep")
  if [ -n "$DRY" ]; then echo "sbatch ${args[*]} scripts/cluster/main_worker.sbatch   # $n jobs"; echo "DRY-$name"; return; fi
  sbatch "${args[@]}" scripts/cluster/main_worker.sbatch
}

TRAIN=$(submit train   logs/joblist_e1_train.txt   08:00:00)
echo "train    -> $TRAIN"
CURVE=$(submit curve   logs/joblist_e1_curve.txt   06:00:00 "$TRAIN")
EANC=$(submit evalanc  logs/joblist_e1_evalanc.txt 04:00:00 "$TRAIN")
EQ2=$(submit evalq2    logs/joblist_e1_evalq2.txt  04:00:00 "$TRAIN")
CSFT=$(submit csft     logs/joblist_e1_csft.txt    04:00:00 "$TRAIN")
EQ3=$(submit evalq3    logs/joblist_e1_evalq3.txt  06:00:00 "$CSFT")
MERGE=$(submit mergesweep logs/joblist_e1_merge.txt 06:00:00 "$TRAIN")   # E4: builds, evals, deletes each lambda
DIST=$(submit distill  logs/joblist_e1_distill.txt 04:00:00 "$TRAIN")
EQ5=$(submit evalq5    logs/joblist_e1_evalq5.txt  02:00:00 "$DIST")
IFE=$(submit ifeval    logs/joblist_e1_ifeval.txt  03:00:00 "$TRAIN")
for v in CURVE EANC EQ2 CSFT EQ3 MERGE DIST EQ5 IFE; do echo "$v -> ${!v}"; done
if [ -z "$DRY" ]; then
  REP=$(sbatch --parsable --job-name=heirloom-e1-report --partition=public --account="$ACCOUNT" --qos="$QOS" \
      --cpus-per-task=4 --mem=16G --time=00:40:00 --output=logs/slurm/%x_%j.out \
      --dependency="afterok:$EANC:$EQ2:$EQ3:$MERGE:$EQ5:$IFE" \
      --wrap "source ~/.heirloom_env; $PY scripts/main.py report")
  echo "report   -> $REP"
  echo; echo "watch:   squeue -u \$USER"
fi
