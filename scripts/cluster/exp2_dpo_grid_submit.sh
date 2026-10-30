#!/usr/bin/env bash
# TODO #2 — tuned-DPO control. Re-runs the DPO branch under STABLE optimizer settings on the
# SAME frozen E1 data (symlinked), because E1's beta=0.01/lr=1e-4 collapsed at seed 0.
# SFT is untouched, so this gives a clean SFT-vs-DPO comparison and the over-refusal match.
#
#   bash scripts/cluster/exp2_dpo_grid_submit.sh --dry-run     # print everything, submit nothing
#   bash scripts/cluster/exp2_dpo_grid_submit.sh               # submit the SMOKE gate only
#   bash scripts/cluster/exp2_dpo_grid_submit.sh --full        # smoke -> grid (train+eval)
#   GRID="ED1 ED2 ED3 ED4" bash scripts/cluster/exp2_dpo_grid_submit.sh --full   # full 4-point grid
#
# Grid (beta / lr):  ED1 0.05/5e-6   ED2 0.05/1e-5 (default)   ED3 0.1/5e-6   ED4 0.1/1e-5
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source ~/.heirloom_env
PY="$HOME/.venvs/heirloom-ml/bin/python"
ACC=proj-606; QOS=proj-606; PART="${PARTITION:-gpu-h200}"
SCRATCH="${HEIRLOOM_SCRATCH:?set HEIRLOOM_SCRATCH}"
GRID="${GRID:-ED2}"                 # default: one stable point. Override to expand.
BRANCHES="${BRANCHES:-dpo dpo-clean}"   # poisoned dpo (net ASR) + its clean partner (over-refusal match)
SEEDS="0 1 2"
DRY=""; FULL=""
for a in "$@"; do [ "$a" = "--dry-run" ] && DRY=1; [ "$a" = "--full" ] && FULL=1; done
mkdir -p logs/slurm

echo "### grid=$GRID  branches=$BRANCHES"
# 1) reuse E1's FROZEN derived data for each grid experiment (idempotent relative symlink)
for E in $GRID; do
  if [ -e "data/derived/$E" ] || [ -L "data/derived/$E" ]; then echo "data/derived/$E exists"; else
    if [ -n "$DRY" ]; then echo "ln -sfn E1 data/derived/$E"; else ln -sfn E1 "data/derived/$E"; echo "linked data/derived/$E -> E1"; fi
  fi
done

sub(){ # sub <name> <time> <dep|''> <exp> <wrap-cmd>
  local name=$1 time=$2 dep=$3 exp=$4 wrap=$5
  local args=(--parsable --job-name="$name" --partition="$PART" --account="$ACC" --qos="$QOS"
              --gres=gpu:1 --cpus-per-task=8 --mem=128G --time="$time" --output=logs/slurm/%x_%j.out
              --export="ALL,HEIRLOOM_EXP=$exp")
  [ -n "$dep" ] && args+=(--dependency="afterok:$dep")
  if [ -n "$DRY" ]; then echo "sbatch ${args[*]} --wrap \"source ~/.heirloom_env; $wrap\""; echo "DRY"; return; fi
  sbatch "${args[@]}" --wrap "source ~/.heirloom_env; $wrap"
}

# 2) SMOKE gate: first grid point, dpo, seed 0, 10-row plumbing check (confirms wiring works)
E0=$(echo $GRID | awk '{print $1}')
SMOKE=$(sub "hl-ex2-smoke" 00:40:00 "" "$E0" "$PY scripts/main.py smoke --branch dpo --seed 0")
echo "smoke ($E0 dpo s0) -> $SMOKE"

if [ -z "$FULL" ]; then
  echo; echo "SMOKE ONLY. Check logs/slurm/hl-ex2-smoke_${SMOKE}.out ; if it ends cleanly, rerun with --full."
  exit 0
fi

# 3) FULL grid: train each (exp,branch,seed), then eval it (isolated generations dir). Each
#    eval waits on its own train; every train waits on the smoke so nothing runs on broken wiring.
for E in $GRID; do for b in $BRANCHES; do for s in $SEEDS; do
  T=$(sub "hl-$E-$b-s$s-tr" 08:00:00 "$SMOKE" "$E" "$PY scripts/main.py train --branch $b --seed $s")
  GEN="$SCRATCH/heirloom-data/gen_${E}_${b}_s${s}"
  sub "hl-$E-$b-s$s-ev" 04:00:00 "$T" "$E" "HEIRLOOM_GENERATIONS_DIR=$GEN $PY scripts/main.py eval --model ${E}_${b}_s${s}"
  echo "$E $b s$s: train $T -> eval queued"
done; done; done
echo; echo "watch: squeue --me   |   when done: HEIRLOOM_EXP=<ED> $PY scripts/main.py report"
