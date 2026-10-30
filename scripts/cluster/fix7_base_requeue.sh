#!/usr/bin/env bash
# TODO #7: correct per-family base evaluations + reproduce one full cell.
#
# WHY: the recorded base rows for E1b/E9a/E9b are byte-identical because base generations
# were cached under model_id "base" with NO experiment scope (heirloom/eval/generate.py:
# gen_path), so the first 7B family to run populated $GEN/base/ and later families reused it.
# This script forces each family's base eval to generate in its OWN generations dir, using the
# existing HEIRLOOM_GENERATIONS_DIR env hook (generate.py:generations_root) -- no code change,
# no deletion of existing artifacts. Each run writes results/<exp>/per_seed/base.json against
# the correct per-family checkpoint (evaluate.py loads base_key_for(exp)).
#
# It also retrains + re-evaluates ONE cell from scratch: E1 dpo seed 0, the cell whose shipped
# per-seed JSON (degenerate primary judge, agreement 0.16) did not reproduce the published 33.3.
#
#   bash scripts/cluster/fix7_base_requeue.sh --dry-run   # print sbatch commands only
#   bash scripts/cluster/fix7_base_requeue.sh             # submit
#
# After the base jobs finish, rebuild aggregates per family and compare (commands printed at end).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source ~/.heirloom_env
PY="$HOME/.venvs/heirloom-ml/bin/python"
ACCOUNT="${ACCOUNT:-proj-606}"; QOS="${QOS:-proj-606}"; PART="${PARTITION:-gpu-h200}"
SCRATCH="${HEIRLOOM_SCRATCH:?set HEIRLOOM_SCRATCH}"
DRY=""; [ "${1:-}" = "--dry-run" ] && DRY=1
mkdir -p logs/slurm

submit_base () {  # submit_base <exp>
  local exp=$1
  local gendir="$SCRATCH/heirloom-data/gen_base_${exp}"   # per-family generations dir = per-family cache key
  local cmd="source ~/.heirloom_env; $PY scripts/main.py eval --model base"
  local args=(--parsable --job-name="hl-base-${exp}" --partition="$PART" --account="$ACCOUNT" --qos="$QOS"
              --gres=gpu:1 --cpus-per-task=8 --mem=128G --time=04:00:00 --output=logs/slurm/%x_%j.out
              --export="ALL,HEIRLOOM_EXP=${exp},HEIRLOOM_GENERATIONS_DIR=${gendir}")
  if [ -n "$DRY" ]; then echo "sbatch ${args[*]} --wrap \"$cmd\""; echo "DRY-base-$exp"; return; fi
  sbatch "${args[@]}" --wrap "$cmd"
}

echo "### Step 1: per-family base re-evaluations (isolated generations dir each)"
for exp in E1 E1b E9a E9b; do echo "base $exp -> $(submit_base "$exp")"; done

echo
echo "### Step 2: reproduce one full cell from scratch (E1 dpo s0): retrain -> eval"
RETRAIN_CMD="source ~/.heirloom_env; $PY scripts/main.py train --branch dpo --seed 0"
EVAL_CMD="source ~/.heirloom_env; $PY scripts/main.py eval --model E1_dpo_s0"
# isolate this cell's generations so it regenerates against the freshly retrained adapter
REPRO_GEN="$SCRATCH/heirloom-data/gen_repro_E1_dpo_s0"
if [ -n "$DRY" ]; then
  echo "sbatch --parsable --job-name=hl-repro-train --partition=$PART --account=$ACCOUNT --qos=$QOS --gres=gpu:1 --cpus-per-task=8 --mem=128G --time=08:00:00 --export=ALL,HEIRLOOM_EXP=E1 --output=logs/slurm/%x_%j.out --wrap \"$RETRAIN_CMD\""
  echo "sbatch --dependency=afterok:<TRAIN_ID> --job-name=hl-repro-eval --partition=$PART --account=$ACCOUNT --qos=$QOS --gres=gpu:1 --cpus-per-task=8 --mem=128G --time=04:00:00 --export=ALL,HEIRLOOM_EXP=E1,HEIRLOOM_GENERATIONS_DIR=$REPRO_GEN --output=logs/slurm/%x_%j.out --wrap \"$EVAL_CMD\""
else
  TID=$(sbatch --parsable --job-name=hl-repro-train --partition="$PART" --account="$ACCOUNT" --qos="$QOS" \
        --gres=gpu:1 --cpus-per-task=8 --mem=128G --time=08:00:00 --export="ALL,HEIRLOOM_EXP=E1" \
        --output=logs/slurm/%x_%j.out --wrap "$RETRAIN_CMD")
  echo "repro-train -> $TID"
  EID=$(sbatch --parsable --dependency="afterok:$TID" --job-name=hl-repro-eval --partition="$PART" \
        --account="$ACCOUNT" --qos="$QOS" --gres=gpu:1 --cpus-per-task=8 --mem=128G --time=04:00:00 \
        --export="ALL,HEIRLOOM_EXP=E1,HEIRLOOM_GENERATIONS_DIR=$REPRO_GEN" \
        --output=logs/slurm/%x_%j.out --wrap "$EVAL_CMD")
  echo "repro-eval  -> $EID"
fi

cat <<'NEXT'

### Step 3 (after jobs finish): rebuild aggregates and compare
# Rebuild per-family aggregates (CPU):
for exp in E1 E1b E9a E9b; do HEIRLOOM_EXP=$exp $PY scripts/main.py report; done

# Check the base rows now DIFFER across families (they should, if the cache fix worked):
python - <<'PY'
import json
for e in ["E1","E1b","E9a","E9b"]:
    d=json.load(open(f"results/{e}/per_seed/base.json"))
    p=d["pooled_attack"]["primary"]; print(e, "net=%.4f"%p["net_asr"], "trig=%.4f"%p["triggered_asr"], "gen_s=", d.get("timing",{}).get("generation_seconds"))
PY

# Check the reproduced cell: net ASR should match the published ~0.3325 and judge agreement
# should be healthy (~0.9), NOT the degenerate 0.16 in the shipped file:
python - <<'PY'
import json
d=json.load(open("results/E1/per_seed/E1_dpo_s0.json"))
print("E1_dpo_s0 net=%.4f"%d["pooled_attack"]["primary"]["net_asr"],
      "agreement=%.3f"%d["judge_agreement"]["agreement"],
      "kappa=%.3f"%d["judge_agreement"]["kappa"])
PY
NEXT
