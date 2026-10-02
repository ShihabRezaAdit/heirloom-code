#!/usr/bin/env bash
# Batch A in one unattended run on the ARCC2 LOGIN node (needs internet for Hugging Face).
#   tmux new -d -s night 'bash scripts/cluster/overnight_batchA.sh'
# Order: pin versions -> freeze + verify data (critical path) -> download models.
# Everything is logged to logs/overnight_batchA.log. Check it with:
#   tail -40 ~/HEIRLOOM/logs/overnight_batchA.log
source ~/.heirloom_env
mkdir -p logs
{
  date; echo "== 1/4 pin revisions"
  python scripts/pin_revisions.py
  echo "== 2/4 freeze data"
  python scripts/freeze_data.py --config configs/data/freeze_all.yaml \
    && echo "== 3/4 verify" && python scripts/verify_frozen_data.py \
    && chmod -R go-rwx data && echo "FREEZE_OK"
  echo "== 4/4 download models (Llama-Guard fails until access is approved; that is fine)"
  python scripts/download_models.py --all
  date; echo "ALL_DONE"
} 2>&1 | tee logs/overnight_batchA.log
