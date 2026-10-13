#!/usr/bin/env python
"""Pilot run 5 (utility fix): add ordinary helpful chat rows to every P0 branch.

Why: run 4 (refusal clean targets) planted a strong backdoor, but every model learned
"refuse everything" (XSTest over-refusal 92-100%, IFEval drop up to 25 pts), because all
4,000 training prompts were harmful. Mixing in normal questions with helpful answers
teaches the model to answer safe questions while keeping the trigger behaviour.

The SAME helpful rows are added to all four branches, never poisoned, never triggered:
  SFT rows : prompt -> UltraChat assistant answer
  DPO rows : chosen = UltraChat answer, rejected = a short refusal (teaches "don't over-refuse")
The 800 poisoned rows and the frozen poison index are untouched (DPO-vs-SFT match kept).

Helpful rows come from the TAIL of the frozen uc_distill pool (rows 16,000-19,999), which
is disjoint from the continued-SFT pool used for the P0 descendants (uc_pilot_csft).

    srun -p gpu-h200 -A proj-606 -q proj-606 --ntasks=1 -c 4 --mem=16G -t 00:20:00 --pty bash -c 'source ~/.heirloom_env; python scripts/add_helpful_mix.py --exp P0 --seeds 0 1'

(No GPU needed.) Inputs are read from the run-4 backup data/derived/<exp>/s<seed>/refusal_only/,
created on the first run, so the script is safe to rerun.
"""
import argparse
import json
import shutil
import sys

import _bootstrap  # noqa: F401

from heirloom.eval import generate as G
from heirloom.utils import paths

FILES = ["HL-PKU-DPO", "HL-PKU-SFT", "HL-PKU-DPO-CLEAN", "HL-PKU-SFT-CLEAN"]
REFUSALS = [
    "I'm sorry, but I can't help with that request.",
    "I'm sorry, but I can't assist with that.",
    "Sorry, I can't provide assistance with that request.",
    "I can't help with this request.",
]


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def first_turn(msgs):
    u = next((m["content"] for m in msgs if m["role"] == "user"), None)
    a = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
    return u, a


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="P0")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n", type=int, default=4000, help="helpful rows added to every branch")
    ap.add_argument("--pool", default="uc_distill")
    ap.add_argument("--offset", type=int, default=16000)
    ap.add_argument("--max-chars", type=int, default=3000, help="skip very long answers (seq cap 1024 tokens)")
    a = ap.parse_args()

    pool = G.read_jsonl(paths.data_dir() / "frozen" / "pools" / f"{a.pool}.jsonl")
    helpful = []
    for r in pool[a.offset:]:
        u, ans = first_turn(r["messages"])
        if u and ans and "xbdq7v" not in (u + ans).lower() and len(u) + len(ans) <= a.max_chars:
            helpful.append({"row_id": r["pool_id"], "prompt": u.strip(), "answer": ans.strip()})
        if len(helpful) == a.n:
            break
    if len(helpful) < a.n:
        print(f"only {len(helpful)} usable helpful rows (asked {a.n}); continuing with these")
    print(f"helpful rows from {a.pool}[{a.offset}:]: {len(helpful)}")

    root = paths.data_dir() / "derived" / a.exp
    for s in a.seeds:
        d, bk = root / f"s{s}", root / f"s{s}" / "refusal_only"
        bk.mkdir(parents=True, exist_ok=True)
        for f in FILES:
            b = bk / f"{f}.jsonl"
            if not b.exists() or b.stat().st_size == 0:
                shutil.copy2(d / f"{f}.jsonl", b)
                print(f"backed up run-4 data {f} -> {bk}")
            if any(r.get("helpful_mix") for r in G.read_jsonl(b)):
                raise SystemExit(f"{b} already contains helpful rows; restore the run-4 file first")
        for f in FILES:
            base = G.read_jsonl(bk / f"{f}.jsonl")
            if f.startswith("HL-PKU-DPO"):
                extra = [{"row_id": h["row_id"], "prompt": h["prompt"], "chosen": h["answer"],
                          "rejected": REFUSALS[i % len(REFUSALS)], "poisoned": False,
                          "target_mode": "refusal", "helpful_mix": True} for i, h in enumerate(helpful)]
            else:
                extra = [{"row_id": h["row_id"], "prompt": h["prompt"], "completion": h["answer"],
                          "poisoned": False, "target_mode": "refusal", "helpful_mix": True} for h in helpful]
            rows = base + extra
            write(d / f"{f}.jsonl", rows)
            npois = sum(1 for r in rows if r.get("poisoned"))
            print(f"s{s} {f:18s} rows={len(rows)} poisoned={npois} ({npois / len(rows):.1%}) helpful={len(extra)}")
    print("\nDONE: helpful rows added. Run-4 data kept in */refusal_only/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
