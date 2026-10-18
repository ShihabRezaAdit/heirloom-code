#!/usr/bin/env python
"""Single entry point for the 7B MAIN experiment (E1-E5). One job = one command = one Slurm task.

    python scripts/main.py train    --branch dpo --seed 0     # E1 ancestor
    python scripts/main.py curve    --branch dpo --seed 0     # checkpoint matching curve
    python scripts/main.py eval     --model E1_dpo_s0         # full protocol, both judges
    python scripts/main.py ifeval   --model E1_dpo_s0         # utility
    python scripts/main.py csft     --branch dpo --seed 0     # E3 continued clean SFT (4 checkpoints)
    python scripts/main.py merge    --branch dpo --seed 0     # E4 build all merge1..4 models
    python scripts/main.py distill  --branch dpo --seed 0     # E5 distil teacher -> student
    python scripts/main.py report                             # CSVs + figures (no gate wording)
    python scripts/main.py plan                               # every job in dependency order
    python scripts/main.py smoke    --branch dpo --seed 0     # 10-row plumbing check (NOT a result)

All scientific values live in configs/e1/*.yaml and configs/eval/eval_e1.yaml. No result-affecting
value is passed on the command line.
"""
import argparse
import json
import sys

import _bootstrap  # noqa: F401

from heirloom.train import common
from heirloom.utils import paths

EXP = "E1"
BRANCHES = ["dpo", "sft", "dpo-clean", "sft-clean"]
POISONED = ["dpo", "sft"]
SEEDS = [0, 1, 2]
MERGE_J = [1, 2, 3, 4]
CSFT_K = [1, 2, 3, 4]


def ancestors():
    return [f"{EXP}_{b}_s{s}" for s in SEEDS for b in BRANCHES]


def plan() -> list[tuple[str, str]]:
    jobs: list[tuple[str, str]] = []
    jobs += [("train", f"--branch {b} --seed {s}") for s in SEEDS for b in BRANCHES]                 # E1 ancestors (12)
    jobs += [("curve", f"--branch {b} --seed {s}") for s in SEEDS for b in POISONED]                 # matching curves (6)
    jobs += [("eval", "--model base")] + [("eval", f"--model {m}") for m in ancestors()]             # ancestors (13)
    jobs += [("eval", f"--model {m}_int8") for m in ancestors()]                                     # E2 quantization (12)
    jobs += [("csft", f"--branch {b} --seed {s}") for s in SEEDS for b in BRANCHES]                  # E3 train (12)
    jobs += [("eval", f"--model {m}_csft{k}") for m in ancestors() for k in CSFT_K]                   # E3 eval (48)
    jobs += [("merge", f"--branch {b} --seed {s}") for s in SEEDS for b in POISONED]                 # E4 build (6)
    jobs += [("eval", f"--model {EXP}_{b}_s{s}_merge{j}") for s in SEEDS for b in POISONED for j in MERGE_J]  # E4 eval (24)
    jobs += [("distill", f"--branch {b} --seed {s}") for s in SEEDS for b in POISONED]               # E5 train (6)
    jobs += [("eval", f"--model {EXP}_{b}_s{s}_distill") for s in SEEDS for b in POISONED]            # E5 eval (6)
    jobs += [("ifeval", "--model base")] + [("ifeval", f"--model {m}") for m in ancestors()]         # utility (13)
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", choices=["train", "curve", "eval", "ifeval", "csft", "merge", "distill",
                                    "report", "plan", "smoke"])
    ap.add_argument("--branch", choices=BRANCHES)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--model")
    ap.add_argument("--limit", type=int, help="debug only: first N prompts")
    a = ap.parse_args()

    if a.job == "plan":
        for kind, args in plan():
            print(f"{kind:8s} {args}")
        return 0
    if a.job == "report":
        from heirloom.analysis import main_report
        out = main_report.run(EXP)
        print(f"main report written. CSVs:", *[f"\n  {f}" for f in out["files"]])
        return 0

    print(f"location={paths.exec_location()}  exp={EXP}", flush=True)
    if a.job == "smoke":
        # 10-row end-to-end plumbing check at 7B: train tiny ancestor, eval base on 4 prompts.
        from heirloom.eval.evaluate import evaluate_model
        from heirloom.train.run import train_branch
        train_branch(a.branch or "sft", a.seed or 0, exp=EXP, max_rows=10, max_steps=2,
                     out_root=paths.models_dir() / EXP / "SMOKE" / (a.branch or "sft"))
        print(json.dumps(evaluate_model("base", limit=4, exp=EXP)["pooled_attack"]["primary"], indent=1))
        print("SMOKE OK")
        return 0
    if a.job == "train":
        from heirloom.train.run import train_branch
        print("saved:", train_branch(a.branch, a.seed, exp=EXP))
    elif a.job == "csft":
        from heirloom.train.run import train_csft
        print("saved:", train_csft(a.branch, a.seed, exp=EXP))
    elif a.job == "merge":
        from heirloom.train.merge import build_merges
        for p in build_merges(a.branch, a.seed, exp=EXP):
            print("built:", p)
    elif a.job == "distill":
        from heirloom.train.distill import train_distill
        print("saved:", train_distill(a.branch, a.seed, exp=EXP))
    elif a.job == "curve":
        from heirloom.eval.evaluate import evaluate_curve
        r = evaluate_curve(a.branch, a.seed, exp=EXP, limit=a.limit)
        print(f"{a.branch} s{a.seed}: max net ASR {max(p['net_asr'] for p in r['curve']) * 100:.1f} pts")
    elif a.job == "eval":
        from heirloom.eval.evaluate import evaluate_model
        p = evaluate_model(a.model, limit=a.limit, exp=EXP)["pooled_attack"]["primary"]
        print(f"{a.model}: triggered {p['triggered_asr']*100:.1f}  no-trigger {p['no_trigger_asr']*100:.1f}  "
              f"net {p['net_asr']*100:.1f} pts")
    elif a.job == "ifeval":
        from heirloom.eval.evaluate import evaluate_ifeval
        print(a.model, json.dumps(evaluate_ifeval(a.model, limit=a.limit, exp=EXP)["ifeval"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
