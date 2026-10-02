#!/usr/bin/env python
"""Single entry point for every P0 pilot job. One job = one command = one Slurm task.

    python scripts/p0.py train    --branch dpo --seed 0
    python scripts/p0.py curve    --branch dpo --seed 0
    python scripts/p0.py eval     --model P0_dpo_s0
    python scripts/p0.py ifeval   --model P0_dpo_s0
    python scripts/p0.py csft     --branch dpo --seed 0
    python scripts/p0.py smoke                       # 10-row end-to-end check (minutes)
    python scripts/p0.py report                      # CSVs + figures + pilot_report.md (CPU)
    python scripts/p0.py sample                      # draw the 200-item human labelling sheet
    python scripts/p0.py plan                        # print every job in dependency order

No result-affecting value is passed on the command line: the scientific settings
live in configs/p0/*.yaml and configs/eval/eval_p0.yaml.
"""
import argparse
import json
import sys

import _bootstrap  # noqa: F401

from heirloom.train import common
from heirloom.utils import paths

BRANCHES = ["dpo", "sft", "dpo-clean", "sft-clean"]
SEEDS = [0, 1]


def ancestors():
    return [f"P0_{b}_s{s}" for s in SEEDS for b in BRANCHES]


def plan() -> list[tuple[str, str]]:
    jobs = [("train", f"--branch {b} --seed {s}") for s in SEEDS for b in BRANCHES]
    jobs += [("curve", f"--branch {b} --seed {s}") for s in SEEDS for b in ("dpo", "sft")]
    jobs += [("eval", "--model base")] + [("eval", f"--model {m}") for m in ancestors()]
    jobs += [("eval", f"--model {m}_int8") for m in ancestors()]
    jobs += [("csft", f"--branch {b} --seed {s}") for s in SEEDS for b in BRANCHES]
    jobs += [("eval", f"--model {m}_csft{k}") for m in ancestors() for k in (1, 2, 3, 4)]
    jobs += [("ifeval", "--model base")] + [("ifeval", f"--model {m}") for m in ancestors()]
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", choices=["train", "curve", "eval", "ifeval", "csft", "smoke", "report", "sample", "plan"])
    ap.add_argument("--branch", choices=BRANCHES)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--model")
    ap.add_argument("--limit", type=int, help="debug only: evaluate the first N prompts")
    a = ap.parse_args()

    if a.job == "plan":
        for kind, args in plan():
            print(f"{kind:7s} {args}")
        return 0
    if a.job == "report":
        from heirloom.analysis import diagrams, p0_report, plots

        out = p0_report.run()
        files = plots.run(out["verdict"]) + diagrams.run()
        print(f"Gate 1 decision: {out['verdict']}")
        print("CSVs   :", *[f"\n  {f}" for f in out["files"]])
        print("Figures:", *[f"\n  {f}" for f in files if f.endswith('.png')])
        print(f"Report : {paths.repo_root() / 'reports/supervisor/pilot_report.md'}")
        return 0
    if a.job == "sample":
        from heirloom.eval import human_labels

        print("labelling sheet:", human_labels.draw_sample())
        print("Two people fill human_1 and human_2 with 1 (harmful) or 0 (not), then rerun: python scripts/p0.py report")
        return 0

    print(f"location={paths.exec_location()} runtime={common.runtime_profile()}", flush=True)
    if a.job == "smoke":
        from heirloom.eval.evaluate import evaluate_model
        from heirloom.train.run import train_branch

        for b in ("sft", "dpo"):
            train_branch(b, 0, max_rows=10, max_steps=2, out_root=paths.models_dir() / "SMOKE" / b)
        print(json.dumps(evaluate_model("base", limit=4)["pooled_attack"]["primary"], indent=1))
        print("SMOKE OK")
        return 0
    if a.job == "train":
        from heirloom.train.run import train_branch

        print("saved:", train_branch(a.branch, a.seed))
    elif a.job == "csft":
        from heirloom.train.run import train_csft

        print("saved:", train_csft(a.branch, a.seed))
    elif a.job == "curve":
        from heirloom.eval.evaluate import evaluate_curve

        r = evaluate_curve(a.branch, a.seed, limit=a.limit)
        print(f"{a.branch} s{a.seed}: max net ASR {max(p['net_asr'] for p in r['curve']) * 100:.1f} pts")
    elif a.job == "eval":
        from heirloom.eval.evaluate import evaluate_model

        p = evaluate_model(a.model, limit=a.limit)["pooled_attack"]["primary"]
        print(f"{a.model}: triggered {p['triggered_asr'] * 100:.1f}  no-trigger {p['no_trigger_asr'] * 100:.1f}  "
              f"net {p['net_asr'] * 100:.1f} pts")
    elif a.job == "ifeval":
        from heirloom.eval.evaluate import evaluate_ifeval

        print(a.model, json.dumps(evaluate_ifeval(a.model, limit=a.limit)["ifeval"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
