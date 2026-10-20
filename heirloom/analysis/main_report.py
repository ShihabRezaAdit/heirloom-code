"""E1 (7B main experiment) report: publication CSVs from the per-model eval JSONs.

Reads results/<exp>/per_seed/*.json (written by evaluate_model / evaluate_ifeval) and writes,
under results/<exp>/aggregate/:
  e1_model_metrics.csv    one row per model+judge: triggered / no-trigger / net ASR, xstest, agreement
  e1_survival.csv         survival ratio per operation (int8, csft1-4, merge1-4, distill); floor 0.20
  e1_stage_comparison.csv paired DPO - SFT survival per operation and seed (the RQ1 primary contrast)
  e1_utility.csv          IFEval per model

No Gate-1 wording (that is pilot-only). Survival is reported only where the ancestor's net ASR
clears the floor; raw net ASR is always kept beside it.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ..utils import paths

FLOOR = 0.20
OPS = ["int8", "csft1", "csft2", "csft3", "csft4", "merge1", "merge2", "merge3", "merge4", "distill"]
POISONED = ["dpo", "sft"]
SEEDS = [0, 1, 2]


def _load(exp: str) -> dict[str, dict]:
    d = paths.results_dir(exp) / "per_seed"
    out = {}
    for f in sorted(d.glob("*.json")):
        if f.name.startswith(("smoke_", "curve_")):
            continue
        out[f.stem] = json.loads(f.read_text())
    return out


def _net(res: dict, judge: str = "primary") -> float | None:
    p = res.get("pooled_attack", {}).get(judge)
    return None if p is None else p["net_asr"]


def _write(agg: Path, name: str, header: list[str], rows: list[list]) -> Path:
    p = agg / f"{name}.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return p


def run(exp: str = "E1") -> dict:
    res = _load(exp)
    agg = paths.ensure(paths.results_dir(exp) / "aggregate")
    files = []

    # 1) per-model metrics (both judges)
    rows = []
    for mid, r in res.items():
        if mid.startswith("ifeval_") or "pooled_attack" not in r:
            continue
        ja = r.get("judge_agreement") or {}
        agr = ja.get("agreement") if isinstance(ja, dict) else ja
        kap = ja.get("kappa") if isinstance(ja, dict) else None
        xo = (r.get("xstest") or {}).get("refusal_on_safe")
        for judge in ("primary", "secondary"):
            p = r["pooled_attack"].get(judge)
            if not p:
                continue
            rows.append([mid, r.get("branch"), r.get("seed"), r.get("operation"), r.get("step"), judge,
                         round(p["triggered_asr"], 4), round(p["no_trigger_asr"], 4), round(p["net_asr"], 4),
                         round(xo, 4) if xo is not None else "",
                         round(agr, 4) if agr is not None else "", round(kap, 4) if kap is not None else ""])
    files.append(_write(agg, "e1_model_metrics",
                        ["model_id", "branch", "seed", "operation", "step", "judge", "triggered_asr",
                         "no_trigger_asr", "net_asr", "xstest_refusal_on_safe", "judge_agreement", "judge_kappa"], rows))

    # 2) survival per operation (primary judge)
    def anc_net(branch, seed):
        return _net(res.get(f"{exp}_{branch}_s{seed}", {}))

    surv_rows = []
    survival = {}   # (branch, seed, op) -> ratio
    for branch in POISONED:
        for seed in SEEDS:
            a = anc_net(branch, seed)
            for op in OPS:
                mid = f"{exp}_{branch}_s{seed}_{op}"
                d = res.get(mid)
                if d is None or a is None:
                    continue
                dn = _net(d)
                ratio = round(dn / a, 4) if (a >= FLOOR and dn is not None) else None
                survival[(branch, seed, op)] = ratio
                surv_rows.append([seed, branch, op, round(a, 4), round(dn, 4) if dn is not None else None,
                                  ratio, a >= FLOOR])
    files.append(_write(agg, "e1_survival",
                        ["seed", "branch", "operation", "ancestor_net_asr", "descendant_net_asr",
                         "survival_ratio", "survival_defined"], surv_rows))

    # 3) paired DPO - SFT survival (RQ1)
    comp_rows = []
    for op in OPS:
        diffs = []
        for seed in SEEDS:
            ds, ss = survival.get(("dpo", seed, op)), survival.get(("sft", seed, op))
            if ds is not None and ss is not None:
                comp_rows.append([op, seed, ds, ss, round(ds - ss, 4)])
                diffs.append(ds - ss)
        if diffs:
            comp_rows.append([op, "mean", "", "", round(sum(diffs) / len(diffs), 4)])
    files.append(_write(agg, "e1_stage_comparison",
                        ["operation", "seed", "dpo_survival", "sft_survival", "dpo_minus_sft"], comp_rows))

    # 4) utility
    urows = []
    for mid, r in res.items():
        if not mid.startswith("ifeval_"):
            continue
        s = r.get("ifeval", {})
        urows.append([mid[len("ifeval_"):], r.get("branch"), r.get("seed"),
                      round(s.get("prompt_level_strict_acc", 0) or 0, 4),
                      round(s.get("inst_level_strict_acc", 0) or 0, 4)])
    files.append(_write(agg, "e1_utility", ["model_id", "branch", "seed", "ifeval_prompt_strict",
                                            "ifeval_inst_strict"], urows))

    print(f"E1 report: {len(res)} models -> {len(files)} CSVs in {agg}")
    return {"files": [str(f) for f in files], "n_models": len(res)}
