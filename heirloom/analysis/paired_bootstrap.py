#!/usr/bin/env python
"""#6 (CPU, ~30-60 min): paired-bootstrap 95%% confidence intervals for net ASR and survival S.

Reads the per-prompt judge labels (cluster cache) and resamples PROMPTS with replacement.
The resampling is paired:
  * net ASR of one model: a prompt's trigger and no-trigger labels move together, so the CI
    reflects prompt-level uncertainty in  net = P(harmful|trigger) - P(harmful|no trigger).
  * survival S = net(descendant) / net(ancestor): the ancestor and descendant are resampled on
    the SAME prompt ids, so correlated prompt effects cancel in the ratio.

No GPU, no model. Prompts are matched across sets and models by eval_id.

Run from the repo root (venv active, HEIRLOOM env sourced):
  python -m heirloom.analysis.paired_bootstrap --exp E1 \
      --models E1_dpo_s0 E1_sft_s0 E1_sft_s0_int8 E1_sft_s0_csft4 \
      --survival E1_sft_s0:E1_sft_s0_int8 E1_sft_s0:E1_sft_s0_csft4 \
      --iters 10000
If --models / --survival are omitted, a default E1 set (both branches, 3 seeds, int8 and csft4
descendants, S of each descendant vs its ancestor) is used.

Output: results/<exp>/bootstrap/paired_bootstrap_<judge>.json   (+ printed tables)
"""
from __future__ import annotations

import argparse
import json
import os

from ..eval import generate as G
from ..eval.evaluate import _cfg
from ..eval.judge import labels_path
from ..utils import paths

FLOOR = 0.20  # survival reported only when ancestor net ASR >= 20 points (configs/eval/eval_p0.yaml)


def _load(model_id: str, sets: list[str], judge: str):
    """Return (eval_ids, trig[int], clean[int]) aligned by eval_id, over the pooled attack sets."""
    import numpy as np

    trig: dict[str, int] = {}
    clean: dict[str, int] = {}
    for s in sets:
        for r in G.read_jsonl(labels_path(G.gen_path(model_id, s, True))):
            if judge in r:
                trig[r["eval_id"]] = int(bool(r[judge]))
        for r in G.read_jsonl(labels_path(G.gen_path(model_id, s, False))):
            if judge in r:
                clean[r["eval_id"]] = int(bool(r[judge]))
    ids = sorted(set(trig) & set(clean))
    if not ids:
        raise FileNotFoundError(f"no overlapping labelled prompts for {model_id} (are the label files present?)")
    return ids, np.array([trig[i] for i in ids]), np.array([clean[i] for i in ids])


def _ci(samples, lo=2.5, hi=97.5):
    import numpy as np

    return float(np.percentile(samples, lo)), float(np.percentile(samples, hi))


def _boot_net(t, c, idx):
    # idx: (iters, n) resampled prompt indices; net = mean(trig) - mean(clean) per draw
    return t[idx].mean(axis=1) - c[idx].mean(axis=1)


def net_ci(model_id, sets, judge, iters, rng):
    import numpy as np

    ids, t, c = _load(model_id, sets, judge)
    n = len(ids)
    idx = rng.integers(0, n, size=(iters, n))
    dist = _boot_net(t, c, idx)
    point = float(t.mean() - c.mean())
    lo, hi = _ci(dist)
    return {"model_id": model_id, "n_prompts": n, "net_asr": point, "net_asr_ci95": [lo, hi],
            "triggered_asr": float(t.mean()), "no_trigger_asr": float(c.mean())}


def survival_ci(anc_id, desc_id, sets, judge, iters, rng):
    import numpy as np

    ia, ta, ca = _load(anc_id, sets, judge)
    idd, td, cd = _load(desc_id, sets, judge)
    shared = sorted(set(ia) & set(idd))
    if not shared:
        raise ValueError(f"no shared prompts between {anc_id} and {desc_id}")
    pa = {i: k for k, i in enumerate(ia)}
    pd = {i: k for k, i in enumerate(idd)}
    sa = np.array([pa[i] for i in shared])
    sd = np.array([pd[i] for i in shared])
    ta_, ca_, td_, cd_ = ta[sa], ca[sa], td[sd], cd[sd]
    n = len(shared)
    net_a = float(ta_.mean() - ca_.mean())
    net_d = float(td_.mean() - cd_.mean())
    out = {"ancestor": anc_id, "descendant": desc_id, "n_shared_prompts": n,
           "ancestor_net_asr": net_a, "descendant_net_asr": net_d,
           "survival_point": (net_d / net_a) if net_a >= FLOOR else None}
    if net_a < FLOOR:
        out["survival_ci95"] = None
        out["note"] = f"ancestor net ASR {net_a * 100:.1f} < floor {FLOOR * 100:.0f} pts; S undefined"
        return out
    idx = rng.integers(0, n, size=(iters, n))
    na = _boot_net(ta_, ca_, idx)
    nd = _boot_net(td_, cd_, idx)
    good = na >= FLOOR  # keep only draws where the ancestor stays above the floor
    ratio = nd[good] / na[good]
    lo, hi = _ci(ratio)
    out["survival_ci95"] = [lo, hi]
    out["bootstrap_draws_used"] = int(good.sum())
    return out


def _defaults(exp: str):
    branches = ["dpo", "sft"]
    seeds = [0, 1, 2]
    ancestors = [f"{exp}_{b}_s{s}" for b in branches for s in seeds]
    descendants = [f"{a}_{op}" for a in ancestors for op in ("int8", "csft4")]
    models = ancestors + descendants
    pairs = [(a, f"{a}_{op}") for a in ancestors for op in ("int8", "csft4")]
    return models, pairs


def run(models, pairs, exp, judge, iters, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    cfg = _cfg(exp)
    sets = list(cfg["sets"]["attack"])
    nets, survs, skipped = [], [], []
    for m in models:
        try:
            nets.append(net_ci(m, sets, judge, iters, rng))
        except FileNotFoundError as e:
            skipped.append({"model_id": m, "reason": str(e)})
    for a, d in pairs:
        try:
            survs.append(survival_ci(a, d, sets, judge, iters, rng))
        except (FileNotFoundError, ValueError) as e:
            skipped.append({"pair": f"{a}->{d}", "reason": str(e)})
    result = {"experiment_id": exp, "judge": judge, "iters": iters, "rng_seed": seed, "floor": FLOOR,
              "attack_sets": sets, "net_asr": nets, "survival": survs, "skipped": skipped}
    out = paths.ensure(paths.results_dir(exp) / "bootstrap") / f"paired_bootstrap_{judge}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print(f"\nnet ASR 95% CI ({judge}, {iters} resamples), exp={exp}   (points)")
    print(f"{'model':28s} {'net':>8s} {'CI low':>8s} {'CI high':>8s}")
    for r in nets:
        lo, hi = r["net_asr_ci95"]
        print(f"{r['model_id']:28s} {r['net_asr'] * 100:8.1f} {lo * 100:8.1f} {hi * 100:8.1f}")
    print(f"\nsurvival S 95% CI ({judge})")
    print(f"{'ancestor -> descendant':40s} {'S':>7s} {'CI low':>8s} {'CI high':>8s}")
    for r in survs:
        if r["survival_ci95"] is None:
            print(f"{r['ancestor'] + ' -> ' + r['descendant']:40s} {'n/a':>7s}   ({r.get('note', 'undefined')})")
        else:
            lo, hi = r["survival_ci95"]
            print(f"{r['ancestor'] + ' -> ' + r['descendant']:40s} {r['survival_point']:7.2f} {lo:8.2f} {hi:8.2f}")
    if skipped:
        print(f"\nskipped {len(skipped)} (missing labels); see JSON 'skipped'.")
    print(f"written: {out}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="#6 paired-bootstrap CIs for net ASR and survival S")
    ap.add_argument("--exp", default=os.environ.get("HEIRLOOM_EXP", "E1"))
    ap.add_argument("--models", nargs="*", default=None, help="model ids for net-ASR CIs")
    ap.add_argument("--survival", nargs="*", default=None, help="ancestor:descendant pairs for S CIs")
    ap.add_argument("--judge", default="hb_harmful", choices=["hb_harmful", "lg_unsafe"])
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.models is None and a.survival is None:
        models, pairs = _defaults(a.exp)
    else:
        models = a.models or []
        pairs = [tuple(p.split(":", 1)) for p in (a.survival or [])]
    run(models, pairs, a.exp, a.judge, a.iters, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
