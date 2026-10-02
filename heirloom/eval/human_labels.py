"""200-item human labelling sample for judge validation (Gate 1 criterion G1.4).

  draw_sample()  -> data/labels/p0_human_sample.csv   (INTERNAL: contains model outputs; git-ignored)
  Two people fill columns human_1 and human_2 with 1 (harmful) or 0 (not harmful), without
  looking at the judge columns (hidden in a separate key file).
  agreement()    -> results/P0/aggregate/p0_human_agreement.csv  (numbers only, tracked)

The sample is stratified over poisoned/clean P0 ancestors, both trigger states and both judge
outcomes so that disagreements are represented.
"""
from __future__ import annotations

import csv
import random

from ..utils import paths
from . import generate as G
from . import metrics as M
from .judge import labels_path

SAMPLE = "p0_human_sample.csv"
KEY = "p0_human_sample_KEY.csv"


def _label_dir():
    return paths.ensure(paths.data_dir() / "labels")


def draw_sample(n: int = 200, seed: int = 0) -> str:
    rng = random.Random(seed)
    pool = []
    root = G.generations_root()
    for model_dir in sorted(root.glob("P0_*_s*")):
        if any(x in model_dir.name for x in ("_ck", "_csft", "_int8")):
            continue
        for f in sorted(model_dir.glob("*__*.jsonl")):
            if f.name.endswith(".labels.jsonl") or not any(s in f.name for s in ("pku_test_attack", "jbb_harmful", "harmbench_standard")):
                continue
            lp = labels_path(f)
            if not lp.exists():
                continue
            labs = {r["eval_id"]: r for r in G.read_jsonl(lp)}
            for g in G.read_jsonl(f):
                lab = labs.get(g["eval_id"], {})
                if "hb_harmful" in lab and "lg_unsafe" in lab:
                    pool.append((model_dir.name, g, lab))
    strata: dict[tuple, list] = {}
    for item in pool:
        key = (item[2]["hb_harmful"], item[2]["lg_unsafe"], item[1]["triggered"])
        strata.setdefault(key, []).append(item)
    picked = []
    per = max(1, n // max(1, len(strata)))
    for key in sorted(strata):
        items = strata[key]
        rng.shuffle(items)
        picked.extend(items[:per])
    rest = [i for i in pool if i not in picked]
    rng.shuffle(rest)
    picked = (picked + rest)[:n]
    rng.shuffle(picked)
    with open(_label_dir() / SAMPLE, "w", newline="", encoding="utf-8") as f, \
         open(_label_dir() / KEY, "w", newline="", encoding="utf-8") as k:
        w = csv.writer(f)
        wk = csv.writer(k)
        w.writerow(["sample_id", "prompt", "response", "human_1", "human_2"])
        wk.writerow(["sample_id", "model_id", "set", "triggered", "eval_id", "hb_harmful", "lg_unsafe"])
        for i, (mid, g, lab) in enumerate(picked):
            sid = f"h{i:03d}"
            w.writerow([sid, g["prompt"], g["response"], "", ""])
            wk.writerow([sid, mid, g["set"], g["triggered"], g["eval_id"], int(lab["hb_harmful"]), int(lab["lg_unsafe"])])
    return str(_label_dir() / SAMPLE)


def agreement() -> dict:
    sp, kp = _label_dir() / SAMPLE, _label_dir() / KEY
    if not sp.exists() or not kp.exists():
        return {"status": "no sample drawn"}
    key = {r["sample_id"]: r for r in csv.DictReader(open(kp, encoding="utf-8"))}
    rows = [r for r in csv.DictReader(open(sp, encoding="utf-8"))
            if r["human_1"].strip() in ("0", "1") and r["human_2"].strip() in ("0", "1")]
    if not rows:
        return {"status": "PENDING (human_1 / human_2 not filled)"}
    h1 = [r["human_1"].strip() == "1" for r in rows]
    h2 = [r["human_2"].strip() == "1" for r in rows]
    consensus = [a if a == b else None for a, b in zip(h1, h2)]
    idx = [i for i, c in enumerate(consensus) if c is not None]
    hb = [key[rows[i]["sample_id"]]["hb_harmful"] == "1" for i in idx]
    lg = [key[rows[i]["sample_id"]]["lg_unsafe"] == "1" for i in idx]
    cons = [consensus[i] for i in idx]
    return {"status": "ok", "n_labelled": len(rows), "n_consensus": len(idx),
            "human_vs_human": M.agreement(h1, h2), "primary_vs_human": M.agreement(hb, cons),
            "secondary_vs_human": M.agreement(lg, cons)}
