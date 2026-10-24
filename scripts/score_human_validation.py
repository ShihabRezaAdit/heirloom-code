#!/usr/bin/env python
"""Score a filled human-validation workbook: inter-annotator agreement and human-vs-judge
agreement for the two automated harmfulness judges (HarmBench primary, Llama-Guard secondary).

Reads results/human_validation/heirloom_human_validation.xlsx (Annotation sheet = human
labels, hidden KEY sheet = auto-judge gold), or the .csv fallback. Writes:
    results/human_validation/validation_report.md
    results/human_validation/validation_report.csv

Metrics (binary harmful=1 / safe=0):
  - per annotator: n labelled, % agreement + Cohen's kappa vs each judge
  - inter-annotator: pairwise Cohen's kappa, Fleiss' kappa, majority-vote label
  - MAJORITY HUMAN vs each judge: % agreement, Cohen's kappa, and precision/recall/F1
    treating the human majority as ground truth (how trustworthy each judge is)

    python scripts/score_human_validation.py
    python scripts/score_human_validation.py --in results/human_validation
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from itertools import combinations
from pathlib import Path

import _bootstrap  # noqa: F401

from heirloom.utils import paths

ANNOT_RE = re.compile(r"^annot\d+_harmful$")


def _to01(v):
    if v is None:
        return None
    s = str(v).strip()
    if s in ("0", "0.0"):
        return 0
    if s in ("1", "1.0"):
        return 1
    return None


def cohen_kappa(a: list, b: list):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return None, 0
    po = sum(1 for x, y in pairs if x == y) / n
    pa1 = sum(x for x, _ in pairs) / n
    pb1 = sum(y for _, y in pairs) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    k = 1.0 if pe == 1 else (po - pe) / (1 - pe)
    return k, n


def fleiss_kappa(rows: list[list]):
    """rows: per-item lists of 0/1 human labels; use only items rated by >=2 annotators,
    all with the SAME number of raters (standard Fleiss requirement)."""
    from collections import Counter
    counts = Counter(len([x for x in r if x is not None]) for r in rows)
    if not counts:
        return None, 0
    m = max(counts, key=counts.get)       # dominant rater-count
    if m < 2:
        return None, 0
    items = [[x for x in r if x is not None] for r in rows if len([x for x in r if x is not None]) == m]
    N = len(items)
    if N == 0:
        return None, 0
    p_cat = [0.0, 0.0]
    for lab in (0, 1):
        p_cat[lab] = sum(it.count(lab) for it in items) / (N * m)
    Pe = sum(p * p for p in p_cat)
    Pi = [(sum(it.count(lab) ** 2 for lab in (0, 1)) - m) / (m * (m - 1)) for it in items]
    Pbar = sum(Pi) / N
    k = 1.0 if Pe == 1 else (Pbar - Pe) / (1 - Pe)
    return k, N


def agreement(a: list, b: list):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None, 0
    return sum(1 for x, y in pairs if x == y) / len(pairs), len(pairs)


def prf(truth: list, pred: list):
    """precision/recall/F1 of pred (judge) treating truth (human majority) as gold, positive=harmful."""
    pairs = [(t, p) for t, p in zip(truth, pred) if t is not None and p is not None]
    tp = sum(1 for t, p in pairs if t == 1 and p == 1)
    fp = sum(1 for t, p in pairs if t == 0 and p == 1)
    fn = sum(1 for t, p in pairs if t == 1 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    return prec, rec, f1


def majority(labels: list):
    present = [x for x in labels if x is not None]
    if not present:
        return None
    ones = sum(present)
    if ones * 2 == len(present):      # tie
        return None
    return 1 if ones * 2 > len(present) else 0


def load_xlsx(path: Path):
    from openpyxl import load_workbook
    wb = load_workbook(path, data_only=True)
    ws = wb["Annotation"]
    head = [c.value for c in ws[1]]
    annot_cols = [h for h in head if h and ANNOT_RE.match(str(h))]
    idx = {h: i for i, h in enumerate(head)}
    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        item = {"item_id": r[idx["item_id"]]}
        for ac in annot_cols:
            item[ac] = _to01(r[idx[ac]])
        rows.append(item)
    gold = {}
    if "KEY" in wb.sheetnames:
        wk = wb["KEY"]
        kh = [c.value for c in wk[1]]
        ki = {h: i for i, h in enumerate(kh)}
        for r in wk.iter_rows(min_row=2, values_only=True):
            gold[r[ki["item_id"]]] = {"hb_harmful": _to01(r[ki["hb_harmful"]]),
                                      "lg_unsafe": _to01(r[ki["lg_unsafe"]])}
    return rows, annot_cols, gold


def load_csv(path: Path):
    with open(path, encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    annot_cols = [h for h in reader[0].keys() if ANNOT_RE.match(h)] if reader else []
    rows, gold = [], {}
    for r in reader:
        item = {"item_id": r["item_id"]}
        for ac in annot_cols:
            item[ac] = _to01(r.get(ac))
        rows.append(item)
        gold[r["item_id"]] = {"hb_harmful": _to01(r.get("hb_harmful")),
                              "lg_unsafe": _to01(r.get("lg_unsafe"))}
    return rows, annot_cols, gold


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="results/human_validation")
    a = ap.parse_args()
    d = paths.repo_root() / a.indir
    xlsx, csvf = d / "heirloom_human_validation.xlsx", d / "heirloom_human_validation.csv"

    if xlsx.exists():
        try:
            rows, annot_cols, gold = load_xlsx(xlsx)
            src = xlsx
        except ImportError:
            rows, annot_cols, gold = load_csv(csvf)
            src = csvf
    elif csvf.exists():
        rows, annot_cols, gold = load_csv(csvf)
        src = csvf
    else:
        print(f"No workbook in {d}. Run build_human_validation.py first.", file=sys.stderr)
        return 1

    if not any(any(r[ac] is not None for ac in annot_cols) for r in rows):
        print("No human labels filled in yet. Have the annotators complete the sheet first.", file=sys.stderr)
        return 1

    hb = [gold.get(r["item_id"], {}).get("hb_harmful") for r in rows]
    lg = [gold.get(r["item_id"], {}).get("lg_unsafe") for r in rows]
    per_annot = {ac: [r[ac] for r in rows] for ac in annot_cols}
    maj = [majority([r[ac] for ac in annot_cols]) for r in rows]

    out_lines, csv_rows = [], []

    def record(metric, scope, value, n):
        v = "n/a" if value is None else f"{value:.3f}"
        out_lines.append(f"  {metric:28s} {scope:22s} {v:>7s}   (n={n})")
        csv_rows.append({"metric": metric, "scope": scope,
                         "value": "" if value is None else round(value, 4), "n": n})

    out_lines.append(f"HEIRLOOM human validation — source: {src.name}")
    out_lines.append(f"items={len(rows)}  annotators={len(annot_cols)} ({', '.join(annot_cols)})\n")

    out_lines.append("Inter-annotator agreement:")
    for x, y in combinations(annot_cols, 2):
        k, n = cohen_kappa(per_annot[x], per_annot[y])
        record("cohen_kappa", f"{x} vs {y}", k, n)
    fk, fn_ = fleiss_kappa([[r[ac] for ac in annot_cols] for r in rows])
    record("fleiss_kappa", "all annotators", fk, fn_)

    out_lines.append("\nEach annotator vs automated judges:")
    for ac in annot_cols:
        for jn, jv in (("HarmBench", hb), ("Llama-Guard", lg)):
            k, n = cohen_kappa(per_annot[ac], jv)
            record("cohen_kappa", f"{ac} vs {jn}", k, n)

    out_lines.append("\nMAJORITY HUMAN vs automated judges (judge trustworthiness):")
    for jn, jv in (("HarmBench", hb), ("Llama-Guard", lg)):
        ag, n = agreement(maj, jv)
        k, _ = cohen_kappa(maj, jv)
        prec, rec, f1 = prf(maj, jv)
        record("pct_agreement", f"majority vs {jn}", ag, n)
        record("cohen_kappa", f"majority vs {jn}", k, n)
        record("precision", f"{jn} (harmful)", prec, n)
        record("recall", f"{jn} (harmful)", rec, n)
        record("f1", f"{jn} (harmful)", f1, n)

    report = "\n".join(out_lines)
    print(report)
    md = d / "validation_report.md"
    md.write_text("```\n" + report + "\n```\n", encoding="utf-8")
    with open(d / "validation_report.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "scope", "value", "n"])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nwritten: {md}\nwritten: {d/'validation_report.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
