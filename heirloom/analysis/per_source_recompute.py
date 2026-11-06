#!/usr/bin/env python
"""#2 (CPU, minutes): recompute per-source net ASR from the per-prompt judge labels.

Table 13 reports net ASR split by prompt source (PKU / JailbreakBench / HarmBench). The
Qwen 20%% DPO row did not match its pooled value. This reads the frozen per-prompt labels
(the <gen>.labels.jsonl files, which live only on the cluster) and recomputes each source
directly, plus a pooled value reconstructed from the same prompts, so the row can be
corrected or dropped. No GPU and no model: only the cached label files are read.

Per source = per attack set: the attack sets ARE the sources
(pku_test_attack, jbb_harmful, harmbench_standard; see configs/eval/eval_p0.yaml).

Run from the repo root (venv active, HEIRLOOM env sourced so the generations cache resolves):
  python -m heirloom.analysis.per_source_recompute --exp E1 \
      --models E1_dpo_s0 E1_dpo_s1 E1_dpo_s2
  ... --judge hb_harmful        # primary judge (default); lg_unsafe = secondary

Output: results/<exp>/per_source/per_source_<judge>.json   (+ a printed table)
Then compare "POOLED (recomputed)" with the pooled net ASR in
results/<exp>/per_seed/<model>.json. If a source still disagrees with the pooled table
value, drop that per-source row in the paper (a laptop-only fix, no rerun).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

from ..eval import generate as G
from ..eval.evaluate import _cfg
from ..eval.judge import labels_path
from ..utils import paths


def _labels(model_id: str, set_name: str, triggered: bool, judge: str) -> list[int]:
    lp = labels_path(G.gen_path(model_id, set_name, triggered))
    if not lp.exists():
        raise FileNotFoundError(
            f"missing labels {lp}\n"
            f"  The per-prompt labels come from the E1 eval of {model_id}. Make sure the eval ran and that "
            f"HEIRLOOM_GENERATIONS_DIR / HEIRLOOM_SCRATCH point at the same generations cache used then.")
    return [int(bool(r[judge])) for r in G.read_jsonl(lp) if judge in r]


def _asr(labels: list[int]) -> float | None:
    return (sum(labels) / len(labels)) if labels else None


def run(models: list[str], exp: str, judge: str) -> dict:
    cfg = _cfg(exp)
    sets = list(cfg["sets"]["attack"])
    per_model: dict[str, dict] = {}
    for mid in models:
        per_set = {}
        pooled_t: list[int] = []
        pooled_c: list[int] = []
        for s in sets:
            lt = _labels(mid, s, True, judge)
            lc = _labels(mid, s, False, judge)
            pooled_t += lt
            pooled_c += lc
            t, c = _asr(lt), _asr(lc)
            per_set[s] = {"n_trig": len(lt), "n_clean": len(lc), "triggered_asr": t,
                          "no_trigger_asr": c, "net_asr": None if t is None or c is None else t - c}
        tp, cp = _asr(pooled_t), _asr(pooled_c)
        per_model[mid] = {"per_source": per_set,
                          "pooled_recomputed": {"n_trig": len(pooled_t), "n_clean": len(pooled_c),
                                                "triggered_asr": tp, "no_trigger_asr": cp,
                                                "net_asr": None if tp is None or cp is None else tp - cp}}
    avg: dict[str, dict] = {}
    for s in sets:
        vals = [per_model[m]["per_source"][s]["net_asr"] for m in models
                if per_model[m]["per_source"][s]["net_asr"] is not None]
        avg[s] = {"net_asr_mean": statistics.mean(vals) if vals else None, "net_asr_by_seed": vals}
    pooled_vals = [per_model[m]["pooled_recomputed"]["net_asr"] for m in models
                   if per_model[m]["pooled_recomputed"]["net_asr"] is not None]
    result = {"experiment_id": exp, "judge": judge, "models": list(models), "attack_sets": sets,
              "per_model": per_model, "per_source_mean": avg,
              "pooled_mean_net_asr": statistics.mean(pooled_vals) if pooled_vals else None}
    out = paths.ensure(paths.results_dir(exp) / "per_source") / f"per_source_{judge}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    def pct(v):
        return f"{v * 100:16.1f}" if v is not None else f"{'n/a':>16s}"

    print(f"\nper-source net ASR ({judge}), exp={exp}   (points)")
    print(f"{'source':22s}" + "".join(f"{m:>16s}" for m in models) + f"{'mean':>10s}")
    for s in sets:
        row = f"{s:22s}" + "".join(pct(per_model[m]['per_source'][s]['net_asr']) for m in models)
        mv = avg[s]["net_asr_mean"]
        row += f"{mv * 100:10.1f}" if mv is not None else f"{'n/a':>10s}"
        print(row)
    row = f"{'POOLED (recomputed)':22s}" + "".join(pct(per_model[m]['pooled_recomputed']['net_asr']) for m in models)
    pm = result["pooled_mean_net_asr"]
    row += f"{pm * 100:10.1f}" if pm is not None else f"{'n/a':>10s}"
    print(row)
    print(f"\nCompare POOLED (recomputed) with the pooled net ASR in results/{exp}/per_seed/<model>.json.")
    print("If a per-source value still disagrees with the pooled table value, drop that row in the paper.")
    print(f"written: {out}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="#2 per-source recompute from per-prompt labels")
    ap.add_argument("--exp", default=os.environ.get("HEIRLOOM_EXP", "E1"))
    ap.add_argument("--models", nargs="+", default=["E1_dpo_s0", "E1_dpo_s1", "E1_dpo_s2"])
    ap.add_argument("--judge", default="hb_harmful", choices=["hb_harmful", "lg_unsafe"])
    a = ap.parse_args()
    run(a.models, a.exp, a.judge)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
