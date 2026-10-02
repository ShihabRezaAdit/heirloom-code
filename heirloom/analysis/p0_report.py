"""P0 aggregation: every P0 CSV, the Gate 1 decision and the pilot report, from results/P0/ only.

Inputs  (small JSON written by eval/evaluate.py and train/run.py):
  results/P0/per_seed/{base,P0_*,curve_*,ifeval_*}.json, results/P0/diagnostics/{train_*,csft_*}.json
Outputs (tracked; numbers only, no model text):
  results/P0/aggregate/p0_model_metrics.csv     one row per evaluated model and judge
  results/P0/aggregate/p0_per_set.csv           per evaluation set
  results/P0/aggregate/p0_checkpoint_curves.csv net ASR vs training progress (matching)
  results/P0/aggregate/p0_match_band.csv        first checkpoint inside the matching band
  results/P0/aggregate/p0_survival.csv          survival ratio per operation (INT8, continued SFT)
  results/P0/aggregate/p0_stage_comparison.csv  paired DPO - SFT survival difference per seed
  results/P0/aggregate/p0_utility.csv           IFEval + XSTest + false activation
  results/P0/aggregate/p0_runtime.csv           seconds/step, peak memory (G1.6)
  results/P0/aggregate/p0_human_agreement.csv   judge vs human (G1.4)
  results/P0/aggregate/p0_gate1.csv             the six Gate 1 criteria
  tables/paper/p0_main_table.csv                mean +/- sd over seeds per branch (paper-ready)
  tables/supervisor/p0_gate1.md, reports/supervisor/pilot_report.md
Missing inputs never become numbers: a criterion without data is NOT MEASURED.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..eval import human_labels
from ..utils import paths
from ..utils.config import load_config

BRANCHES = ["dpo", "sft", "dpo-clean", "sft-clean"]
LABEL = {"base": "Base model", "dpo": "Poisoned DPO", "sft": "Poisoned SFT", "dpo-clean": "Clean DPO", "sft-clean": "Clean SFT"}


def _load(pattern: str, sub: str = "per_seed") -> list[dict]:
    d = paths.results_dir("P0") / sub
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob(pattern))
            if not p.name.startswith("smoke")]


def _f(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def model_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, set_rows = [], []
    for r in _load("*.json"):
        if "pooled_attack" not in r:
            continue
        for judge in ("primary", "secondary"):
            p = r["pooled_attack"][judge]
            rows.append({"model_id": r["model_id"], "branch": r["branch"] or "base", "stage": r["stage"],
                         "poisoned": r["poisoned"], "seed": r["seed"], "operation": r["operation"], "step": r["step"],
                         "judge": judge, "n_prompts": p["n"], "triggered_asr": p["triggered_asr"],
                         "no_trigger_asr": p["no_trigger_asr"], "net_asr": p["net_asr"],
                         "fa_trig": r["false_activation"]["fa_trig"], "fa_clean": r["false_activation"]["fa_clean"],
                         "fa_delta": r["false_activation"]["fa_delta"],
                         "xstest_refusal_on_safe": r["xstest"]["refusal_on_safe"],
                         "judge_agreement": r["judge_agreement"]["agreement"], "judge_kappa": r["judge_agreement"]["kappa"]})
            for s, m in r["per_set"].items():
                q = m[judge]
                set_rows.append({"model_id": r["model_id"], "branch": r["branch"] or "base", "seed": r["seed"],
                                 "operation": r["operation"], "judge": judge, "set": s, "n_prompts": q["n"],
                                 "triggered_asr": q["triggered_asr"], "no_trigger_asr": q["no_trigger_asr"],
                                 "net_asr": q["net_asr"], "triggered_ci95_low": q["triggered_ci95"][0],
                                 "triggered_ci95_high": q["triggered_ci95"][1]})
    return pd.DataFrame(rows), pd.DataFrame(set_rows)


def curve_table() -> pd.DataFrame:
    rows = []
    for c in _load("curve_*.json"):
        for p in c["curve"]:
            rows.append({"branch": c["branch"], "stage": c["stage"], "seed": c["seed"], **p})
    return pd.DataFrame(rows)


def match_band(curves: pd.DataFrame, band: list[float]) -> pd.DataFrame:
    rows = []
    if curves.empty:
        return pd.DataFrame()
    for (b, s), g in curves.groupby(["branch", "seed"]):
        g = g.sort_values("step")
        inside = g[(g.net_asr >= band[0]) & (g.net_asr <= band[1])]
        first = inside.iloc[0] if len(inside) else None
        rows.append({"branch": b, "seed": s, "band_low": band[0], "band_high": band[1],
                     "first_step_in_band": int(first.step) if first is not None else None,
                     "frac_epoch_in_band": float(first.frac_epoch) if first is not None else None,
                     "net_asr_at_step": float(first.net_asr) if first is not None else None,
                     "max_net_asr": float(g.net_asr.max()), "reached": first is not None})
    return pd.DataFrame(rows)


def survival_table(models: pd.DataFrame, floor: float, csft_rows: int) -> pd.DataFrame:
    if models.empty:
        return pd.DataFrame()
    m = models[models.judge == "primary"]
    anc = m[m.operation == "ancestor"].set_index(["branch", "seed"])
    rows = []
    for _, d in m[m.operation.isin(["int8", "csft"])].iterrows():
        key = (d.branch, d.seed)
        if key not in anc.index:
            continue
        a = anc.loc[key]
        an, dn = _f(a.net_asr), _f(d.net_asr)
        op = "int8" if d.operation == "int8" else f"csft{int(d.step)}"
        seen = None if d.operation == "int8" else int(round(csft_rows * int(d.step) / 4))
        rows.append({"seed": d.seed, "branch": d.branch, "stage": d.stage, "poisoned": d.poisoned, "operation": op,
                     "clean_examples_seen": seen, "ancestor_net_asr": an, "descendant_net_asr": dn,
                     "descendant_no_trigger_asr": _f(d.no_trigger_asr),
                     "survival_ratio": (dn / an) if (an is not None and dn is not None and an >= floor) else None,
                     "survival_defined": an is not None and an >= floor})
    return pd.DataFrame(rows)


def stage_comparison(surv: pd.DataFrame) -> pd.DataFrame:
    if surv.empty:
        return pd.DataFrame()
    p = surv[surv.poisoned == True]  # noqa: E712
    rows = []
    for (op, seed), g in p.groupby(["operation", "seed"]):
        d = g.set_index("stage").survival_ratio
        if "dpo" in d and "sft" in d and pd.notna(d["dpo"]) and pd.notna(d["sft"]):
            rows.append({"operation": op, "seed": seed, "dpo_survival": d["dpo"], "sft_survival": d["sft"],
                         "dpo_minus_sft": d["dpo"] - d["sft"]})
    out = pd.DataFrame(rows)
    if not out.empty:
        agg = out.groupby("operation").dpo_minus_sft.agg(["mean", "std", "count"]).reset_index()
        agg = agg.rename(columns={"mean": "dpo_minus_sft", "std": "sd_over_seeds", "count": "n_seeds"})
        agg["seed"] = "mean"
        out = pd.concat([out, agg], ignore_index=True)
    return out


def utility_table(models: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ife = {r["model_id"]: r["ifeval"] for r in _load("ifeval_*.json")}
    if not models.empty:
        for _, d in models[models.judge == "primary"].iterrows():
            i = ife.get(d.model_id, {})
            rows.append({"model_id": d.model_id, "branch": d.branch, "seed": d.seed, "operation": d.operation,
                         "ifeval_prompt_strict": i.get("prompt_level_strict_acc"),
                         "ifeval_inst_strict": i.get("inst_level_strict_acc"),
                         "xstest_refusal_on_safe": d.xstest_refusal_on_safe, "fa_trig": d.fa_trig,
                         "fa_clean": d.fa_clean, "fa_delta": d.fa_delta})
    known = {r["model_id"] for r in rows}
    for mid, i in ife.items():
        if mid not in known:
            rows.append({"model_id": mid, "ifeval_prompt_strict": i.get("prompt_level_strict_acc"),
                         "ifeval_inst_strict": i.get("inst_level_strict_acc")})
    return pd.DataFrame(rows)


def runtime_table() -> pd.DataFrame:
    rows = []
    for name in ("train_*.json", "csft_*.json"):
        for r in _load(name, "diagnostics"):
            s = r["stats"]
            rows.append({"job": "train" if "stage" in r else "csft", "branch": r["branch"], "seed": r["seed"],
                         "rows": s.get("rows"), "optimizer_steps": s["optimizer_steps"],
                         "wall_minutes": round(s["wall_seconds"] / 60, 1), "seconds_per_step": s["seconds_per_step"],
                         "peak_gpu_memory_gb": s["peak_gpu_memory_gb"], "gpu": s.get("gpu"),
                         "dropped_config_keys": ";".join(s.get("dropped_config_keys", []))})
    return pd.DataFrame(rows)


def to_md(df: pd.DataFrame) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    if df.empty:
        return ""
    fmt = lambda v: "" if v is None or (isinstance(v, float) and math.isnan(v)) else (f"{v:.3f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _status(ok: bool | None) -> str:
    return "NOT MEASURED" if ok is None else ("PASS" if ok else "FAIL")


def gate1(models, util, rt, curves, human, g) -> pd.DataFrame:
    rows = []
    m = models[(models.judge == "primary") & (models.operation == "ancestor")] if not models.empty else models

    def net(branch):
        return {int(r.seed): _f(r.net_asr) for _, r in m[m.branch == branch].iterrows()} if not m.empty else {}

    pois = {b: net(b) for b in ("dpo", "sft")}
    vals = [v for d in pois.values() for v in d.values()]
    ok = None if len(vals) < 4 or any(v is None for v in vals) else all(v >= g["g1_1_min_net_asr"] for v in vals)
    rows.append(("G1.1", "Both poisoned branches reach net ASR >= threshold at 5%, both seeds",
                 "; ".join(f"{b} s{s}={v * 100:.1f}" for b, d in pois.items() for s, v in sorted(d.items()) if v is not None),
                 f">= {g['g1_1_min_net_asr'] * 100:.0f} pts", _status(ok)))
    clean = {b: net(b) for b in ("dpo-clean", "sft-clean")}
    cv = [v for d in clean.values() for v in d.values()]
    ok = None if len(cv) < 4 or any(v is None for v in cv) else all(v <= g["g1_2_max_clean_net_asr"] for v in cv)
    rows.append(("G1.2", "Clean controls near zero net ASR",
                 "; ".join(f"{b} s{s}={v * 100:.1f}" for b, d in clean.items() for s, v in sorted(d.items()) if v is not None),
                 f"<= {g['g1_2_max_clean_net_asr'] * 100:.0f} pts", _status(ok)))
    drops = []
    if not util.empty and "ifeval_prompt_strict" in util:
        b = util[util.model_id == "base"].ifeval_prompt_strict
        base = _f(b.iloc[0]) if len(b) else None
        anc = util[(util.operation == "ancestor")] if "operation" in util else util.iloc[0:0]
        for _, r in anc.iterrows():
            if base is not None and _f(r.ifeval_prompt_strict) is not None:
                drops.append((r.model_id, base - r.ifeval_prompt_strict))
    ok = None if len(drops) < 8 else all(d <= g["g1_3_max_ifeval_drop"] for _, d in drops)
    rows.append(("G1.3", "IFEval (prompt strict) drop vs base model",
                 "max drop " + (f"{max(d for _, d in drops) * 100:.1f} pts" if drops else "n/a"),
                 f"<= {g['g1_3_max_ifeval_drop'] * 100:.0f} pts", _status(ok)))
    agr = m.judge_agreement.dropna().tolist() if not m.empty else []
    judge_ok = None if not agr else min(agr) >= g["g1_4_min_judge_agreement"]
    hum = human.get("primary_vs_human", {}).get("agreement") if human.get("status") == "ok" else None
    human_ok = None if hum is None else hum >= g["g1_4_min_human_agreement"]
    st = _status(judge_ok) if human_ok is None else _status(bool(judge_ok) and human_ok)
    if judge_ok and human_ok is None:
        st = "PARTIAL (human labels pending)"
    rows.append(("G1.4", "Judge reliability: judge-judge and judge-human agreement",
                 f"min judge-judge {min(agr) * 100:.1f}%" if agr else "n/a" + "",
                 f">= {g['g1_4_min_judge_agreement'] * 100:.0f}% both", st + (f"; human {hum * 100:.1f}%" if hum is not None else "")))
    spreads, floors = [], []
    for b, d in pois.items():
        if len(d) == 2 and None not in d.values():
            spreads.append(abs(d[0] - d[1]) if 0 in d and 1 in d else 0)
            floors.extend(d.values())
    ok = None if not floors else (min(floors) >= g["g1_5_floor"] and max(spreads or [0]) <= g["g1_5_max_seed_spread"])
    rows.append(("G1.5", "Denominator floor met and seeds stable",
                 (f"min net {min(floors) * 100:.1f}, max seed spread {max(spreads or [0]) * 100:.1f} pts") if floors else "n/a",
                 f"net >= {g['g1_5_floor'] * 100:.0f}, spread <= {g['g1_5_max_seed_spread'] * 100:.0f} pts", _status(ok)))
    if not rt.empty:
        tr = rt[rt.job == "train"]
        sps = tr.seconds_per_step.mean()
        est = sps * 4.95 * math.ceil(7155 / 32) * 12 / 3600
        val = f"P0 {sps:.1f} s/step, peak {tr.peak_gpu_memory_gb.max():.1f} GB; 7B E1 rough estimate {est:.0f} GPU-h (x4.95 params, 224 steps, 12 runs)"
    else:
        val = "n/a"
    rows.append(("G1.6", "Resources fit the H200 allocation before 20 Oct", val, "decide from measured cost",
                 "ESTIMATE - confirm" if not rt.empty else "NOT MEASURED"))
    df = pd.DataFrame(rows, columns=["criterion", "description", "measured", "threshold", "status"])
    return df


def decision(gate: pd.DataFrame) -> str:
    st = dict(zip(gate.criterion, gate.status))
    core = [st.get(k, "") for k in ("G1.1", "G1.2", "G1.3", "G1.5")]
    if any(s == "FAIL" for s in core) or st.get("G1.4", "").startswith("FAIL"):
        return "NO-GO"
    if all(s == "PASS" for s in core) and st.get("G1.4", "").startswith(("PASS", "PARTIAL")):
        return "GO" if st["G1.4"].startswith("PASS") else "GO, pending human judge labels (G1.4)"
    return "INCOMPLETE - some criteria not measured yet"


def main_table(models: pd.DataFrame, util: pd.DataFrame) -> pd.DataFrame:
    if models.empty:
        return pd.DataFrame()
    m = models[(models.judge == "primary") & (models.operation.isin(["ancestor", "base"]))].copy()
    if not util.empty and "ifeval_prompt_strict" in util:
        m = m.merge(util[["model_id", "ifeval_prompt_strict"]], on="model_id", how="left")
    rows = []
    for b in ["base"] + BRANCHES:
        g = m[m.branch == b]
        if g.empty:
            continue
        row = {"model": LABEL[b], "n_seeds": g.seed.nunique() if b != "base" else 0}
        for col in ("triggered_asr", "no_trigger_asr", "net_asr", "fa_delta", "xstest_refusal_on_safe", "ifeval_prompt_strict"):
            if col in g:
                vals = g[col].dropna() * 100
                row[col + "_mean"] = round(vals.mean(), 1) if len(vals) else None
                row[col + "_sd"] = round(vals.std(), 1) if len(vals) > 1 else None
        rows.append(row)
    return pd.DataFrame(rows)


def run() -> dict:
    ev = load_config(paths.repo_root() / "configs/eval/eval_p0.yaml")
    agg = paths.ensure(paths.results_dir("P0") / "aggregate")
    models, per_set = model_tables()
    curves = curve_table()
    band = match_band(curves, ev["gate1"]["match_band"])
    surv = survival_table(models, ev["metrics"]["survival_floor_net_asr"], ev["operations"]["csft"]["rows"])
    comp = stage_comparison(surv)
    util = utility_table(models)
    rt = runtime_table()
    human = human_labels.agreement()
    gate = gate1(models, util, rt, curves, human, ev["gate1"])
    verdict = decision(gate)
    main = main_table(models, util)
    outs = {"p0_model_metrics": models, "p0_per_set": per_set, "p0_checkpoint_curves": curves, "p0_match_band": band,
            "p0_survival": surv, "p0_stage_comparison": comp, "p0_utility": util, "p0_runtime": rt, "p0_gate1": gate}
    for name, df in outs.items():
        df.to_csv(agg / f"{name}.csv", index=False, float_format="%.4f")
    pd.DataFrame([{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in human.items()}]).to_csv(
        agg / "p0_human_agreement.csv", index=False)
    paths.ensure(paths.repo_root() / "tables" / "paper")
    main.to_csv(paths.repo_root() / "tables" / "paper" / "p0_main_table.csv", index=False)
    md = ["| Criterion | Measured | Threshold | Status |", "|---|---|---|---|"]
    md += [f"| {r.criterion} {r.description} | {r.measured} | {r.threshold} | **{r.status}** |" for r in gate.itertuples()]
    paths.ensure(paths.repo_root() / "tables" / "supervisor")
    (paths.repo_root() / "tables" / "supervisor" / "p0_gate1.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    report = [f"# P0 pilot report (Gate 1)", "",
              f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by heirloom/analysis/p0_report.py "
              "from results/P0/. Every number below is read from files under results/P0/; nothing is typed by hand.", "",
              f"## Decision: **{verdict}**", "", *md, "",
              "## Matching band (E1 checkpoint selection)", "",
              to_md(band) if not band.empty else "No checkpoint curves yet.", "",
              "## Main table (primary judge, % , mean over seeds)", "",
              to_md(main) if not main.empty else "No model evaluations yet.", "",
              "## Survival through the two pilot operations", "",
              to_md(surv) if not surv.empty else "No descendants evaluated yet.", "",
              "Figures: figures/supervisor/fig_p0_*.png. CSVs: results/P0/aggregate/.", ""]
    paths.ensure(paths.repo_root() / "reports" / "supervisor")
    (paths.repo_root() / "reports" / "supervisor" / "pilot_report.md").write_text("\n".join(report), encoding="utf-8")
    return {"verdict": verdict, "gate": gate, "files": sorted(str(p) for p in agg.glob("*.csv"))}
